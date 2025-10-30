from functools import partial
import jax
import time

from jaxtyping import Array
from typing import NamedTuple, Any
import jax.numpy as jnp
import numpy as np

from watchers import marketenv_stats, marketenv_eval_stats
from utils import TrainingState, MemoryState
import wandb
import optax
from environment.wrappers import (
    DummyDoubleVecObsWrapper,
    NormalizeDoubleVecObservation,
    NormalizeDoubleVecReward,
    DummyDoubleVecRewWrapper,
)

# region RUNNER
MAX_WANDB_CALLS = 1000


class Sample(NamedTuple):
    """Object containing a batch of data
    In practice, we fill them to be [num_outer, num_inner, num_opps, num_envs, ...]"""

    observations: Array
    actions: Array
    rewards: Array
    behavior_log_probs: Array
    behavior_values: Array
    dones: Array
    hiddens: Array
    unnormalized_rewards: Array


class EvalSample(NamedTuple):
    """Object containing a batch of data
    In practice, we fill them to be [T, n_o, n_e]"""

    observations: Array
    actions: Array
    rewards_rescaled: Array
    rewards_unnormalized: Array
    extras: Array


@jax.jit
def reduce_outer_traj(traj: Sample) -> Sample:
    """Used to collapse lax.scan outputs dims"""
    # x: [outer_loop, inner_loop, num_opps, num_envs ...]
    # x: [timestep=n_outer*n_inner, batch_size=num_opps*num_envs, ...]
    num_envs = traj.rewards.shape[2] * traj.rewards.shape[3]  # n_opps * n_envs
    num_timesteps = traj.rewards.shape[0] * traj.rewards.shape[1]  # n_outer * n_inner
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_timesteps, num_envs) + x.shape[4:]),
        traj,
    )


class TwoAgentGridsearchRunner:
    """
    This class runs two agents in our market environment. It takes in agents and the environment, and runs the training loop.
    Args:
        agents (Tuple[agents]):
            Set of agents to run in the environment. Order doesn't matter,
            both train at the same points in time and actions are chosen simultaneously.
        env (MarketEnv):
            Environment in which agents are run.
        save_dir (string):
            Directory in which to save the model.
        args (NamedTuple):
            A tuple of experiment arguments used (later on we'll use Hydra to do the configuration).
    """

    def __init__(self, agents, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args["num_opps"]
        self.random_key = jax.random.PRNGKey(args["seed"])
        self.save_dir = save_dir
        self.competitive_profits = jnp.array(args["competitive_profits"])
        self.competitive_profits_episodetotal = jnp.array(args["competitive_profits_episodetotal"])
        self.collusive_profits = jnp.array(args["collusive_profits"])
        self.collusive_profits_episodetotal = jnp.array(args["collusive_profits_episodetotal"])

        # we don't use num_opps (yet) as it's an opponent shaping feature.
        # but since I'm borrowing a lot of code from them, the observations have a num_opps dimension (always 1 but still)
        def _reshape_opp_dim(x):
            # input x: [num_opps, num_envs ...]
            # returns x: [batch_size = n_o*n_e, ...]
            batch_size = args["num_envs"] * args["num_opps"]
            return jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), x)

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        self.marketenv_stats = jax.jit(marketenv_stats)
        self.marketenv_eval_stats = jax.jit(marketenv_eval_stats)

        ## We VMAP the environment and random functions to vmap over 2 extra dims: num_opponents and num_envs.
        ## Their un-vmapped versions expect arrays of size [...].
        # VMAP the environment twice: for num_opps and num_envs. Expects rng: [n_o(!), n_e(!), keydim] and params [...]. Returns all_obs, state: [n_o(!), n_e(!), ...]
        env.batch_reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=0)
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, in_axes=(0, None), out_axes=0))
        # VMAP the step twice: for num_opps and num_envs. Expects for rng, state, actions: [n_o(!), n_e(!), ...] and params [...]. Returns obs, env_state, rewards, done, info: [n_o(!), n_e(!),...]
        env.batch_step = jax.vmap(
            env.step, in_axes=(0, 0, 0, None), out_axes=0
        )  # vmap num_envs. rng (+), state (+), actions (+), params (-) -> all outputs
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step, in_axes=(0, 0, 0, None), out_axes=0)
        )  # vmap num_opps.

        if args["normalize_obs"]:
            env = NormalizeDoubleVecObservation(env)
        else:
            env = DummyDoubleVecObsWrapper(env)

        # this overwrites the batch_step and batch_reset methods to rescale & clip rewards (if wanted)
        if args["normalize_rewards_wrapper"]:
            if args["clip_rewards"]:
                env = NormalizeDoubleVecReward(
                    env,
                    gamma=args["normalizing_rewards_gamma"],
                    clip=args["reward_clipping_limit"],
                )
            else:
                env = NormalizeDoubleVecReward(
                    env, gamma=args["normalizing_rewards_gamma"], clip=1000000
                )
        else:
            env = DummyDoubleVecRewWrapper(
                env
            )  # wrapper that does nothing except make state a special obj so metrics don't break

        # VMAP split twice: for num_opps and num_envs. Expects for rng: [n_o(!), n_e(!), keydim]. Returns split key [n_o(!), n_e(!), splits, keydim]
        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        # number of times rollout_outer is called (#episodes played)
        num_outer_steps = self.args["num_outer_steps"]

        agent1, agent2 = agents

        ## Here we set up agents
        ## Agent 1 batches differently from the other agents (who could have n_o many copies of themselves):
        # Ag1 can play vs n_o many copies of ag2. So ag2 needs to remember which copy it is, each one has their own trajectories, actions, TS, MS,
        # so all these are batched with [n_o(+), …] and all functions interacting with them are batched in n_o.
        # Ag1 only needs to remember who they're playing against (traj, actions, MS), but their TrainingState
        # is the same vs each opponent, so TS is NOT batched.
        ## The update functions collect (obs, action, next_obs, reward) from each env and treat them as one big batch to learn from.
        agent1.batch_init = jax.vmap(
            agent1.make_initial_state, in_axes=(None, 0), out_axes=(None, 0)
        )  # vmap: num_opps. rng(-), hidden(+) -> TrainState(-), MemState(+)
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )  # vmap: num_opps. MemState(+), eval(-) -> MemState(+)
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, in_axes=(None, 0, 0), out_axes=(0, None, 0))
        )  # vmap: num_opps. TrainState(-), obs(+), MemState(+) -> actions(+), TrainState(-), MemState(+)

        agent1.batch_eval_policy = jax.jit(
            jax.vmap(agent1._eval_policy, in_axes=(None, 0, 0), out_axes=(0, None, 0, 0))
        )  # vmap: num_opps. TrainState(-), obs(+), MemState(+) -> actions(+), TrainState(-), MemState(+), extras (+)

        agent2.batch_init = jax.vmap(
            agent2.make_initial_state,
            (0, None),
            0,  # in the init below it gets key [n_o(!)] hidden [n_o, n_e, 1] (doesn't matter) ->
        )  # vmap num_opps. rng(+), hidden(-) -> TrainState(+), MemState(+)
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )  # vmap: num_opps. MemState(+), eval(-) -> MemState(+)
        # Policy: Ag2 DOES batch TrainingState over num_opponents. Reason: ag1 can play vs n_o copies of ag2, so ag2 must keep track of n_o many copies of itself each with their own TS, MS.
        agent2.batch_policy = jax.jit(
            jax.vmap(agent2._policy)
        )  # vmap: num_opps. TrainState(+), obs(+), MemState(+) -> actions(+), TrainState(+), MemState(+)

        agent2.batch_eval_policy = jax.jit(
            jax.vmap(agent2._eval_policy)
        )  # vmap: num_opps. TrainState(-), obs(+), MemState(+) -> actions(+), TrainState(-), MemState(+), extras (+)

        agent2.batch_update = jax.jit(
            jax.vmap(agent2.update, (1, 0, 0, 0), 0)
        )  # vmap: num_opps. trajectory(+, [n_inner, n_opps(!!), n_envs, ...]), obs(+), TrainState(+), MemState(+) -> TrainState(+), MemState(+), metrics(+)

        ## init agents. _mem and _state are result of make_initial_state(key, zeros(1)) so not batched. have shape Train [...] Mem [num_envs, ...]
        init_hidden1 = jnp.tile(
            agent1._mem.hidden, (args["num_opps"], 1, 1)
        )  # _mem.hidden has shape [n_envs, 1] then init_hidden has shape: [num_opps, num_envs, 1]
        agent1._state, agent1._mem = agent1.batch_init(
            agent1._state.random_key, init_hidden1
        )  # vmapped over 2nd arg only, OK. IN: [], [num_opps (!), num_envs, 1] (doesn't matter) -> _state [...], _mem [num_opps(!), num_envs, ...]

        init_hidden2 = jnp.tile(
            agent2._mem.hidden, (args["num_opps"], 1, 1)
        )  # _mem.hidden has shape [n_envs, 1] then init_hidden has shape: [num_opps, num_envs, 1]
        a2_rng = jax.random.split(
            agent2._state.random_key, args["num_opps"]
        )  # get array of keys [num_opps,]
        agent2._state, agent2._mem = agent2.batch_init(
            a2_rng,
            init_hidden2,
        )  # vmapped over 1st arg only. IN: [num_opps (!),], [num_opps, num_envs, 1] -> OUT: _state [num_opps(!), ...], _mem [num_opps(!), num_envs, ...]

        def _inner_rollout(carry, unused):
            """Plays 1 step, gets scanned over to produce 1 episode
            carry: rngs, obs, rewards, agent and env states"""
            (
                rngs,  # [n_o, n_e, keydim] uint32
                obs1,  # [n_o, n_e, ...] int32
                obs2,  # [n_o, n_e, ...] int32
                # r1,  # [n_o, n_e], float32 (one float per oppo&env)
                # r2,  # [n_o, n_e], float32
                a1_state,  # params: [...], keys: [keydim], timesteps: int (unbatched)
                a1_mem,  # [n_o, n_e, ...]
                a2_state,  # params: [n_o, ...], keys: [n_o, keydim], timesteps: [n_o]
                a2_mem,  # [n_o, n_e, ...]
                env_state,  # [n_o, n_e, X] (X= invs: 2, prices: 2, t: 1)
                env_params,  # unbatched
            ) = carry
            # unpack rngs
            rngs = self.split(
                rngs, 4
            )  # vmapped split. IN: keys [num_opps(!), num_envs(!), ] -> OUT: [num_opps(!), num_envs(!), 4]
            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :] # UNUSED!
            # a2_rng = rngs[:, :, 2, :] # UNUSED!
            passthrough_rngs = rngs[:, :, 3, :]

            # get actions for both agents, as each policy expects [batch_size, ...] we use a version that's vmapped once
            # this way the batched version sees [num_opps(!), num_envs,...], so each parallel version sees the expected [num_envs,...]

            # ag1: TrainState is NOT batched (in&out)!
            a1, a1_state, new_a1_mem = agent1.batch_policy(a1_state, obs1, a1_mem)

            # ag2: all in & outs are batched in dim0 (num_opps)
            a2, a2_state, new_a2_mem = agent2.batch_policy(a2_state, obs2, a2_mem)

            (
                (next_obs1, next_obs2),
                env_state,
                rewards,
                unnormalized_rewards,
                done,
                info,
            ) = env.batch_step(
                env_rng,
                env_state,
                (a1, a2),
                env_params,
            )

            # added this: squeeze done to remove trailing 1-dim (think this is needed?)
            done = jnp.squeeze(done, axis=-1)  # [n_o, n_e, 1] -> [n_o, n_e]

            # rewards
            r1 = rewards[:, :, 0]  # /jnp.std(rewards[:,:,0])  # [n_o, n_e]
            r2 = rewards[:, :, 1]  # /jnp.std(rewards[:,:,0])  # [n_o, n_e]

            if args["normalize_rewards_wrapper"]:
                r1_unnormalized = unnormalized_rewards[:, :, 0]  # .mean()
                r2_unnormalized = unnormalized_rewards[:, :, 1]  # .mean()
            else:
                r1_unnormalized = r1
                r2_unnormalized = r2

            ## manual reward rescaling attempts (rather use the normalizing wrapper above controlled via config)
            # r1_rescaled = r1 / r1.std()
            # r2_rescaled = r2 / r2.std()
            # r1_rescaled = jnp.log(r1)
            # r2_rescaled = jnp.log(r2)  # issue that lower rew's are closer together...
            if args["normalize_rewards_manually"]:
                r1_rescaled = (r1 - args["normalizing_rewards_min"]) / (
                    args["normalizing_rewards_max"] - args["normalizing_rewards_min"]
                )
                r2_rescaled = (r2 - args["normalizing_rewards_min"]) / (
                    args["normalizing_rewards_max"] - args["normalizing_rewards_min"]
                )
            else:
                r1_rescaled = r1
                r2_rescaled = r2

            # save trajectories as Samples (these go into updates)
            traj1 = Sample(
                obs1,
                a1,
                r1_rescaled,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
                r1_unnormalized,
            )
            traj2 = Sample(
                obs2,
                a2,
                r2_rescaled,
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
                r2_unnormalized,
            )

            # return vals, trajectories. traj's get stacked by the scan s.t. they're [n_inner, num_opps, num_envs, ...]
            return (
                passthrough_rngs,
                next_obs1,
                next_obs2,
                # r1,
                # r2,
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
                env_params,
            ), (traj1, traj2, env_state, info)

        def _outer_rollout(carry, unused):
            """Plays 1 episode, trains agent 2, gets scanned over to produce n_outer episodes
            carry: rngs, agent states, env states"""
            # Sim 1 episode. Scan compiles _inner_rollout, so jax.jit not needed.
            vals, trajectories = jax.lax.scan(_inner_rollout, carry, None, args["num_inner_steps"])

            # unpack vals (last state of episode)
            (
                rngs,  # [num_opps, num_envs, keydim]
                obs1,  # [num_opps, num_envs, ...] result of env.batch_step
                obs2,  # [num_opps, num_envs, ...] result of env.batch_step
                # r1,  # [num_opps(!), num_envs(!)] result of env.batch_step
                # r2,  # [num_opps(!), num_envs(!)] result of env.batch_step
                a1_state,  # params [...], keys [keydim], timesteps [] result of agent1.batch_policy
                a1_mem,  # [n_o, n_e, ...] result of agent1.batch_policy
                a2_state,  # [num_opps(!), ...] result of agent2.batch_policy
                a2_mem,  # [num_opps(!), num_envs, ...] result of agent2.batch_policy
                env_state,  # [num_opps, num_envs, X] (X= invs: 2 int32, prices: 2 float32, actions: 2 int32, t: 1 int16) result of env.batch_step
                env_params,  # passthrough, unbatched
            ) = vals

            # jax.debug.breakpoint()
            # update second agent. this only needs "rewards" and "dones" inside the trajectory
            # -> whether env passes reset_state, reset_obs or the actual state for terminal states doesn't make a difference here
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[
                    1
                ],  # traj_2 (stacked, class Sample): [n_inner, num_opps(!), num_envs, ...], so slice sees [n_inner, num_envs, ...]
                obs2,  # [num_opps(!), num_envs, ...] from env.batch_step
                a2_state,  # [num_opps(!), ...]
                a2_mem,  # [num_opps(!), num_envs, ...]
            )  # all outputs are [num_opps(!), ...].

            # vals: last state of last episode. trajectories: all trajectories of all episodes [n_outer, n_inner, n_opps, n_envs, ...]
            return (
                rngs,
                obs1,
                obs2,
                # r1,
                # r2,
                a1_state,
                a1_mem,
                a2_state,  # [num_opps, ...]
                a2_mem,  # [num_opps, ...]
                env_state,
                env_params,
            ), (*trajectories, a2_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _a2_mem: MemoryState,
            _env_params: Any,
        ):
            """"""
            # generate vmapped RNG:
            rngs_split_envs = jax.random.split(_rng_run, args["num_envs"])  # [num_envs, keydim]
            rngs_split_opponents_envs = jnp.concatenate(
                [rngs_split_envs] * args["num_opps"]
            )  # ([num_envs, keydim], [num_envs, keydim], ...)->[num_opps*num_envs, keydim] s.t. [o1e1,o1e2; o2e1,o2e2; ...]
            rngs = rngs_split_opponents_envs.reshape(
                (args["num_opps"], args["num_envs"], -1)
            )  # [num_opps, num_envs, keydim] s.t. [[o1e1, o1e2], [o2e1, o2e2], ...]

            # reset env
            obs, env_state = env.batch_reset(
                rngs, _env_params
            )  # IN: keys (00), params (--), OUT: obs (00), env_state (00)
            rewards = [
                jnp.zeros((args["num_opps"], args["num_envs"])),
                jnp.zeros((args["num_opps"], args["num_envs"])),
            ]

            # reset Player 1's memory
            _a1_mem = agent1.batch_reset(_a1_mem, False)  # MemState(+) -> MemState(+)

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,  # init: [num_opps, num_envs, keydim] from above
                    *obs,  # init: dict of [n_o, n_e, ...] from env.batch_reset
                    # *rewards,  # init: 0s [n_o, n_e] from above
                    _a1_state,  # init: params: [...], keys: [keydim], timesteps: int; passthrough from run_loop
                    _a1_mem,  # init: freshly reset [n_o, n_e, ...] from passthrough from run_loop
                    _a2_state,  # init: params: [n_o, ...], keys: [n_o, keydim], timesteps: [n_o] passthrough from run_loop
                    _a2_mem,  # init: [n_o, n_e, ...] passthrough from run_loop
                    env_state,  # init: [n_o, n_e, X] (X= invs: 2, prices: 2, t: 1) from env.batch_reset
                    _env_params,  # init: unbatched from env.batch_reset
                ),
                None,
                length=num_outer_steps,
            )
            # vals: last state of last episode.
            (
                rngs,
                obs1,  # [num_opps, num_envs, ...]
                obs2,
                # r1,
                # r2,
                a1_state,  # ?
                a1_mem,  # ? run_loop->batch_reset->batch_policy
                a2_state,  # [num_opps, ...]
                a2_mem,  # [num_opps, ...]
                env_state,
                env_params,
            ) = vals

            # stack: all trajectories of all episodes. both trajectories are [n_outer, n_inner, n_opps, n_envs, ...]
            traj_1, traj_2, env_traj, info_traj, a2_metrics = stack

            # with profiler.StepTraceAnnotation("ag1_update"):
            # update outer agent
            # TODO: mem in is [n_o*n_e, 1] but out is [1] -- this is fine b/c it's never used and immediately reset, resulting in [n_e, 1]
            a1_state, _, a1_metrics = agent1.update(
                reduce_outer_traj(traj_1),  # dim [n_outer*n_inner, n_opps*n_envs, ...]
                self.reduce_opp_dim(obs1),  # dim [n_opps*n_envs, ...]
                a1_state,  # dim [...]
                self.reduce_opp_dim(a1_mem),  # dim [n_opps*n_envs, ...]
            )

            # reset memory of ag1 and ag2 (hidden isn't reset)
            a1_mem = agent1.batch_reset(
                a1_mem, False
            )  # dim [n_opps(!), ...] where mem has hardcoded hidden: [n_envs, 1] and extras (changed here): [n_envs]
            a2_mem = agent2.batch_reset(a2_mem, False)  # dim [n_opps(!), ...]

            # rewards
            # rewards_1 = traj_1.rewards.mean() # mean over n_opps, n_envs
            # rewards_2 = traj_2.rewards.mean()

            # with profiler.StepTraceAnnotation("env_stats_calc"):
            # Stats
            env_stats = {}
            env_stats = jax.tree_util.tree_map(  # takes mean over n_outer, n_opponents
                lambda x: x,
                self.marketenv_stats(
                    env_state,
                    traj_1,
                    traj_2,
                    env_traj,
                    info_traj,
                    args["num_envs"],
                    args["num_opps"],
                    args["num_outer_steps"],
                    env_params.initial_inventories,
                    self.competitive_profits,
                    self.collusive_profits,
                ),
            )
            env_stats = jax.tree_util.tree_map(lambda x: x.astype(jnp.float16), env_stats)

            return (
                env_stats,
                # rewards_1,
                # rewards_2,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,  # [num_outer, num_opps, ..]
            )

        self.rollout = jax.jit(_rollout)

        def _eval_ep(carry, unused):
            """Runs 1 deterministic episode
            expects inputs [n_opps, n_envs, ...]"""
            (
                rngs,
                obs1,
                obs2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = carry
            a1, a1_state, new_a1_mem, a1_extras = agent1.batch_eval_policy(a1_state, obs1, a1_mem)
            a2, a2_state, new_a2_mem, a2_extras = agent2.batch_eval_policy(a2_state, obs2, a2_mem)
            (
                (next_obs1, next_obs2),
                env_state,
                rewards,
                unnormalized_rewards,
                _,
                info,
            ) = env.batch_step(
                rngs,
                env_state,
                (a1, a2),
                env_params,
            )
            r1 = rewards[:, :, 0]
            r2 = rewards[:, :, 1]
            if args["normalize_rewards_wrapper"]:
                r1_unnormalized = unnormalized_rewards[:, :, 0]  # .mean()
                r2_unnormalized = unnormalized_rewards[:, :, 1]  # .mean()
            else:
                r1_unnormalized = r1
                r2_unnormalized = r2

            if args["normalize_rewards_manually"]:
                r1_rescaled = (r1 - args["normalizing_rewards_min"]) / (
                    args["normalizing_rewards_max"] - args["normalizing_rewards_min"]
                )
                r2_rescaled = (r2 - args["normalizing_rewards_min"]) / (
                    args["normalizing_rewards_max"] - args["normalizing_rewards_min"]
                )
            else:
                r1_rescaled = r1
                r2_rescaled = r2
            traj1 = EvalSample(obs1, a1, r1_rescaled, r1_unnormalized, a1_extras)
            traj2 = EvalSample(obs2, a2, r2_rescaled, r2_unnormalized, a2_extras)

            return (
                rngs,
                next_obs1,
                next_obs2,
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
                env_params,
            ), (traj1, traj2, env_state, info)

        def _eval_rollout(rng_eval, a1_state, a1_mem, a2_state, a2_mem, env_params_eval):
            # result: [num_opps, 1 (num_envs), keydim]
            rngs_eval = jax.random.split(rng_eval, 1)
            rngs_eval = jnp.concatenate([rngs_eval] * args["num_opps"]).reshape(
                args["num_opps"], 1, -1
            )

            # reset things that need to be reset
            # reset env -- expects rng: [num_opps, num_envs, keydim] but params []
            obs_eval, env_state_eval = env.batch_reset(rngs_eval, env_params_eval)
            # rewards = [
            #     jnp.zeros((args["num_opps"], 1)),
            #     jnp.zeros((args["num_opps"], 1)),
            # ]

            ## run an episode
            # reset both mems
            a1_mem_eval = agent1.batch_reset(a1_mem, True)  # eval=True
            a2_mem_eval = agent2.batch_reset(a2_mem, True)

            # agent states not reset (params!), but mems reset (hidden states)
            initial_carry = (
                rngs_eval,
                *obs_eval,
                a1_state,
                a1_mem_eval,
                a2_state,
                a2_mem_eval,
                env_state_eval,
                env_params_eval,
            )

            # run an episode
            _, eval_trajectories = jax.lax.scan(
                _eval_ep, initial_carry, None, args["num_inner_steps"]
            )
            (eval_traj1, eval_traj2, eval_env_traj, eval_info_traj) = eval_trajectories

            eval_log_data = {}
            eval_log_data = jax.tree_util.tree_map(  # takes mean over n_outer, n_opponents
                lambda x: x,
                self.marketenv_eval_stats(
                    eval_env_traj,
                    eval_traj1,
                    eval_traj2,
                    eval_info_traj,
                    env_params_eval.initial_inventories,
                    self.competitive_profits,
                    self.collusive_profits,
                ),  # result should have shape [T] in all dims
            )
            return eval_log_data

        self.eval_rollout = jax.jit(_eval_rollout)

        # Add the forced_deviation_rollout method
        def _forced_deviation_rollout(
            rng_eval,
            a1_state,
            a1_mem,
            a2_state,
            a2_mem,
            env_params_eval,
            deviation_times,
        ):
            """
            Run a forced deviation rollout, where agent 1 deviates at a specified time.
            Input deviation_times is an array of times at which agent 1 deviates. One episode is run for each deviation time.
            Assumes num_envs=1.
            """

            # Define episode with forced deviation, this gets scanned over num_inner_steps (once per deviation time)
            def _run_episode_with_deviation(carry, deviation_time):
                rng_eval = carry  # Carry over the RNG

                # Initialize RNGs for the environment
                rngs_eval = jax.random.split(rng_eval, 1)  # [num_opps, 1 (num_envs), keydim]
                rngs_eval = jnp.concatenate([rngs_eval] * args["num_opps"]).reshape(
                    args["num_opps"], 1, -1
                )

                # Reset environment and agent memories
                obs_eval, env_state_eval = env.batch_reset(rngs_eval, env_params_eval)
                a1_mem_eval = agent1.batch_reset(a1_mem, True)  # eval=True
                a2_mem_eval = agent2.batch_reset(a2_mem, True)

                time_step = 0  # Initialize time step

                # Initial carry for the scan
                initial_carry = (
                    rngs_eval,
                    *obs_eval,
                    a1_state,  # from forced_deviation_rollout input
                    a1_mem_eval,
                    a2_state,  # from forced_deviation_rollout input
                    a2_mem_eval,
                    env_state_eval,
                    env_params_eval,  # from forced_deviation_rollout input
                    time_step,
                )

                # Define the per-step function with forced deviation
                def _eval_ep_with_deviation(carry, unused):
                    (
                        rngs,
                        obs1,
                        obs2,
                        a1_state,
                        a1_mem,
                        a2_state,
                        a2_mem,
                        env_state,
                        env_params,
                        time_step,
                    ) = carry

                    # Get actions from both agents
                    a1, a1_state, new_a1_mem, a1_extras = agent1.batch_eval_policy(
                        a1_state, obs1, a1_mem
                    )
                    a2, a2_state, new_a2_mem, a2_extras = agent2.batch_eval_policy(
                        a2_state, obs2, a2_mem
                    )

                    # Force agent 1 to deviate at the specified time. Don't overwrite anything else, s.t. rest of agent evolution is unaffected
                    competitive_action = jnp.full(
                        (self.num_opps, 1),  # num_envs=1
                        self.args["competitive_action"],
                        dtype=a1.dtype,
                    )
                    a1 = jax.lax.cond(
                        time_step == deviation_time,
                        lambda _: competitive_action,
                        lambda _: a1,
                        operand=None,
                    )

                    # Proceed with environment step
                    (
                        (next_obs1, next_obs2),
                        env_state,
                        rewards,
                        unnormalized_rewards,
                        _,
                        info,
                    ) = env.batch_step(
                        rngs,
                        env_state,
                        (a1, a2),
                        env_params,
                    )

                    # Process rewards (similar to eval_rollout)
                    r1 = rewards[:, :, 0]
                    r2 = rewards[:, :, 1]
                    if args["normalize_rewards_wrapper"]:
                        r1_unnormalized = unnormalized_rewards[:, :, 0]  # .mean()
                        r2_unnormalized = unnormalized_rewards[:, :, 1]  # .mean()
                    else:
                        r1_unnormalized = r1
                        r2_unnormalized = r2

                    # Manually normalize rewards, uses values calculated in main.py from config.
                    if args["normalize_rewards_manually"]:
                        r1_rescaled = (r1 - args["normalizing_rewards_min"]) / (
                            args["normalizing_rewards_max"] - args["normalizing_rewards_min"]
                        )
                        r2_rescaled = (r2 - args["normalizing_rewards_min"]) / (
                            args["normalizing_rewards_max"] - args["normalizing_rewards_min"]
                        )
                    else:
                        r1_rescaled = r1
                        r2_rescaled = r2

                    # Create trajectory samples
                    traj1 = EvalSample(obs1, a1, r1_rescaled, r1_unnormalized, a1_extras)
                    traj2 = EvalSample(obs2, a2, r2_rescaled, r2_unnormalized, a2_extras)

                    # Increment time step
                    time_step = time_step + 1

                    # Update carry and outputs
                    new_carry = (
                        rngs,
                        next_obs1,
                        next_obs2,
                        a1_state,
                        new_a1_mem,
                        a2_state,
                        new_a2_mem,
                        env_state,
                        env_params,
                        time_step,
                    )
                    outputs = (traj1, traj2, env_state, info)

                    return new_carry, outputs

                # Run the episode with forced deviation.
                _, eval_trajectories = jax.lax.scan(
                    _eval_ep_with_deviation,
                    initial_carry,
                    None,
                    args["num_inner_steps"],
                )

                (eval_traj1, eval_traj2, eval_env_traj, eval_info_traj) = eval_trajectories

                # Collect evaluation data
                eval_log_data = {}
                eval_log_data = self.marketenv_eval_stats(
                    eval_env_traj,
                    eval_traj1,
                    eval_traj2,
                    eval_info_traj,
                    env_params_eval.initial_inventories,
                    self.competitive_profits,
                    self.collusive_profits,
                )  # each value: shape [num_envs, num_timestep]

                # Return the RNG and evaluation data
                return rng_eval, eval_log_data

            # Use jax.lax.scan to run over all deviation times
            initial_carry = rng_eval
            _, forced_deviation_log_data = jax.lax.scan(
                _run_episode_with_deviation, initial_carry, deviation_times
            )  # output shape [num_deviation_times, num_timesteps]

            return forced_deviation_log_data

        self.forced_deviation_rollout = jax.jit(_forced_deviation_rollout)

    @partial(
        jax.jit,
        static_argnums=(
            0,
            4,
        ),
    )
    def run_loop(self, seed, env_params, agents, num_iters):
        """Run training of agents in environment
        Runs the rollout num_iter times:
            rollout (num_iter times):
                runs outer rollout (num_outer_steps):
                    1 episode (inner rollout) with num_inner_steps steps
                    trains ag2
                trains ag1
                resets both ags' memory
        So by setting num_outer_steps=1, this becomes symmetrical and in effect:
        for i=0...num_iter:
            run 1 episode with num_inner_steps steps
            train ag1, train ag2
            reset both ags' memory
        """
        # RNG is entirely useless here, RNG is only used in agent creation before run_loop is ever called
        rng = jax.random.PRNGKey(seed)
        init_rng = rng

        # Start at: dim Train [...], Mem [num_envs, ...]
        a1_state, a1_mem = (agents[0]._state, agents[0]._mem)
        a2_state, a2_mem = (agents[1]._state, agents[1]._mem)

        def scan_body(carry, i):
            rng, a1_state, a1_mem, a2_state, a2_mem = carry
            rng, rng_run = jax.random.split(rng)

            (
                env_stats,
                # rewards_1,
                # rewards_2,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            ) = self.rollout(rng_run, a1_state, a1_mem, a2_state, a2_mem, env_params)

            # log_data = (env_stats, rewards_1, rewards_2, a1_metrics, a2_metrics)
            log_data = (env_stats, a1_metrics, a2_metrics)

            return (rng, a1_state, a1_mem, a2_state, a2_mem), log_data

        # with profiler.StepTraceAnnotation("main_rollout"):
        initial_carry = (rng, a1_state, a1_mem, a2_state, a2_mem)
        (rng, a1_state, a1_mem, a2_state, a2_mem), log_data = jax.lax.scan(
            scan_body, initial_carry, jnp.arange(num_iters)
        )

        train_log_data = log_data

        agents[0]._state, agents[0]._mem = a1_state, a1_mem
        agents[1]._state, agents[1]._mem = a2_state, a2_mem

        ### EVAL PART
        eval_log_data = self.eval_rollout(rng, a1_state, a1_mem, a2_state, a2_mem, env_params)

        deviation_times = jnp.arange(self.args["num_inner_steps"])
        ### FORCED DEVIATION PART
        forced_deviation_log_data = self.forced_deviation_rollout(
            rng, a1_state, a1_mem, a2_state, a2_mem, env_params, deviation_times
        )

        log_data = train_log_data, eval_log_data, forced_deviation_log_data

        return agents, log_data, init_rng

    def log_data(self, log_data, agents, num_iters, watchers):
        """Process and log the data collected during training."""

        log_data, eval_log_data, forced_deviation_log_data = log_data

        last_trained_metrics_a1 = {}
        last_trained_metrics_a2 = {}
        last_logged_ep_a1 = -1
        last_logged_ep_a2 = -1

        # unpack the log_data as it's a tuple of tuples
        (
            all_env_stats,  # [T]
            all_a1_metrics,  # [T, ...]
            all_a2_metrics,  # [T, num_outer, num_opps, ...]
        ) = log_data

        # assert num_iters == all_env_stats
        log_interval = max(num_iters // MAX_WANDB_CALLS, 5 if num_iters > 1000 else 1)

        def transpose_metrics_np(metrics, has_extra_dim=False):
            """colab runtime: 14mus (25mus with dtype casts) np.() vs 640mus (1ms) jnp.()
            our runtime: 0.212s
            IN: metrics as dict, where keys = metric name and values = tensor with 0th axis indexing time
            i.e., IN: {keys: [T, extra_dims]}
            OUT: list, each entry list[t] is a dict for timestep t containing metrics at time t
            i.e., OUT: [t=0: {keys: [extra_dims]}, t=1: {keys: [extra_dims]}, ...]
            Note:
            - dtypes are not preserved (see comments), so this works with metrics containing numbers/bools only
            - if extra_dims is 0, return values are scalars rather than 0-dim Arrays
            """
            keys = list(metrics.keys())
            # dtypes = [v.dtype for v in metrics.values()]
            values = np.array([metrics[k] for k in keys])  # dim [keys, T, extra_dims]
            # transpose to: [T, keys, extra_dims]
            transposed = np.transpose(values, (1, 0, *range(2, values.ndim)))
            # [t=0: {keys: [extra_dims]}, t=1: {keys: [extra_dims]}, ...]
            # return [{k: row.astype(dtype_k) for k, row, dtype_k in zip(fkeys, mat, dtypes)} for mat in transposed]
            return [{k: row for k, row in zip(keys, mat)} for mat in transposed]

        transpose_time = time.time()
        a1_metrics_transposed = transpose_metrics_np(all_a1_metrics)
        print(f"Transposing a1 metrics took {time.time() - transpose_time:.3f} seconds")
        transpose_time = time.time()
        a2_metrics_transposed = transpose_metrics_np(all_a2_metrics, has_extra_dim=True)
        print(f"Transposing a2 metrics took {time.time() - transpose_time:.3f} seconds")
        env_stats_transposed = transpose_metrics_np(all_env_stats)
        print(f"Transposing env stats took {time.time() - transpose_time:.3f} seconds")

        logging_loop_time = time.time()
        for i in range(num_iters):
            # Some agents don't train every episode. For those, keep track of the most recent training ep's metrics, and which ep it was
            a1_metrics = a1_metrics_transposed[i]
            a2_metrics = a2_metrics_transposed[i]

            if a1_metrics.get("trained"):
                last_trained_metrics_a1 = a1_metrics
                last_training_ep_a1 = i
            if a2_metrics.get(
                "trained"
            ):  # this'll throw errors if num_opps or num_outer > 1. Solution: use .all().
                last_trained_metrics_a2 = a2_metrics
                last_training_ep_a2 = i

            # Check if this is a logging iteration
            if i % log_interval == 0:
                # get the stats, rewards, and metrics for this iteration
                env_stats = jax.tree.map(
                    lambda x: x[i], all_env_stats
                )  # could also use transpose_metrics_np but this is called less often, it's fine

                ## print logging (commented out -- no point if we're not live logging anyway)
                ## if num_iters is large, space out prints to avoid spam (20% speed loss if it's are fast!)
                # if num_iters > 1000:
                #     log_print_interval_factor: int = num_iters / 100
                # else:
                #     log_print_interval_factor: int = 1
                # if i % log_print_interval_factor == 0:
                #     print(f"Episode {i}")
                #     for stat in env_stats.keys():
                #         if stat.startswith(
                #             # "train/0th_env/"
                #             "train/all_envs/"
                #         ):  # print stat with stripped out "train/"
                #             stripped_stat = stat[len("train/") :]
                #             print(stripped_stat + f": {env_stats[stat].item()}")

                # if most recent training ep not yet logged: merge its metrics into a1_metrics and a2_metrics
                if last_trained_metrics_a1:
                    a1_metrics = a1_metrics | last_trained_metrics_a1
                if last_trained_metrics_a2:
                    a2_metrics = a2_metrics | last_trained_metrics_a2

                # logging learning rate (tricky because LR scheduling may be on/off)
                def retrieve_learning_rate(agent, agent_metrics):
                    """Retrieves learning rate from agent. Agent must implement attributes _lr_scheduling, _initial_learning_rate, _total_num_transitions."""
                    try:
                        if agent._lr_scheduling:
                            scheduler_step = agent_metrics["scheduler_steps"]
                            total_num_transitions = agent._total_num_transitions
                            scheduler = optax.linear_schedule(
                                init_value=agent._initial_learning_rate,
                                end_value=0,
                                transition_steps=total_num_transitions,
                            )  # create scheduler
                            current_learning_rate = scheduler(scheduler_step)
                        else:  # use fixed LR
                            current_learning_rate = agent._initial_learning_rate
                    except Exception as e:
                        # print(f"Error: {e}")
                        current_learning_rate = None
                    return current_learning_rate

                # Vectorize the learning rate retrieval func for agents trained in inner rollout with [n_outer, n_opps] metrics.
                retrieve_learning_rate_vmap = jax.vmap(
                    jax.vmap(retrieve_learning_rate, in_axes=(None, 0)),
                    in_axes=(None, 0),
                )

                # Apply LR retrieval func to agent 1 and 2
                try:
                    a1_current_learning_rate = retrieve_learning_rate(agents[0], a1_metrics)
                    a1_metrics["learning_rate"] = a1_current_learning_rate

                    a2_current_learning_rates = retrieve_learning_rate_vmap(agents[1], a2_metrics)
                    a2_current_learning_rates = jnp.tile(
                        a2_current_learning_rates, (self.args["num_outer_steps"], 1)
                    )
                    a2_metrics["learning_rate"] = a2_current_learning_rates
                except Exception as e:  # if agent doesn't implement it (e.g. random)
                    print(f"Error: {e}")
                    pass

                # log metrics to wandb
                if watchers:
                    # Merge metrics into agents' logger.metrics

                    # It could happen that there's training and non-training episodes (e.g. DQN), and since the last time we logged,
                    # there wasn't a new training episode. In that case, we don't want to log the (unchanged) agent metrics again.
                    # This is not relevant for PPO: last_logged <= i-1, last_trained = i so always TRUE.
                    if last_trained_metrics_a1 and last_logged_ep_a1 < last_training_ep_a1:
                        # agent 1: metrics []. Averaging over metrics dims
                        flattened_metrics_1 = jax.tree_util.tree_map(
                            lambda x: jnp.mean(x), a1_metrics
                        )
                        agents[0]._logger.metrics = agents[0]._logger.metrics | flattened_metrics_1
                        # merging a1_metrics into this before could reset it to False
                        agents[0]._logger.metrics["trained"] = True
                        last_logged_ep_a1 = last_training_ep_a1
                        # print(f"logging agents for ep {i}!")
                    else:  # don't log! doing it this way ensures we can still call watchers on all agents (as DQN watchers check for trained=False)
                        # ensure that DQN watchers don't log this
                        agents[0]._logger.metrics["trained"] = False

                    if last_trained_metrics_a2 and last_logged_ep_a2 < last_training_ep_a2:
                        # agent >2: metrics [outer_timesteps, num_opps]. Averaging over num_opps, then summing over outer_timesteps
                        flattened_metrics_2 = jax.tree_util.tree_map(
                            lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                        )
                        agents[1]._logger.metrics = agents[1]._logger.metrics | flattened_metrics_2
                        agents[1]._logger.metrics["trained"] = True
                        last_logged_ep_a2 = last_training_ep_a2
                    else:
                        agents[1]._logger.metrics["trained"] = False

                    # this calls ppo_log on the agent, which calls losses_ppo, which reads (newly updated) agent._logger.metrics and logs to wandb
                    for watcher, agent in zip(watchers, agents):
                        watcher(agent)

                    # log env stats into wandb
                    env_stats = jax.tree_util.tree_map(lambda x: x.item(), env_stats)
                    log_data = {
                        "train_iteration": i,
                    } | env_stats
                    wandb.log(
                        log_data,
                        step=i,
                        commit=True,
                    )
                else:
                    # print(f"{i}: No watchers found!")
                    pass

        print(f"Big logging loop took {time.time() - logging_loop_time:.3f} seconds")

    def log_data_vmap(self, log_data, agents, num_iters, watchers):
        """Process and log the data collected during training.
        This data is from doing a gridsearch over rng and configs.
        Everything has leading dimensions [num_seeds, num_configs]"""

        pass

    ## Defining this as a pytree
    def tree_flatten(self):
        children = (self.random_key,)
        aux_data = {
            "rollout": self.rollout,
            "eval_rollout": self.eval_rollout,
            "forced_deviation_rollout": self.forced_deviation_rollout,
            # everything rollout depends on
            "args": self.args,
            "reduce_opp_dim": self.reduce_opp_dim,
            "marketenv_stats": self.marketenv_stats,
            "marketenv_eval_stats": self.marketenv_eval_stats,
            "split": self.split,
            "competitive_profits": self.competitive_profits,
            "collusive_profits": self.collusive_profits,
            "competitive_profits_episodetotal": self.competitive_profits_episodetotal,
            "collusive_profits_episodetotal": self.collusive_profits_episodetotal,
            "num_opps": self.num_opps,
            # other stuff
            "start_time": self.start_time,
            "save_dir": self.save_dir,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.random_key = children

        # Restore aux_data
        for key, value in aux_data.items():
            setattr(obj, key, value)

        return obj


jax.tree_util.register_pytree_node(
    TwoAgentGridsearchRunner,
    TwoAgentGridsearchRunner.tree_flatten,
    TwoAgentGridsearchRunner.tree_unflatten,
)

# endregion
