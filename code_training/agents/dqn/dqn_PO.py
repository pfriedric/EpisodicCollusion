# Adapted from https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/dqn/learning_lib.py

from multiprocessing import dummy
import random
from re import U
import jax
from jaxtyping import Array
from typing import NamedTuple, Tuple, Dict, Any
from matplotlib.pyplot import bar
import typing_extensions
import optax
import haiku as hk
import jax.numpy as jnp
from agents.agent import AgentInterface
from agents.dqn.networks import make_dqn_marketenv_network, QNetwork_marketenv
from utils import (
    MemoryState,
    TrainingState,
    Logger,
    add_batch_dim,
    get_advantages,
    float_precision,
)

import omegaconf
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from functools import partial
import pickle


class TimeStep(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Array
    actions: Array
    rewards: Array
    # discounts: Array
    dones: Array


class DQNTrainingState(NamedTuple):
    """Holds the agent's training state"""

    params: hk.Params
    target_params: hk.Params
    opt_state: optax.GradientTransformation
    n_episodes: int  # this controls when to train & update target network. incremented by 1 every time agent.update() is called (after 1 episode for ag2, n_outer episodes for ag1)
    n_epsilon: int  # this controls epsilon-greedy annealing. incremented by n_envs everytime policy() is called.
    current_epsilon: float
    random_key: Array
    buffer_state: TrajectoryBufferState


# The pmap axis name. Data means data parallelization.
PMAP_AXIS_NAME = "data"


class DQN:
    # Inputs that go in the loss function and not the agent:
    # Q: discount, max_abs_reward
    # Regularized Q: discount, regularizer_coeff (no reward clipping, vanilla TD error, adds reg. term)
    # Münchhausen Q (M-DQN): entropy_temperature, munchhausen_coefficient, clip_value_min, discount, max_abs_reward, huber_loss_parameter
    # Quantile Regression DQN (QrDQN): num_atoms, huber_param
    # Prioritized Double Q: discount, importance_sampling_exponent, max_abs_reward, huber_loss_parameter

    # Start with Double-Q, no clipping.

    def __init__(
        self,
        network: NamedTuple,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: dict,
        obs_dtypes: dict,
        obs_limits: dict,
        num_envs: int = 4,
        player_id: int = 0,
        num_iterations: int = 1,
        initial_learning_rate: float = 1e-3,
        lr_scheduling: bool = False,
        buffer_size: int = 1000000,  # tbd
        buffer_batch_size: int = 32,  # number of transitions to sample at once
        discount: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_finish: float = 0.1,
        epsilon_anneal_time: int = 1000000,
        epsilon_anneal_type: str = "linear",
        epsilon_clipping: bool = False,
        polyak_tau: float = 0.005,
        initial_exploration_episodes: int = 0,
        training_interval_episodes: int = 1,
        target_update_interval_episodes: int = 1,
    ):

        @jax.jit
        def rescale_to_minus_one_one(x, lower, upper):
            return 2 * (x - lower) / (upper - lower) - 1

        @jax.jit
        def rescale_to_zero_one(x, lower, upper):
            return (x - lower) / (upper - lower)

        @jax.jit
        def obscure_observations(observation: Dict):
            """obscures inventories and last_actions to make DQN training more difficult"""
            new_observation = observation.copy()
            new_observation["inventories"] = jnp.zeros_like(observation["inventories"])
            new_observation["inventories"] = (
                new_observation["inventories"]
                .at[player_id]
                .set(observation["inventories"][player_id])
            )
            return new_observation

        @jax.jit
        def rescale_observations(observation: Dict, obs_limits: dict):
            """takes in observations dict, applies element wise natural log to inventories
            Args: observation: dict with keys 'inventories', 'last_actions', 't'
            Returns: observation: dict where 'inventories' has been elementwise log transformed

            Options:
            - inventories: 'log' or [-1,1] or [0,1]
            - last_prices: [-1,1] or [0,1]
            - last_actions: [-1,1] or [0,1]
            - t: t/time_horizon -> [0,1] or 1-t/time_horizon -> [1,0]
            """
            new_observation = obscure_observations(observation)
            # new_observation["inventories"] = jnp.log(observation["inventories"])
            new_observation["inventories"] = rescale_to_zero_one(
                observation["inventories"], 0, obs_limits["inventory_uppers"]
            )

            new_observation["last_actions"] = rescale_to_zero_one(
                observation["last_actions"], 0, obs_limits["last_actions_upper"]
            )
            new_observation["last_prices"] = rescale_to_zero_one(
                observation["last_prices"],
                obs_limits["last_prices_lower"],
                obs_limits["last_prices_upper"],
            )
            # time: 1-t/T: 1->0
            new_observation["t"] = rescale_to_zero_one(
                observation["t"], 0, obs_limits["t_upper"]
            )
            new_observation["t"] = 1 - new_observation["t"]

            return new_observation

        @partial(jax.jit, donate_argnames=["state"])  # donates buffer for 'state'
        def policy_state_update(state: DQNTrainingState, key: jnp.ndarray, eps: float):
            """Update the state's key, epsilon, and epsilon_counter"""
            state = state._replace(
                random_key=key,
                current_epsilon=eps,
                n_epsilon=state.n_epsilon + num_envs,
            )
            return state

        @partial(jax.jit, donate_argnames=["state"])  # donates buffer for 'state'
        def sgd_state_update(
            state: DQNTrainingState,
            params: hk.Params,
            opt_state: optax.GradientTransformation,
            key: jnp.ndarray,
        ):
            """Update the state's params, opt_state and key for SGD step"""
            state = state._replace(params=params, opt_state=opt_state, random_key=key)
            return state

        @partial(jax.jit, donate_argnames=["state"])  # donates buffer for 'state'
        def target_update(state: DQNTrainingState):
            """Update the target network with the current network's parameters using polyak averaging

            Returns state where target params are updated to:
                tau * current_params + (1-tau) * old_target_params
            """
            state = state._replace(
                target_params=optax.incremental_update(
                    state.params, state.target_params, polyak_tau
                )
            )
            return state

        # @jax.jit
        def epsilon_greedy_exploration(rng, q_vals, timestep):
            """Epsilon-greedy exploration strategy. Returns chosen actions and epsilon value (for metrics)

            Can receive batched input (B, ...) or unbatched (...). Output is (B,) or ().
            """
            rng_a, rng_e = jax.random.split(rng)  # rng_a: sample action, rng_e: epsilon
            if epsilon_anneal_type == "linear":
                eps = jnp.clip(
                    ((epsilon_finish - epsilon_start) / epsilon_anneal_time) * timestep
                    + epsilon_start,
                    epsilon_finish,
                )
            if epsilon_anneal_type == "exponential":
                decay_rate = (epsilon_finish / epsilon_start) ** (
                    1 / epsilon_anneal_time
                )
                eps = epsilon_start * (decay_rate**timestep)
                if epsilon_clipping:
                    eps = jnp.clip(eps, epsilon_finish)
            greedy_actions = jnp.argmax(
                q_vals, axis=-1
            )  # batched or unbatched, num_actions is last axis of q_vals. output: (B,) or () if unbatched
            pick_random = (
                jax.random.uniform(rng_e, greedy_actions.shape) < eps
            )  # pick action randomly
            random_actions = jax.random.randint(
                rng_a,
                shape=greedy_actions.shape,
                minval=0,
                maxval=q_vals.shape[-1],
            )
            actions = jnp.where(pick_random, random_actions, greedy_actions)
            return actions, eps, greedy_actions

        def policy(state: DQNTrainingState, observation: Dict, mem: MemoryState):
            """Policy function for DQN agent
            Receives batched input (batch dim=num_envs).
            In: obs (batch_dim, obs_dim)"""

            key, subkey = jax.random.split(state.random_key)

            # rescale observations
            rescaled_observations = rescale_observations(observation, obs_limits)
            q_vals = network.apply(
                state.params, rescaled_observations
            )  # (batch_dim, num_actions)

            # get actions via epsilon-greedy exploration
            actions, eps, greedy_actions = epsilon_greedy_exploration(
                subkey, q_vals, state.n_epsilon
            )  # (batch_dim,)
            state = policy_state_update(
                state, key, eps
            )  # update key, epsilon, epsilon_counter+=n_envs
            # write greedy actions to mem
            mem.extras["values"] = greedy_actions
            mem = mem._replace(extras=mem.extras)
            return actions, state, mem

        def eval_policy(state: DQNTrainingState, observation: Dict, mem: MemoryState):
            """Evaluation policy for DQN agent
            Receives batched input (batch dim=num_envs).
            In: obs (batch_dim, obs_dim)"""
            rescaled_observations = rescale_observations(observation, obs_limits)
            q_vals = network.apply(
                state.params, rescaled_observations
            )  # (batch_dim, num_actions)

            greedy_actions = jnp.argmax(q_vals, axis=-1)
            extras = {"q_vals": q_vals}
            return greedy_actions, state, mem, extras

        def loss(params: hk.Params, target_params: hk.Params, data: TimeStep):
            """Loss function for DQN agent. Expects batched data (B, ...). Computes batch loss.
            This outputs the loss val + auxiliary metrics. It's called by sgd_step as grad(loss) which returns gradients and metrics.
            """
            obs_t = data.first.observations  # (B, obs_dim)
            action_t = data.first.actions  # (B,)
            reward_t = data.first.rewards  # (B,)
            obs_next = data.second.observations  # (B, obs_dim)
            dones_t = data.first.dones  # (B,)

            # RESCALE OBSERVATIONS
            rescaled_obs_t = rescale_observations(obs_t, obs_limits)
            rescaled_obs_next = rescale_observations(obs_next, obs_limits)

            # TD-target via Double Q-learning (target network selects next_action)
            q_next_target = network.apply(
                target_params, rescaled_obs_next
            )  # (B, num_actions)
            next_action_qvals_target = jnp.max(q_next_target, axis=-1)  # (B,)
            td_target = (
                reward_t + (1 - dones_t) * discount * next_action_qvals_target
            )  # (B,)

            # Q-val of chosen action at t
            q_t = network.apply(params, rescaled_obs_t)  # (B, num_actions)
            # Essentially: q_t[:, action_t]. Shape: (B,)
            chosen_action_qvals = jnp.take_along_axis(
                q_t, jnp.expand_dims(action_t, axis=-1), axis=-1
            ).squeeze(axis=-1)

            # TD-error: Q_policy(a_t) - (r_t + discount*Q_target(a_t+1))
            td_error = chosen_action_qvals - td_target
            loss_val = jnp.mean(jnp.square(td_error))  # MSE of TD-error
            loss_metrics = {
                "loss_value": loss_val,
                "max_q_val": jnp.max(q_t),  # , axis=-1),
                "min_q_val": jnp.min(q_t),  # , axis=-1),
                "mean_q_val": jnp.mean(q_t),  # , axis=-1), # (B)
                "mean_chosen_q_val": jnp.mean(chosen_action_qvals),
                "mean_td_target": jnp.mean(td_target),
                "td_error": jnp.mean(td_error),
            }
            return loss_val, loss_metrics

        def sgd_step(
            state: DQNTrainingState,
        ) -> Tuple[DQNTrainingState, Dict[str, jnp.ndarray]]:
            """SGD step for DQN agent. Updates model params using optimizer."""

            # sample from replay buffer
            next_key, buffer_key = jax.random.split(state.random_key, 2)
            learn_batch = self._buffer_fn.sample(
                state.buffer_state, buffer_key
            ).experience  # returns experience.first, experience.second w/ size (buffer_batch_size, ...)

            # update model, 1 epoch
            params = state.params
            target_params = state.target_params
            opt_state = state.opt_state

            # compute gradients
            grad_fn = jax.jit(jax.grad(loss, has_aux=True))
            gradients, metrics = grad_fn(
                params, target_params, learn_batch
            )  # compute gradient on single batch

            # update params
            updates, new_opt_state = optimizer.update(gradients, opt_state)
            new_params = optax.apply_updates(params, updates)

            # metrics
            metrics["norm_grad"] = optax.global_norm(gradients)
            metrics["norm_updates"] = optax.global_norm(updates)
            metrics["trained"] = True
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)  # potentially unneeded?

            # new state: change params & key
            new_state = sgd_state_update(state, new_params, new_opt_state, next_key)

            new_memory = MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros(num_envs, dtype=jnp.int32),
                    "log_probs": jnp.zeros(num_envs),
                },
            )
            return new_state, new_memory, metrics

        def make_initial_state(
            key: Any, hidden: jnp.ndarray
        ) -> Tuple[DQNTrainingState, MemoryState]:
            key, subkey = jax.random.split(key)
            # # create dummy transition
            # dummy_obs = {}
            # for k, v in obs_spec.items():
            #     if v == ():
            #         dummy_obs[k] = jnp.zeros(1)
            #     else:
            #         dummy_obs[k] = jnp.zeros(v)

            if isinstance(obs_spec, dict):
                dummy_obs = {}
                for k, shape in obs_spec.items():
                    dtype = obs_dtypes[k]
                    if shape == ():  # Handle scalar case differently if needed
                        dummy_obs[k] = jnp.zeros(1, dtype=dtype)
                    if k == "t":  # not elif as t is scalar but needs diff. dtype
                        dummy_obs[k] = jnp.zeros(shape=1, dtype=jnp.int32)
                    else:
                        dummy_obs[k] = jnp.zeros(shape=shape, dtype=dtype)

            # using TimeStep guarantees that samples are a TimeStep as well
            init_transition = TimeStep(
                dummy_obs,
                actions=jnp.zeros((), dtype=jnp.int32),  # () i.e. scalar
                rewards=jnp.zeros((), dtype=jnp.float32),  # () i.e. scalar
                dones=jnp.zeros((), dtype=jnp.bool_),  # () i.e. scalar
            )

            buffer_state = self._buffer_fn.init(
                init_transition
            )  # type TrajectoryBufferState
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            initial_target_params = jax.tree.map(lambda x: jnp.copy(x), initial_params)

            initial_training_state = DQNTrainingState(
                params=initial_params,
                target_params=initial_target_params,
                opt_state=initial_opt_state,
                n_episodes=0,
                n_epsilon=0,
                current_epsilon=epsilon_start,
                random_key=key,
                buffer_state=buffer_state,
            )

            initial_memory_state = MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros((num_envs), dtype=jnp.int32),
                    "log_probs": jnp.zeros((num_envs)),
                },
            )
            return initial_training_state, initial_memory_state

        def prepare_batch(traj_batch: NamedTuple, done: Any):
            """if needed, prepare batch for training -- do something with 'done' states."""
            return traj_batch

        # Define the agent's replay buffer. This function is used for sample, add, can_sample
        # Stores data in [buffer_size, sequence_length, data_dims] shape. Stores buffer_size unique transitions.
        # Flat buffer: sample sequence length 2 (s and s'), period 1 (uniformly random).
        # .init expects [...] (no leading batch or sequence axes!)
        # .add expects [add_batch_size, sequence_length, ...].
        # .sample returns [sample_batch_size, ...]
        # We add n_envs (*n_opps) sequences of length n_inner (*n_outer) to the buffer at a time for ag2 (ag1).
        buffer_fn = fbx.make_flat_buffer(
            max_length=buffer_size,  # max number of transitions stored
            min_length=buffer_batch_size,  # min size after which we can sample
            sample_batch_size=buffer_batch_size,  # exact number of transitions to sample
            add_sequences=True,
            add_batch_size=num_envs,  # if n_outer != 1, this must be =num_envs*num_opps for ag1!
        )

        # Bind buffer functions to agent and jit them.
        # This is stateless b/c buffer state is passed through each of these functions.
        self._buffer_fn = buffer_fn.replace(
            init=jax.jit(buffer_fn.init),
            add=jax.jit(
                buffer_fn.add, donate_argnums=0
            ),  # buffer_state is "donated" i.e. mutated in place via add(). otherwise, would have to reserve memory for both in- and output.
            sample=jax.jit(buffer_fn.sample),
            can_sample=jax.jit(buffer_fn.can_sample),
        )

        # initialise training state
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(random_key, jnp.zeros(1))
        self._prepare_batch = jax.jit(prepare_batch)
        self._sgd_step = jax.jit(sgd_step)

        # initialize functions
        self._policy = policy
        self._target_update = target_update
        self.network = network
        self._eval_policy = eval_policy

        # hyperparams
        self.player_id = player_id
        self._num_envs = num_envs
        self._lr_scheduling = lr_scheduling
        self._initial_learning_rate = initial_learning_rate
        self._total_num_transitions = (
            num_iterations - initial_exploration_episodes
        ) // training_interval_episodes  # needed to retrieve LR in the runner
        self._initial_exploration_episodes = initial_exploration_episodes
        self._training_interval_episodes = training_interval_episodes
        self._target_update_interval_episodes = target_update_interval_episodes
        self._buffer_size = buffer_size
        self._buffer_batch_size = buffer_batch_size
        self._epsilon_finish = epsilon_finish

        # set up logger
        self._logger = Logger()
        self._logger.metrics = {
            "trained": None,
            # "total_steps": None,  # TODO: these aren't tracked
            "sgd_steps": None,  # TODO: these are tracked, but updating doesn't work (b/c they're not traced?)
            "scheduler_steps": None,
            "loss_value": None,
            "max_q_val": None,
            "min_q_val": None,
            "mean_q_val": None,
            "mean_chosen_q_val": None,
            "mean_td_target": None,
            "td_error": None,
            "norm_grad": None,
            "norm_updates": None,
            "rewards_mean": None,  # unused
            "rewards_std": None,  # unused
            "learning_rate": None,  # unused
            "explo_epsilon": None,
        }

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs, dtype=jnp.int32),
                "log_probs": jnp.zeros(num_envs),
            },
        )
        return memory

    def update(
        self, traj_batch, obs: jnp.ndarray, state: DQNTrainingState, mem: MemoryState
    ):
        """Update the agent's state based on a batch of data. Called at end of rollout.
        Expects batched inputs for traj_batch, state, mem.
        traj_batch: [num_timesteps, batch_size, ...]
        obs: [batch_size, ...] (not actually used, but needed for interface consistency)
        state: [batch_size, ...]
        mem: sub-arrays all [batch_size, ...] (not actually used, but needed for interface consistency)
        """
        # do something with the data that correctly handles the "done" states
        # maybe call _, _, mem=self._policy(state, obs, mem) to get some extras that we need?
        # in PPO's case, mem_extras gives value estimates and "done" states should use value estimate of 0 instead.
        # then call prepare_batch
        # --> Think it's unnecessary. Just set target Q-val to immediate reward only.

        # feed replay buffer

        # extract data from traj_batch. these all have shape [num_steps, num_envs, ...]
        (
            observations,
            actions,
            rewards,
            dones,
        ) = (
            traj_batch.observations,
            traj_batch.actions,
            traj_batch.rewards,
            traj_batch.dones,
        )

        # create a TimeStep with that data. [num_steps, num_envs, ...]
        trajectories = TimeStep(
            observations=observations,  # inv: (B, T, ags), prices: (B, T, ags), actions: (B, T, ags), t: (B, T, 1)
            actions=actions,  # (B, T)
            rewards=rewards,  # (B, T)
            dones=dones,  # (B, T)
        )

        # feed into replay buffer. since we set add_sequences=True, it expects [add_batch_size, sequence_dim (variable), ...]
        trajectories = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), trajectories
        )  # [num_steps, num_envs (=batchdim), ...]->[num_envs, num_steps, ...]

        new_buffer_state = self._buffer_fn.add(state.buffer_state, trajectories)
        state = state._replace(buffer_state=new_buffer_state)
        state = state._replace(n_episodes=state.n_episodes + 1)

        # determine whether we update. boolean.
        is_learn_time = (
            self._buffer_fn.can_sample(new_buffer_state)  # enough experience in buffer
            & (
                state.n_episodes > self._initial_exploration_episodes
            )  # pure exploration phase over
            & (
                state.n_episodes % self._training_interval_episodes == 0
            )  # training interval (1=every time agent.update is called)
        )

        empty_metrics = {
            "trained": 0.0,
            "loss_value": 0.0,
            "max_q_val": 0.0,
            "min_q_val": 0.0,
            "mean_q_val": 0.0,
            "mean_chosen_q_val": 0.0,
            "mean_td_target": 0.0,
            "td_error": 0.0,
            "norm_grad": 0.0,
            "norm_updates": 0.0,
        }
        # update model via SGD step
        state, mem, metrics = jax.lax.cond(
            is_learn_time,
            lambda state: self._sgd_step(state),  # update model
            lambda state: (state, mem, empty_metrics),  # do nothing
            state,
        )
        # state, mem, metrics = self._sgd_step(state)

        # add to metrics (up until now, contain either {} or loss & max/min/mean q_val)
        metrics["sgd_steps"] = state.opt_state[1][0]
        metrics["scheduler_steps"] = state.opt_state[2][0]
        metrics["rewards_mean"] = jnp.mean(jnp.abs(jnp.mean(rewards, axis=(0, 1))))
        metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))
        metrics["explo_epsilon"] = state.current_epsilon

        # determine if update target network
        is_update_target_time = (
            state.n_episodes % self._target_update_interval_episodes == 0
        )

        # update target network
        state = jax.lax.cond(
            is_update_target_time,
            lambda state: self._target_update(state),
            lambda state: state,
            state,
        )
        # state = self._target_update(state)

        return state, mem, metrics

    def save_state(self, filepath: str):
        """
        Save the agent's state to a file.
        """
        # Extract picklable components
        state_data = {
            "training_state": self._state,
            "memory_state": self._mem,
            "hyperparameters": {
                "player_id": self.player_id,
                "buffer_size": self._buffer_size,
                "buffer_batch_size": self._buffer_batch_size,
                "num_envs": self._num_envs,
            },
            # Include network parameters
            "network_params": self._state.params,
            "target_network_params": self._state.target_params,
            "optimizer_state": self._state.opt_state,
            # Add buffer state
            "buffer_state": self._state.buffer_state,
            # Add any other necessary data
        }

        # Save to file using pickle
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

    @classmethod
    def load_state(
        cls, filepath: str, network, optimizer: optax.GradientTransformation
    ):
        """
        Load the agent's state from a file and return a DQN instance.
        """
        with open(filepath, "rb") as f:
            state_data = pickle.load(f)

        print(f"Loading state from {filepath}")
        # Create a new instance without invoking __init__
        obj = cls.__new__(cls)

        # Restore training state
        obj._state = state_data["training_state"]
        obj._mem = state_data["memory_state"]

        # Restore hyperparameters
        obj.player_id = state_data["hyperparameters"]["player_id"]

        # Restore network and optimizer states
        obj.network = network  # Must provide the network architecture during loading
        # obj._state = obj._state._replace(
        #     params=state_data["network_params"],
        #     target_params=state_data["target_network_params"],
        #     opt_state=state_data["optimizer_state"],
        #     buffer_state=state_data["buffer_state"],
        # )

        obj._buffer_fn = fbx.make_flat_buffer(
            max_length=state_data["hyperparameters"][
                "buffer_size"
            ],  # max number of transitions stored
            min_length=state_data["hyperparameters"][
                "buffer_batch_size"
            ],  # min size after which we can sample
            sample_batch_size=state_data["hyperparameters"][
                "buffer_batch_size"
            ],  # exact number of transitions to sample
            add_sequences=True,
            add_batch_size=state_data["hyperparameters"][
                "num_envs"
            ],  # if n_outer != 1, this must be =num_envs*num_opps for ag1!
        )

        # Return the reconstructed object
        return obj

    def tree_flatten(self):
        children = (
            self._state,
            self._mem,
            # self.optimizer,
        )
        # anything that's static:
        aux_data = {
            "player_id": self.player_id,
            "_target_update": self._target_update,
            "_num_envs": self._num_envs,
            "_lr_scheduling": self._lr_scheduling,
            "_initial_learning_rate": self._initial_learning_rate,
            "_total_num_transitions": self._total_num_transitions,
            "_initial_exploration_episodes": self._initial_exploration_episodes,
            "_training_interval_episodes": self._training_interval_episodes,
            "_target_update_interval_episodes": self._target_update_interval_episodes,
            "_buffer_fn": self._buffer_fn,
            "_policy": self._policy,
            "_eval_policy": self._eval_policy,
            "_target_update": self._target_update,
            "_prepare_batch": self._prepare_batch,
            "_sgd_step": self._sgd_step,
            "make_initial_state": self.make_initial_state,
            "network": self.network,
            "_logger": self._logger,
            "_buffer_size": self._buffer_size,
            "_buffer_batch_size": self._buffer_batch_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._state, obj._mem = children
        for key, value in aux_data.items():
            setattr(obj, key, value)

        return obj


jax.tree_util.register_pytree_node(DQN, DQN.tree_flatten, DQN.tree_unflatten)


def make_DQN_agent(
    args,
    agent_args,
    obs_spec,
    obs_dtypes,
    action_spec,
    seed: int,
    num_iterations: int,  # this is number of times the rollout is called
    player_id: int,
):
    """Make DQN agent"""
    print(f"Making network for {args.get('env_id')}")
    if (
        args.get("env_id") == "MarketEnv-v1"
        or args.get("env_id") == "MarketEnv-InfiniteInventoryEpisodic"
        or args.get("env_id") == "MarketEnv-InfiniteInventoryInfiniteEpisode"
    ):
        network = make_dqn_marketenv_network(
            action_spec, agent_args.get("hidden_sizes")
        )

    else:
        raise NotImplementedError(
            f"No DQN network implemented for env {args.get('env_id')}"
        )

    # LR scheduling: number of optimizer.update() calls: 1x per sgd_step
    # NOTE: below assumes 1x call of update() per rollout -- ag2 has n_outer updates per rollout.
    num_optimizer_calls = int(
        (num_iterations - agent_args.get("initial_exploration_episodes"))
        * agent_args.get("lr_anneal_duration")
    ) // agent_args.get("training_interval_episodes")

    # learning rate annealing from LR to 0 over exactly duration of the run (hopefully lol)
    if agent_args.get("lr_scheduling"):
        scale = optax.inject_hyperparams(optax.scale)(step_size=-1.0)
        scheduler = optax.linear_schedule(
            init_value=agent_args.get("learning_rate"),
            end_value=0,
            transition_steps=num_optimizer_calls,
        )  # scheduler will step the LR every time it's called
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.get("max_gradient_norm")),
            optax.scale_by_adam(eps=agent_args.get("adam_epsilon")),
            optax.scale_by_schedule(scheduler),
            scale,
        )

    else:  # fixed LR
        scale = optax.inject_hyperparams(optax.scale)(
            step_size=-agent_args.get("learning_rate")
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.get("max_gradient_norm")),
            optax.scale_by_adam(eps=agent_args.get("adam_epsilon")),
            scale,
        )

    random_key = jax.random.PRNGKey(seed=seed)

    # MarketEnv specific:
    obs_limits = {
        "inventory_uppers": jnp.array(args.get("initial_inventories")),
        "last_actions_upper": args.get("num_prices"),
        "last_prices_lower": args.get("possible_prices")[0],
        "last_prices_upper": args.get("possible_prices")[-1],
        "t_upper": args.get("time_horizon"),
    }

    agent = DQN(
        network=network,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        obs_dtypes=obs_dtypes,
        obs_limits=obs_limits,
        num_envs=args.get("num_envs"),
        player_id=player_id,
        num_iterations=num_iterations,
        initial_learning_rate=agent_args.get("learning_rate"),
        lr_scheduling=agent_args.get("lr_scheduling"),
        buffer_size=agent_args.get("buffer_size"),
        buffer_batch_size=agent_args.get("buffer_batch_size"),
        discount=agent_args.get("discount"),
        epsilon_start=agent_args.get("epsilon_start"),
        epsilon_finish=agent_args.get("epsilon_finish"),
        epsilon_anneal_time=agent_args.get("epsilon_anneal_time"),
        epsilon_anneal_type=agent_args.get("epsilon_anneal_type"),
        epsilon_clipping=agent_args.get("epsilon_clipping"),
        polyak_tau=agent_args.get("polyak_tau"),
        initial_exploration_episodes=agent_args.get("initial_exploration_episodes"),
        training_interval_episodes=agent_args.get("training_interval_episodes"),
        target_update_interval_episodes=agent_args.get(
            "target_update_interval_episodes"
        ),
    )
    return agent
