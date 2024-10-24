from multiprocessing import dummy
import jax
from jaxtyping import Array
from typing import NamedTuple, Tuple, Dict, Any
import optax
import haiku as hk
import jax.numpy as jnp
from agents.agent import AgentInterface
from utils import (
    MemoryState,
    TrainingState,
    Logger,
    add_batch_dim,
    get_advantages,
    float_precision,
)
from agents.ppo.networks import make_marketenv_network
import omegaconf
import pickle


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: Array
    actions: Array
    advantages: Array
    # Target value estimate used to bootstrap the value function.
    target_values: Array

    # Value estimate and action log-prob at behavior time.
    behavior_values: Array
    behavior_log_probs: Array


class PPO:
    """A simple PPO agent using JAX"""

    def __init__(
        self,
        network: NamedTuple,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: Tuple,
        obs_limits: dict,
        num_envs: int = 4,
        num_minibatches: int = 16,
        num_epochs: int = 4,
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3_000_000,
        entropy_clipping: bool = False,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        tabular: bool = False,
        player_id: int = 0,
        num_iterations: int = 1,
        initial_learning_rate: float = 1e-3,
        lr_scheduling: bool = False,
    ):
        self._initial_learning_rate = initial_learning_rate
        self._lr_scheduling = lr_scheduling
        self._total_num_transitions = num_iterations * num_epochs * num_minibatches

        ## Normalization methods. These need to be called directly before a network sees obs (network.apply())
        @jax.jit
        def rescale_to_minus_one_one(x, lower, upper):
            return 2 * (x - lower) / (upper - lower) - 1

        @jax.jit
        def rescale_to_zero_one(x, lower, upper):
            return (x - lower) / (upper - lower)

        @jax.jit
        def rescale_observations(observation: Dict, obs_limits: dict):
            """takes in observations dict, applies element wise natural log to inventories
            Args: observation: dict with keys 'inventories', 'last_actions', 't'
            Returns: observation: dict where 'inventories' has been elementwise log transformed

            Options:
            - inventories: log
            - last_prices: leave as is? log, rescale to [-1, 1] or [0,1]
            - last_actions: divide by num_actions to get [0,1] discrete
            - t: divide by time_horizon to get [0,1] discrete
            """
            new_observation = observation.copy()
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
            new_observation["t"] = rescale_to_zero_one(
                observation["t"], 0, obs_limits["t_upper"]
            )
            # new_observation["t"] = obs_limits["t_upper"]-observation["t"]
            # new_observation["t"] = rescale_to_zero_one(
            #     new_observation["t"], 0, obs_limits["t_upper"]
            # )
            # new_observation["last_prices"]: 1) log 2) rescale to [-1, 1]
            # new_observation["last_actions"]: lower=0, upper=num actions
            # new_observation["t"]: lower=0, upper=time horizon
            # return new_observation
            return new_observation

        @jax.jit
        def policy(state: TrainingState, observation: Dict, mem: MemoryState):
            """Agent policy to select actions and calculate agent specific information
            Probably batched (batch_dim = num_envs?):
                observation shape [batch_dim, obs_dims]
                mem shape [batch_dim, mem_dims]"""
            key, subkey = jax.random.split(state.random_key)

            ## Observation normalization
            normalized_observation = rescale_observations(observation, obs_limits)
            dist, values = network.apply(state.params, normalized_observation)
            # Calculating logprob separately can cause numerical issues
            # https://github.com/deepmind/distrax/issues/7
            actions, log_prob = dist.sample_and_log_prob(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = log_prob
            mem = mem._replace(extras=mem.extras)
            state = state._replace(random_key=key)
            return actions, state, mem

        @jax.jit
        def eval_policy(state: TrainingState, observation: Dict, mem: MemoryState):
            """passthrough, just the policy with empty extras"""
            actions, state, mem = policy(state, observation, mem)
            extras = {}
            return actions, state, mem, extras

        @jax.jit
        def gae_advantages(
            rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculates the gae advantages from a sequence. Note that the
            arguments are of length = rollout length + 1"""
            # 'Zero out' the terminated states
            discounts = gamma * jnp.logical_not(dones)

            # Go in reverse from T to 0 b/c this way we can do R_t^discounted = R_t + gamma * R_{t+1}^discounted
            reverse_batch = (
                jnp.flip(values[:-1], axis=0),
                jnp.flip(rewards, axis=0),
                jnp.flip(discounts, axis=0),
            )

            _, advantages = jax.lax.scan(
                get_advantages,
                (
                    jnp.zeros_like(values[-1]),
                    values[-1],
                    jnp.ones_like(values[-1]) * gae_lambda,
                ),
                reverse_batch,
            )

            advantages = jnp.flip(advantages, axis=0)
            target_values = values[:-1] + advantages  # Q-value estimates
            target_values = jax.lax.stop_gradient(target_values)
            return advantages, target_values

        def loss(
            params: hk.Params,
            timesteps: int,
            observations: jnp.ndarray,
            actions: jnp.array,
            behavior_log_probs: jnp.array,
            target_values: jnp.array,
            advantages: jnp.array,
            behavior_values: jnp.array,
        ):
            """Surrogate loss using clipped probability ratios."""
            normalized_observations = rescale_observations(observations, obs_limits)
            distribution, values = network.apply(params, normalized_observations)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            # Compute importance sampling weights: current policy pi_theta(a_t|s_t) / behavior policy pi_theta_k(a_t|s_t).
            # a/b = exp(log(a) - log(b)), but the latter is numerically better for gradient ascent (nicer derivatives b/c add>>>mult)
            ratios = jnp.exp(log_prob - behavior_log_probs)

            # Policy loss: Clipping
            clipped_ratios_t = jnp.clip(
                ratios, 1.0 - ppo_clipping_epsilon, 1.0 + ppo_clipping_epsilon
            )  # clip ratios within [1-eps, 1+eps] to prevent large updates
            clipped_objective = jnp.fmin(
                ratios * advantages, clipped_ratios_t * advantages
            )  # take the min of the clipped and unclipped objective (conservative)
            policy_loss = -jnp.mean(
                clipped_objective
            )  # minus b/c it's positive & we want to maximize, but adam minimizes

            # Value loss: MSE
            value_cost = value_coeff
            unclipped_value_error = target_values - values
            unclipped_value_loss = unclipped_value_error**2

            # Value clipping
            if clip_value:
                # Clip values to reduce variablility during critic training.
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -ppo_clipping_epsilon,
                    ppo_clipping_epsilon,
                )
                clipped_value_error = target_values - clipped_values
                clipped_value_loss = clipped_value_error**2
                value_loss = jnp.mean(
                    jnp.fmax(unclipped_value_loss, clipped_value_loss)
                )
            else:
                value_loss = jnp.mean(unclipped_value_loss)

            # Entropy loss: Standard entropy term
            # Calculate the new value based on linear annealing formula
            if anneal_entropy == "linear":
                fraction = jnp.fmax(1 - timesteps / entropy_coeff_horizon, 0)
                entropy_cost = (
                    fraction * entropy_coeff_start + (1 - fraction) * entropy_coeff_end
                )
            elif anneal_entropy == "exponential":
                decay_rate = (entropy_coeff_end / entropy_coeff_start) ** (
                    1 / entropy_coeff_horizon
                )
                entropy_cost = entropy_coeff_start * (decay_rate**timesteps)
                if entropy_clipping:
                    entropy_cost = jnp.clip(entropy_cost, entropy_coeff_end)
            # Constant Entropy term
            else:
                entropy_cost = entropy_coeff_start
            entropy_loss = -jnp.mean(entropy)

            # Total loss is minimized: Minimize value loss (positive); maximize advantages and entropy (hence negative)
            total_loss = (
                policy_loss + entropy_cost * entropy_loss + value_loss * value_cost
            )

            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
                "entropy_cost": entropy_cost,
            }

        @jax.jit
        def sgd_step(
            state: TrainingState, sample: NamedTuple
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics.
            Args:
                state's arrays assumed [num_envs, ...] (see make_initial_state)
                sample is a NamedTuple with arrays [num_steps, num_envs, ..]
            Outputs:
                new_state: TrainingState with arrays [num_envs, ...]
                new_memory: MemoryState with arrays [num_envs, ...] (hidden: [n_e, 1]; extras['norm_grad']: [n_e]; extras['norm_updates']: [n_e])
                metrics: Dict[str, jnp.ndarray]
            """

            # Extract data
            (
                observations,
                actions,
                rewards,
                behavior_log_probs,
                behavior_values,
                dones,
            ) = (
                sample.observations,
                sample.actions,
                sample.rewards,
                sample.behavior_log_probs,
                sample.behavior_values,
                sample.dones,
            )

            advantages, target_values = gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_steps, num_envs, ..]
            behavior_values = behavior_values[
                :-1, :
            ]  # values from "behaviour policy" i.e. the policy that generated the data
            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            # Concatenate all trajectories. Reshape from [num_steps, num_envs, ..]
            # to [num_envs * num_steps,..]
            # TODO: is this kosher? do we keep episodes together?
            assert len(target_values.shape) > 1
            num_envs = target_values.shape[1]
            num_steps = target_values.shape[0]
            batch_size = num_envs * num_steps
            assert batch_size % num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            # Compute gradients.
            grad_fn = jax.jit(jax.grad(loss, has_aux=True))

            @jax.jit
            def model_update_minibatch(
                carry: Tuple[hk.Params, optax.OptState, int],
                minibatch: Batch,
            ) -> Tuple[Tuple[hk.Params, optax.OptState, int], Dict[str, jnp.ndarray]]:
                """Performs model update for a single minibatch."""
                params, opt_state, timesteps = carry
                # Normalize advantages at the minibatch level before using them.
                advantages = (
                    minibatch.advantages - jnp.mean(minibatch.advantages, axis=0)
                ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
                gradients, metrics = grad_fn(
                    params,
                    timesteps,
                    minibatch.observations,
                    minibatch.actions,
                    minibatch.behavior_log_probs,
                    minibatch.target_values,
                    advantages,
                    minibatch.behavior_values,
                )
                # Apply updates
                updates, opt_state = optimizer.update(gradients, opt_state)
                params = optax.apply_updates(params, updates)

                metrics["norm_grad"] = optax.global_norm(gradients)
                metrics["norm_updates"] = optax.global_norm(updates)
                return (params, opt_state, timesteps), metrics

            @jax.jit
            def model_update_epoch(
                carry: Tuple[jnp.ndarray, hk.Params, optax.OptState, int, Batch],
                unused_t: Tuple[()],
            ) -> Tuple[
                Tuple[jnp.ndarray, hk.Params, optax.OptState, Batch],
                Dict[str, jnp.ndarray],
            ]:
                """Performs model updates based on one epoch of data."""
                key, params, opt_state, timesteps, batch = carry
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )

                (params, opt_state, timesteps), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state, timesteps),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, timesteps, batch), metrics

            params = state.params
            opt_state = state.opt_state
            timesteps = state.timesteps

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            # signature is scan(function, carry, tuple to iterate over, length)
            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (state.random_key, params, opt_state, timesteps, batch),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["sgd_steps"] = opt_state[1][0]
            metrics["scheduler_steps"] = opt_state[2][0]
            metrics["rewards_mean"] = jnp.mean(jnp.abs(jnp.mean(rewards, axis=(0, 1))))
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))
            metrics["trained"] = True
            metrics["mean_advantages"] = jnp.mean(advantages)
            metrics["var_advantages"] = jnp.var(advantages)

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps + batch_size,
            )

            new_memory = MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "log_probs": jnp.zeros(num_envs),
                    "values": jnp.zeros(num_envs),
                },
            )

            return new_state, new_memory, metrics

        def make_initial_state(
            key: Any, hidden: jnp.ndarray
        ) -> Tuple[TrainingState, MemoryState]:
            """Initialises the training state (parameters and optimiser state).
            Expects unbatched input key (training), returns batched output
            In:
                key: single key (unbatched, for agent training init)
                hidden: array [...] (doesn't matter)
            Out:
                TrainingState (NOT batched!): params [..], opt_state [..], random_key: Key, timesteps: int
                MemoryState: hidden: [num_envs, 1], extras['values']: [num_envs] , extras['log_probs']: [num_envs]
            """
            key, subkey = jax.random.split(key)  # this NEEDS a single key! (unbatched)

            if isinstance(obs_spec, dict):
                dummy_obs = {}
                for k, v in obs_spec.items():
                    if v == ():
                        dummy_obs[k] = jnp.zeros(1)
                    else:
                        dummy_obs[k] = jnp.zeros(shape=v)
                    # print(
                    #     f"{k}: {dummy_obs[k]} with shape {v} resulting in shape {dummy_obs[k].shape}"
                    # )

            elif not tabular:
                dummy_obs = jnp.zeros(shape=obs_spec, dtype=float_precision)
                dummy_obs = dummy_obs.at[0].set(1)
                dummy_obs = dummy_obs.at[9].set(1)
                dummy_obs = dummy_obs.at[18].set(1)
                dummy_obs = dummy_obs.at[27].set(1)
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)

            dummy_obs = add_batch_dim(dummy_obs)  # adds batch dim at axis 0
            # print(f"obs_spec: {obs_spec}")
            # print(f"subkey: {subkey} shape {subkey.shape}, dummy_obs: {dummy_obs}")
            initial_params = network.init(
                subkey, dummy_obs
            )  # network seems to expect input (obs) with batch dim at axis 0
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        def prepare_batch(traj_batch: NamedTuple, done: Any, action_extras: dict):
            """Prepare a batch of data for training.
            IN: (dims are for each vmapped slice)
                traj_batch: NamedTuple of obs (itself a dict), actions, rewards, behaviour_log_probs, behavior_values, dones, hiddens. Arrays have dim [num_timesteps, num_envs] except hidden which has an extra trailing 1
                done: Any, dim [num_envs]
                action_extras: dim [num_envs]"""
            # Rollouts complete -> Training begins
            # Add an additional rollout step for advantage calculation

            # Set final value estimate to 0 for bootstrapping advantages (pi_theta_k(a_T | s_T))
            _value = jax.lax.select(
                done,
                jnp.zeros_like(action_extras["values"]),
                action_extras["values"],
            )  # should be dim [n_envs]

            _value = jax.lax.expand_dims(_value, [0])  # should be [1, n_envs]
            # need to add final value here
            traj_batch = traj_batch._replace(
                behavior_values=jnp.concatenate(
                    [traj_batch.behavior_values, _value], axis=0
                )
            )
            return traj_batch

        # Initialise training state (parameters, optimiser state, extras).
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(random_key, jnp.zeros(1))
        self._prepare_batch = jax.jit(prepare_batch)
        self._sgd_step = jax.jit(sgd_step)

        # Set up counters and logger
        self._logger = Logger()  # atm a class that contains only the `metrics` dict
        # self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {
            "trained": True,
            # "total_steps": 0,  # TODO: these aren't tracked
            "sgd_steps": 0,  # TODO: these are tracked, but updating doesn't work (b/c they're not traced?)
            "scheduler_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
            "norm_grad": 0,
            "norm_updates": 0,
            "rewards_mean": 0,  # unused
            "rewards_std": 0,  # unused
            "mean_advantages": 0,
            "var_advantages": 0,
            "learning_rate": 0,
        }

        # Initialize functions
        self._policy = policy
        self.player_id = player_id
        self.network = network
        self._eval_policy = eval_policy

        # Other useful hyperparameters
        self._num_envs = num_envs  # number of environments
        self._num_minibatches = num_minibatches  # number of minibatches
        self._num_epochs = num_epochs  # number of epochs to use sample

        jax.debug.print("reset agent {} metrics", self.player_id)

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
        )
        return memory

    def update(
        self,
        traj_batch,  # NamedTuple of obs, actions, rewards, dones. Arrays have dim [num_timesteps, num_envs, ...]. ag2: [n_inner, n_envs,..], ag1: [n_outer*n_inner,n_opps*n_envs,..]
        obs: jnp.ndarray,  # dim [num_envs, obs_dims]
        state: TrainingState,  # dim [state_dims]
        mem: MemoryState,  # dim [num_envs, mem_dims]
    ):
        """Update the agent -> only called at the end of a trajectory
        Expects batched inputs for traj_batch, state, mem.
        traj_batch: [num_timesteps, batch_size, ...]
        obs: [batch_size, ...]
        state: [batch_size, ...]
        mem: sub-arrays all [batch_size, ...]"""
        _, _, mem = self._policy(
            state, obs, mem
        )  # value estimates of pi_theta_k(obs) are stored in mem.extras["values"], used for bootstrapping
        # if the state is "done", then the obs could be a reset one and shouldn't be used
        # therefore, in that case, prepare_batch() discards & overwrites the value estimate to 0

        # traj_batch.dones[-1, ...] is the done of the last step in the trajectory and should have shape [num_envs]
        traj_batch = self._prepare_batch(
            traj_batch, traj_batch.dones[-1, ...], mem.extras
        )

        state, mem, metrics = self._sgd_step(state, traj_batch)

        return state, mem, metrics

    def save_state(self, filepath: str):
        state_data = {
            "training_state": self._state,
            "memory_state": self._mem,
            "hyperparameters": {
                "player_id": self.player_id,
                "num_minibatches": self._num_minibatches,
                "num_epochs": self._num_epochs,
                "num_envs": self._num_envs,
            },
            # Include network parameters
            "network_params": self._state.params,
            "optimizer_state": self._state.opt_state,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state_data, f)

    @classmethod
    def load_state(
        cls, filepath: str, network, optimizer: optax.GradientTransformation
    ):
        with open(filepath, "rb") as f:
            state_data = pickle.load(f)

        print(f"Loading state from {filepath}")
        # Create a new instance without invoking __init__
        obj = cls.__new__(cls)

        obj._state = state_data["training_state"]
        obj._mem = state_data["memory_state"]

        obj.player_id = state_data["hyperparameters"]["player_id"]
        obj._num_envs = state_data["hyperparameters"]["num_envs"]
        obj._num_minibatches = state_data["hyperparameters"]["num_minibatches"]
        obj._num_epochs = state_data["hyperparameters"]["num_epochs"]

        obj.network = network
        return obj

    def tree_flatten(self):
        children = (
            self._state,
            self._mem,
            # self.optimizer,
        )
        # all other attributes are static:
        aux_data = {
            "player_id": self.player_id,
            "_lr_scheduling": self._lr_scheduling,
            "_initial_learning_rate": self._initial_learning_rate,
            "_total_num_transitions": self._total_num_transitions,
            "_num_envs": self._num_envs,
            "_num_minibatches": self._num_minibatches,
            "_num_epochs": self._num_epochs,
            "network": self.network,
            "_logger": self._logger,
            "_until_sgd": self._until_sgd,
            "_policy": self._policy,
            "_eval_policy": self._eval_policy,
            "_sgd_step": self._sgd_step,
            "make_initial_state": self.make_initial_state,
            "_prepare_batch": self._prepare_batch,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._state, obj._mem = children
        for key, value in aux_data.items():
            setattr(obj, key, value)
        return obj


jax.tree_util.register_pytree_node(PPO, PPO.tree_flatten, PPO.tree_unflatten)


def make_agent(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    num_iterations: int,
    player_id: int,
    tabular=False,
):
    """Make PPO agent"""
    print(f"Making network for {args.get('env_id')}")
    if (
        args.get("env_id") == "MarketEnv-v1"
        or args.get("env_id") == "MarketEnv-InfiniteInventoryEpisodic"
        or args.get("env_id") == "MarketEnv-InfiniteInventoryInfiniteEpisode"
    ):
        network = make_marketenv_network(
            action_spec,
            agent_args.get("separate"),
            agent_args.get("hidden_sizes"),
        )
    else:
        raise NotImplementedError(
            f"No ppo network implemented for env {args.get('env_id')}"
        )

    # Optimizer
    transition_steps = (
        num_iterations
        * agent_args.get("num_epochs")
        * agent_args.get("num_minibatches")
    )

    if agent_args.get("lr_scheduling"):
        scale = optax.inject_hyperparams(optax.scale)(step_size=-1.0)
        scheduler = optax.linear_schedule(
            init_value=agent_args.get("learning_rate"),
            end_value=0,
            transition_steps=transition_steps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.get("max_gradient_norm")),
            optax.scale_by_adam(eps=agent_args.get("adam_epsilon")),
            optax.scale_by_schedule(scheduler),
            scale,
        )
        # optimizer = optax.inject_hyperparams(optimizer, static_args="foo_LR")(
        #     foo_LR=agent_args.get("learning_rate")
        # )
        # optimizer = optax.inject_hyperparams(optimizer)(learning_rate=agent_args.learning_rate)

    else:
        scale = optax.inject_hyperparams(optax.scale)(
            step_size=-agent_args.get("learning_rate")
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.get("max_gradient_norm")),
            optax.scale_by_adam(eps=agent_args.get("adam_epsilon")),
            scale,
        )
        # optimizer = optax.inject_hyperparams(optimizer)(learning_rate=agent_args.learning_rate)

    # Random key
    random_key = jax.random.PRNGKey(seed=seed)

    # MarketEnv specific:
    obs_limits = {
        "inventory_uppers": jnp.array(args.get("initial_inventories")),
        "last_actions_upper": args.get("num_prices"),
        "last_prices_lower": args.get("possible_prices")[0],
        "last_prices_upper": args.get("possible_prices")[-1],
        "t_upper": args.get("time_horizon"),
    }

    agent = PPO(
        network=network,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        obs_limits=obs_limits,
        num_envs=args.get("num_envs"),
        num_minibatches=agent_args.get("num_minibatches"),
        num_epochs=agent_args.get("num_epochs"),
        clip_value=agent_args.get("clip_value"),
        value_coeff=agent_args.get("value_coeff"),
        anneal_entropy=agent_args.get("anneal_entropy"),
        entropy_coeff_start=agent_args.get("entropy_coeff_start"),
        entropy_coeff_end=agent_args.get("entropy_coeff_end"),
        entropy_coeff_horizon=agent_args.get("entropy_coeff_horizon"),
        ppo_clipping_epsilon=agent_args.get("ppo_clipping_epsilon"),
        entropy_clipping=agent_args.get("entropy_clipping"),
        gamma=agent_args.get("gamma"),
        gae_lambda=agent_args.get("gae_lambda"),
        tabular=tabular,
        player_id=player_id,
        num_iterations=num_iterations,
        initial_learning_rate=agent_args.get("learning_rate"),
        lr_scheduling=agent_args.get("lr_scheduling"),
    )
    return agent
