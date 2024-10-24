from agents.agent import AgentInterface
from utils import Logger, MemoryState, TrainingState


import jax
import jax.numpy as jnp


from typing import Callable, NamedTuple
from typing import Union

### This file defines 'dumb' agents with fixed strategies ###


def initial_state_fun(num_envs: int) -> Callable:
    def fun(key, hidden):
        return (
            TrainingState(None, None, jax.random.PRNGKey(0), None),
            MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            ),
        )

    return fun


def reset_mem_fun(num_envs: int) -> Callable:
    def fun(memory, eval=False):
        memory = memory._replace(
            extras={
                "values": jnp.zeros(1 if eval else num_envs),
                "log_probs": jnp.zeros(1 if eval else num_envs),
            },
        )
        return memory

    return fun


class Deterministic(AgentInterface):
    # This class gets initialized with a certain price or action value and will always play that value
    def __init__(
        self,
        num_actions: int,
        num_envs: int,
        obs_shape: Union[dict, tuple],
        fixed_action: int,
    ):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self.reset_memory = reset_mem_fun(num_envs)
        self._logger = Logger()
        self._logger.metrics = {}
        self._num_actions = num_actions

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:
            # state is [batch x time_step x num_players]
            # return [batch]
            if isinstance(obs_shape, dict):
                batch_size = obs["inventories"].shape[0]
            else:
                batch_size = obs.shape[0]
            action = jnp.ones(batch_size) * fixed_action
            return action.astype(jnp.int32), state, mem

        self._policy = jax.jit(_policy)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class Random(AgentInterface):
    def __init__(self, num_actions: int, num_envs: int, obs_shape: Union[dict, tuple]):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self.reset_memory = reset_mem_fun(num_envs)
        self._logger = Logger()
        self._logger.metrics = {}
        self._num_actions = num_actions

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:
            # state is [batch x time_step x num_players]
            # return [batch]
            if isinstance(obs_shape, dict):
                batch_size = obs["inventories"].shape[0]
            else:
                batch_size = obs.shape[0]
            new_key, _ = jax.random.split(state.random_key)
            action = jax.random.randint(new_key, (batch_size,), 0, num_actions)
            state = state._replace(random_key=new_key)
            return action, state, mem

        self._policy = jax.jit(_policy)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem
