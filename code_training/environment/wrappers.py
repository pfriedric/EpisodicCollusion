import jax
import jax.numpy as jnp
from functools import partial

from environment.market_env import EnvState

# from market_env import EnvState
import chex

## The reward and obs wrappers can be stacked, b/c the step() method hands down the state.env_state
# into the wrapped self._env.step() method, which doesn't know it's wrapped. So if a "state" on the outside is a
# NormalizeVecRewEnvState(NormalizeVecObsEnvState(EnvState)) it has the attributes:
# mean, var, count, env_state=NormalizeVecObsEnvState(mean, var, count, env_state=EnvState())
# to get the actual state, I'd have to do state.env_state.env_state


class GymnaxWrapper(object):
    """
    Base class for Gymnax wrappers.

    This class provides a base for wrapping environments in Gymnax. It stores
    a reference to the original environment and allows proxy access to its
    attributes.

    Args:
        env (environment.Environment): The environment to be wrapped.
    """

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnv(GymnaxWrapper):
    """
    Wrapper to vectorize the environment.

    This wrapper uses jax.vmap to vectorize the reset and step functions,
    enabling parallel execution across multiple environments.

    Args:
        env (environment.Environment): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))  # key(+), params(-)
        self.step = jax.vmap(
            self._env.step, in_axes=(0, 0, 0, None)
        )  # key(+), EnvState (+), actions (+), params (-)


@chex.dataclass
class NormalizeDoubleVecObsEnvState:
    mean: dict
    var: dict
    count: float
    env_state: EnvState


class DummyDoubleVecObsWrapper(GymnaxWrapper):
    """dummy wrapper, doesn't do anything to rewards except correct piping"""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, key, params=None):
        obs, state = self._env.batch_reset(key, params)
        state = NormalizeDoubleVecObsEnvState(mean=None, var=None, count=None, env_state=state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, key, state, action, params=None):
        obs, env_state, rewards, done, info = self._env.batch_step(
            key, state.env_state, action, params
        )
        state = NormalizeDoubleVecObsEnvState(mean=None, var=None, count=None, env_state=env_state)
        return obs, state, rewards, done, info


class NormalizeDoubleVecObservation(GymnaxWrapper):
    """Assumes that env.batch_reset and env.batch_step are defined by double-vmapping env.reset, env.step

    I.e. the env is a normal MarketEnv, not a vector-wrapped one"""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, key, params=None):
        """is not vmapped, expects input & output to be [n_o, n_e, ...]"""
        all_obs, state = self._env.batch_reset(
            key, params
        )  # assumes env.batch_reset is double-vmapped and produces [n_o, n_e, ..]

        obs = all_obs[0]
        obs = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, axis=0), obs
        )  # squeeze leading dim to get [n_e]

        state = NormalizeDoubleVecObsEnvState(
            mean=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), obs),  # shape [n_envs, ..]
            var=jax.tree_util.tree_map(lambda x: jnp.ones_like(x), obs),  # [n_e, ..]
            count=1e-4,
            env_state=state,  # [n_o, n_e]
        )

        batch_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), obs)
        batch_var = jax.tree_util.tree_map(lambda x: jnp.var(x, axis=0), obs)
        batch_count = obs[next(iter(obs))].shape[0]  # n_envs

        delta = jax.tree_util.tree_map(lambda bm, sm: bm - sm, batch_mean, state.mean)
        total_count = batch_count + state.count

        new_mean = jax.tree_util.tree_map(
            lambda sm, delta: sm + delta * batch_count / total_count, state.mean, delta
        )  # [n_e, ..]
        m_a = jax.tree_util.tree_map(lambda x: x * state.count, state.var)
        m_b = jax.tree_util.tree_map(lambda x: x * batch_count, batch_var)
        M2 = jax.tree_util.tree_map(
            lambda ma, mb, d: ma + mb + jnp.square(d) * state.count * batch_count / total_count,
            m_a,
            m_b,
            delta,
        )
        new_var = jax.tree_util.tree_map(lambda x: x / total_count, M2)  # [n_e, ..]
        new_count = total_count

        state = NormalizeDoubleVecObsEnvState(
            mean=new_mean,  # [n_e, obs]
            var=new_var,  # [n_e, obs]
            count=new_count,  # float
            env_state=state.env_state,  # [n_o, n_e, obs]
        )

        normalized_obs = jax.tree_util.tree_map(
            lambda x, mean, var: (x - mean) / jnp.sqrt(var + 1e-8),
            obs,  # [n_e, obs]
            state.mean,  # [n_e, obs]
            state.var,  # [n_e, obs]
        )
        normalized_obs = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), normalized_obs
        )  # dict, all [n_o, n_e, obs_dim]
        all_normalized_obs = tuple([normalized_obs for _ in range(self.num_players)])

        return all_normalized_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, key, state, action, params=None):
        obs_tuple, env_state, reward, done, info = self._env.batch_step(
            key, state.env_state, action, params
        )
        obs = obs_tuple[0]  # they're all identical, so just pull out the first one

        obs = jax.tree_util.tree_map(
            lambda x: jnp.squeeze(x, axis=0), obs
        )  # squeeze leading dim to get [n_e]

        state = NormalizeVecObsEnvState(
            mean=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), obs),  # shape [n_envs, ..]
            var=jax.tree_util.tree_map(lambda x: jnp.ones_like(x), obs),  # [n_e, ..]
            count=1e-4,
            env_state=state,  # [n_o, n_e]
        )

        batch_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), obs)
        batch_var = jax.tree_util.tree_map(lambda x: jnp.var(x, axis=0), obs)
        batch_count = obs[next(iter(obs))].shape[0]  # n_envs

        delta = jax.tree_util.tree_map(lambda bm, sm: bm - sm, batch_mean, state.mean)
        total_count = batch_count + state.count

        new_mean = jax.tree_util.tree_map(
            lambda sm, delta: sm + delta * batch_count / total_count, state.mean, delta
        )  # [n_e, ..]
        m_a = jax.tree_util.tree_map(lambda x: x * state.count, state.var)
        m_b = jax.tree_util.tree_map(lambda x: x * batch_count, batch_var)
        M2 = jax.tree_util.tree_map(
            lambda ma, mb, d: ma + mb + jnp.square(d) * state.count * batch_count / total_count,
            m_a,
            m_b,
            delta,
        )
        new_var = jax.tree_util.tree_map(lambda x: x / total_count, M2)  # [n_e, ..]
        new_count = total_count

        state = NormalizeDoubleVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )

        normalized_obs = jax.tree_util.tree_map(
            lambda x, mean, var: (x - mean) / jnp.sqrt(var + 1e-8),
            obs,  # [n_e, obs]
            state.mean,  # [n_e, obs]
            state.var,  # [n_e, obs]
        )
        normalized_obs = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), normalized_obs
        )  # dict, all [n_o, n_e, obs_dim]

        all_normalized_obs = tuple([normalized_obs for _ in range(self.num_players)])

        return all_normalized_obs, state, reward, done, info


@chex.dataclass
class NormalizeDoubleVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float  # [batch_size, 1]
    env_state: EnvState


class DummyDoubleVecRewWrapper(GymnaxWrapper):
    """dummy wrapper, doesn't do anything to rewards except correct piping"""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, key, params=None):
        obs, state = self._env.batch_reset(key, params)
        state = NormalizeDoubleVecRewEnvState(
            mean=jnp.zeros((self.num_players,)),
            var=jnp.zeros((self.num_players,)),
            count=0,
            return_val=0,
            env_state=state,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, key, state, action, params=None):
        obs, env_state, rewards, done, info = self._env.batch_step(
            key, state.env_state, action, params
        )
        state = NormalizeDoubleVecRewEnvState(
            mean=state.mean,
            var=state.var,
            count=state.count,
            return_val=state.return_val,
            env_state=env_state,
        )
        return obs, state, rewards, 0, done, info


class NormalizeDoubleVecReward(GymnaxWrapper):
    """
    Wrapper to normalize the rewards of vectorized environments.

    This wrapper maintains running statistics (mean and variance) of the
    rewards and normalizes them during the step function. It also applies a
    discount factor to the rewards.

    Args:
        env (environment.Environment): The environment to be wrapped.
        self._env calls the env 'one level up' in the wrapping (not necessarily the 'base' env!)
        gamma (float): Discount factor for rewards.
    """

    def __init__(self, env, gamma, clip):
        super().__init__(env)
        self.gamma = gamma
        self.clip = clip

    @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, key, params=None):
        obs, state = self._env.batch_reset(key, params)
        first_obs = obs[0]
        batch_count = first_obs[next(iter(first_obs))].shape[1]

        # flat_obs, _ = jax.tree_util.tree_flatten(obs)
        # batch_count = flat_obs[0].shape[1]  # n_envs
        state = NormalizeDoubleVecRewEnvState(
            mean=200 * jnp.ones((self.num_players)),  # jnp.zeros((self.num_players)),  # [n_a]
            var=jnp.zeros((self.num_players)),  # jnp.ones((self.num_players)),  # [n_a]
            count=1,  # 1e-4,
            return_val=jnp.zeros((batch_count, self.num_players)),  # [n_e, n_a]
            env_state=state,  # [n_o, n_e, ..]
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, key, state, action, params=None):
        obs, env_state, rewards, done, info = self._env.batch_step(
            key, state.env_state, action, params
        )  # these are all [n_o, n_e, ..]

        # squeeze leading dim to get [n_e]
        # obs_squeezed = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), obs)
        done_squeezed = jnp.squeeze(done, axis=0)  # [n_e]
        done_tiled = jnp.tile(done_squeezed, (1, self.num_players))  # [n_e, n_a]
        rewards_squeezed = jnp.squeeze(rewards, axis=0)  # [n_e, n_a]

        return_val = (
            state.return_val * self.gamma * (1 - done_squeezed) + rewards_squeezed
        )  # [n_e, a] * [] * (1-[n_e, n_a]) + [n_e, n_a] -> [n_e, n_a]

        batch_mean = jnp.mean(return_val, axis=0)  # [n_a]
        batch_var = jnp.var(return_val, axis=0)  # [n_a]

        first_obs = obs[0]  # (obs, obs, ..) -> obs
        batch_count = first_obs[next(iter(first_obs))].shape[
            1
        ]  # 0th in obs dict (=inventories): [n_o, n_e (!), ..] -> n_e

        # flat_obs, _ = jax.tree_util.tree_flatten(obs)
        # batch_count = flat_obs[0].shape[1]  # n_envs

        delta = batch_mean - state.mean  # [n_a]-[n_a] -> [n_a]
        tot_count = state.count + batch_count  # []

        new_mean = state.mean + delta * batch_count / tot_count  # [n_a]+[n_a]*[]/[]
        m_a = state.var * state.count  # [n_a] | start 1
        m_b = batch_var * batch_count  # [n_a] | start ~10??
        M2 = (
            m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        )  # start 11 + 200*0.0001*4/4
        new_var = M2 / (tot_count)
        new_count = tot_count

        state = NormalizeDoubleVecRewEnvState(
            mean=new_mean,  # [n_a]
            var=new_var,  # [n_a]
            count=new_count,  # []
            return_val=return_val,  # [n_e, n_a]
            env_state=env_state,  # [n_o, n_e, ..]
        )

        # rescaled_rewards = rewards / jnp.sqrt(state.var + 1e-4)  # [n_o, n_e, n_a]
        # rescaled_rewards = jnp.log(rewards)
        rescaled_rewards = rewards / self.time_horizon
        # rescaled_rewards = rewards / 200
        # rescaled_rewards = rewards

        clipped_rewards = jnp.clip(rescaled_rewards, -self.clip, self.clip)

        return obs, state, clipped_rewards, rewards, done, info
        # return obs, state, rewards, done, info


@chex.dataclass
class NormalizeVecObsEnvState:
    """Stores vectorized env state"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: EnvState


class NormalizeVecObservation(GymnaxWrapper):
    """
    Wrapper to normalize the observations of vectorized environments.

    This wrapper maintains running statistics (mean and variance) of the
    observations and normalizes them during the reset and step functions.

    The i-th obs entry (e.g. inventory of agent i) is normalized
    by the mean and stddev of the i-th entry across the batch.

    Args:
        env (environment.Environment): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        """Default implementation assumed that env is VecEnv, so self._env.reset is vmapped but self.reset() is NOT vmapped.
        So the below code assumes that obs and state have [n_batch, ...] shapes."""
        obs, state = self._env.reset(
            key, params
        )  # if env is VecEnv wrapped, this calls VecEnv.reset() (vmapped!) and returns [n_b, ...] obs, state.
        # when we normalize we do it element-wise,
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),  # shape [n_batch, obs_dims]
            var=jnp.ones_like(obs),
            count=1e-4,  # to not divide by 0 in mean
            env_state=state,
        )
        # batch_mean and _var have shape [obs_dims]. e.g. obs = [a, b] then it's [mean(a's), mean(b's)] over batch
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]  # =n_batch

        delta = batch_mean - state.mean  # same shape as obs.
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@chex.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float  # [batch_size, 1]
    env_state: EnvState


class NormalizeVecReward(GymnaxWrapper):
    """
    Wrapper to normalize the rewards of vectorized environments.

    This wrapper maintains running statistics (mean and variance) of the
    rewards and normalizes them during the step function. It also applies a
    discount factor to the rewards.

    Args:
        env (environment.Environment): The environment to be wrapped.
        self._env calls the env 'one level up' in the wrapping (not necessarily the 'base' env!)
        gamma (float): Discount factor for rewards.
    """

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(
            key, params
        )  # if VecEnv: reset() expects & returns [n_batch, ...] things
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


if __name__ == "__main__":
    # dummy observation in unbatched form. the batching will add 2 leading dimensions [n_opponents, n_envs] to everything
    dummy_obs = {
        "inventories": jnp.array(
            jax.random.randint(jax.random.PRNGKey(0), shape=(1, 5, 3), minval=0, maxval=10)
        ),  # shape [2], dtype int
        "last_prices": jnp.array(
            jax.random.uniform(jax.random.PRNGKey(0), shape=(1, 5, 2), minval=0, maxval=1)
        ),  # shape [2], dtype float
    }
    flat_obs, _ = jax.tree_util.tree_flatten(dummy_obs)
    print(flat_obs)
    bc = flat_obs[0].shape[1]
    print(bc)
    a = jnp.array([[408.19998, 247.68001]])
    print(jnp.var(a, axis=1))
