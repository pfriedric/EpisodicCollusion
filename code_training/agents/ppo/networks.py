from typing import Optional
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
from distrax import Categorical
from jaxtyping import Array, Integer


# region InTheMatrix
class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(num_values)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: Array):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return distrax.Categorical(logits=logits), value


# region MarketEnv
class CategoricalValueHeadSeparate_marketenv(hk.Module):
    """
    Network head that produces a categorical distribution and value.
    Assumes that inputs are a dictionary with keys "inventories", "last_actions", and "t".
    """

    def __init__(self, num_actions: int, hidden_sizes, name: Optional[str] = None):
        super().__init__(name=name)
        self._action_body = hk.nets.MLP(
            hidden_sizes,
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activation=jnp.tanh,  # or jax.nn.relu
            name="A",
        )
        self._value_body = hk.nets.MLP(
            hidden_sizes,
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activation=jnp.tanh,
            name="C",
        )
        self._logit_layer = hk.Linear(
            num_actions,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
            name="logits",
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            b_init=hk.initializers.Constant(0),
            name="value",
        )

        self._num_actions = num_actions

    def __call__(self, inputs):
        inventories = inputs["inventories"]
        last_prices = inputs["last_prices"]
        last_actions = inputs["last_actions"]
        t = inputs["t"]

        # obs = jnp.concatenate([inventories, last_actions, t], axis=-1)
        # obs = jnp.concatenate([inventories, last_actions], axis=-1)
        # obs = last_actions
        # obs = last_prices  # for non-inventory constrained envs
        obs = jnp.concatenate(
            [last_prices, inventories, t], axis=-1
        )  # for inventory constrained envs

        # Actor
        logits = self._action_body(obs)
        logits = self._logit_layer(logits)

        # Critic
        value = self._value_body(obs)
        value = self._value_layer(value)
        return (
            distrax.Categorical(logits=logits),
            jnp.squeeze(value, axis=-1),
        )


class Body_marketenv(hk.Module):
    """Body network for the market environment.
    Assumes that inputs are a dictionary with keys "inventories", "last_actions", and "t".
    """

    def __init__(self, hidden_sizes, name: Optional[str] = None):
        super().__init__(name=name)
        self._body = hk.nets.MLP(hidden_sizes, activation=jnp.tanh, name="body")

    def __call__(self, inputs):
        inventories = inputs["inventories"]
        last_actions = inputs["last_actions"]
        last_prices = inputs["last_prices"]
        t = inputs["t"]
        # concatenate all inputs, remembering that t is a scalar
        # obs = jnp.concatenate([inventories, last_actions], axis=-1)
        obs = last_prices

        # concatenate all inputs
        x = self._body(obs)
        return x


def make_marketenv_network(
    num_actions: int,
    separate: bool,
    hidden_sizes: Integer[Array, "..."],
):
    def forward_fn(inputs: dict):
        layers = []
        if separate:
            cvh = CategoricalValueHeadSeparate_marketenv(
                num_actions=num_actions,
                hidden_sizes=hidden_sizes,
                name="market_value_head_separate",
            )
            layers.extend([cvh])
        else:
            body = Body_marketenv(hidden_sizes=hidden_sizes, name="market_body")
            cvh = CategoricalValueHead(num_values=num_actions, name="market_value_head")
            layers.extend([body, cvh])

        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


@jax.jit
def log_inventories(observation: dict):
    """takes in observations dict, applies element wise natural log to inventories
    Args: observation: dict with keys 'inventories', 'last_actions', 't'
    Returns: observation: dict where 'inventories' has been elementwise log transformed
    """
    new_observation = observation.copy()
    new_observation["inventories"] = jnp.log(observation["inventories"])
    return new_observation


def test_PPO():
    key = jax.random.PRNGKey(0)
    num_actions = 5
    key, subkey = jax.random.split(key)

    dummy_obs = {
        "inventories": 1000 * jnp.ones(2),
        "last_prices": 1.4 * jnp.ones(2),
        "last_actions": -jnp.ones(2),
        "t": jnp.array([1]),
    }
    dummy_obs = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0), dummy_obs
    )  # batch dim
    network = make_marketenv_network(num_actions, separate=True, hidden_sizes=[16, 16])
    params = network.init(subkey, dummy_obs)
    print(params)
    for i in range(100):
        observation1 = {
            "inventories": jnp.array([10 * i, 10 * i]),
            "last_prices": jnp.array([1.4, 1.4]),
            "last_actions": jnp.array([-1, -1]),
            "t": jnp.array([0]),
        }
        observation_rescaled = log_inventories(observation1)
        dist, values = network.apply(params, observation_rescaled)
        actions, log_probs = dist.sample_and_log_prob(seed=subkey, sample_shape=(10,))
        print(f"sampling from obs: inventory = {10*i}")
        print(f"actions: {actions}")
        probs = jnp.round(dist.probs, 3)
        print(f"values: {values}, probs: {probs}, mode: {dist.mode()}")
        # print the probs of all possible actions
        # create vector of all possible actions, as integers from 0 to num_actions
    return network


# endregion

if __name__ == "__main__":
    test_PPO()
