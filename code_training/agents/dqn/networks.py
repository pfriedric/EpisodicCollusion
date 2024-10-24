import haiku as hk
from typing import Optional
from jaxtyping import Array, Integer
import jax
import jax.numpy as jnp


class QNetwork_marketenv(hk.Module):
    """Q-Network for the market environment.
    Assumes that inputs are a dictionary with keys "inventories", "last_actions", and "t".
    """

    def __init__(self, num_actions: int, hidden_sizes, name: Optional[str] = None):
        super().__init__(name=name)
        self._body = hk.nets.MLP(hidden_sizes, activation=jax.nn.relu, name="body")
        self._q_value_layer = hk.Linear(num_actions, name="q_values")

    def __call__(self, inputs):
        inventories = inputs["inventories"]
        last_actions = inputs["last_actions"]
        last_prices = inputs["last_prices"]
        t = inputs["t"]
        # concatenate all inputs, remembering that t is a scalar
        ## inventory constrained:
        obs = jnp.concatenate(
            [last_prices, inventories, t], axis=-1
        )  # for inventory constrained envs
        ## non-inventory constrained:
        # obs = jnp.concatenate([last_prices, t], axis=-1)
        # try: t, T-t, t/T, 1-t/T -> in dqn.py's obs rescaling
        # obs = last_prices  # for non-inventory constrained envs

        # concatenate all inputs
        x = self._body(obs)
        q_values = self._q_value_layer(x)
        return q_values


def make_dqn_marketenv_network(
    num_actions: int,
    hidden_sizes: Integer[Array, "..."],
):
    """Makes a DQN network for the market environment."""

    def forward_fn(inputs: dict):
        layers = []
        body = QNetwork_marketenv(
            num_actions=num_actions, hidden_sizes=hidden_sizes, name="market_q_network"
        )
        layers.extend([body])
        q_network = hk.Sequential(layers)
        return q_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network
