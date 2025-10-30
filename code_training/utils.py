from jaxtyping import Array
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
import pickle
from typing import NamedTuple, Mapping
import chex
import wandb
import sys


# region UTILS
class Sample(NamedTuple):
    """This object contains a batch of data from the environment."""

    observations: Array
    actions: Array
    rewards: Array
    behavior_log_probs: Array
    behavior_values: Array
    hiddens: Array
    dones: Array


def add_batch_dim(values):
    """Add a batch dimension to all values in a tree."""
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), values)


def to_numpy(values):
    """Convert all values in a tree to numpy arrays."""
    return jax.tree_util.tree_map(np.asarray, values)


class TrainingState(NamedTuple):
    """Training state consists of network parameters, optimiser state, random key, timesteps"""

    params: hk.Params  # Is a dict of "layer_name": {'bias': Array, 'weight': Array}
    opt_state: optax.GradientTransformation
    random_key: Array
    timesteps: int


class MemoryState(NamedTuple):
    """State consists of network extras (to be batched)"""

    hidden: Array
    extras: Mapping[str, Array]


class Logger:
    metrics: dict

    def tree_flatten(self):
        children = (self.metrics,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.metrics = children
        return obj


jax.tree_util.register_pytree_node(Logger, Logger.tree_flatten, Logger.tree_unflatten)


def save(log: chex.ArrayTree, filename: str):
    """Save different parts of logger in .pkl file."""
    with open(filename, "wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename: str):
    """Reload the pickle logger and return dictionary."""
    with open(filename, "rb") as handle:
        es_logger = pickle.load(handle)
    return es_logger


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v.tolist() if isinstance(v, jax.Array) else v))
    return dict(items)


def get_advantages(carry, transition):
    """Calculate Generalised Advantage Estimation (GAE) for a batch of transitions.
    Output: G_t+1 = R_t - V_t + discount*V_t+1 + discount * gae_lambda * G_t"""
    gae, next_value, gae_lambda = carry
    value, reward, discounts = transition
    value_diff = discounts * next_value - value
    delta = reward + value_diff
    gae = delta + discounts * gae_lambda * gae
    return (gae, value, gae_lambda), gae


# TODO make this part of the args
float_precision = jnp.float32


def get_size_in_megabytes(obj):
    file_size = sys.getsizeof(pickle.dumps(obj))
    return file_size / (1024 * 1024)


def get_unique_run_name(base_name, project_name, entity):
    api = wandb.Api()
    # Assuming you have set your project and entity appropriately.
    runs = api.runs(f"{entity}/{project_name}")
    existing_names = {run.name for run in runs}

    # Check if the base name exists and append a number to make it unique
    if base_name in existing_names:
        i = 1
        new_name = f"{base_name}-{i}"
        while new_name in existing_names:
            i += 1
            new_name = f"{base_name}-{i}"
        return new_name
    else:
        return base_name
