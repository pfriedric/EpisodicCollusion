import time
import logging
import shutil
import os
import pickle
from typing import Dict, Any
from copy import deepcopy
import collections.abc
from functools import partial
from hydra.core.hydra_config import HydraConfig

import wandb
import omegaconf
import hydra
import logging


import jax
import jax.numpy as jnp
import numpy as np

from experiment import global_setup, env_setup, agent_setup, watcher_setup, runner_setup
from utils import get_unique_run_name, get_size_in_megabytes, flatten_dict

config_path = "conf"
omegaconf.OmegaConf.register_new_resolver("product", lambda x, y: x * y)


def possible_prices_func(
    p_N=1.471, p_M=1.925, num_price_steps=15, xi=0.1, which_price_grid="constrained"
):
    """Uses Calvano's discretized interval of possible prices. Outputs list to store in omegaconf.
    if want p[k]=p_N, p[end-k]=p_M: use xi:= k/(N-1-2k)
    which_price_grid: if unconstrained, outputs default calvano price grid,
    but calculates the comp./coll. prices&action indices on that grid with the given p_N, p_M (which depend on initial inventories set in config)
    price grid is what the agents see, and the collusive/competitive action/price values are for plotting
    """
    if which_price_grid == "unconstrained":
        lower = 1.471 - xi * (1.925 - 1.471)
        upper = 1.925 + xi * (1.925 - 1.471)
    else:
        lower = p_N - xi * (p_M - p_N)
        upper = p_M + xi * (p_M - p_N)
    res = np.linspace(lower, upper, num_price_steps)
    # round res to 3 decimals
    res = np.round(res, 3)

    possible_price_nash = min(res, key=lambda x: abs(x - p_N))
    possible_action_nash: int = res.tolist().index(possible_price_nash)
    possible_price_monopolistic = min(res, key=lambda x: abs(x - p_M))
    possible_action_monopolistic: int = res.tolist().index(possible_price_monopolistic)

    return (
        res.tolist(),
        lower,
        upper,
        possible_action_nash,
        possible_action_monopolistic,
        float(possible_price_nash),
        float(possible_price_monopolistic),
    )


def equilibrium_profits(
    episode_length: int,
    price_nash: float,
    quantity_nash: int,
    price_monopolistic: float,
    quantity_monopolistic: int,
    marginal_costs: list,
):
    """Calculates the profits of the Nash equilibrium and the monopolistic price."""
    profits_nash = []
    profits_nash_episode = []
    profits_monopolistic = []
    profits_monopolistic_episode = []

    for cost_i in marginal_costs:
        profits_nash.append((price_nash - cost_i) * quantity_nash)
        profits_nash_episode.append(profits_nash[-1] * episode_length)
        profits_monopolistic.append((price_monopolistic - cost_i) * quantity_monopolistic)
        profits_monopolistic_episode.append(profits_monopolistic[-1] * episode_length)
    return (
        profits_nash,
        profits_nash_episode,
        profits_monopolistic,
        profits_monopolistic_episode,
    )


def calc_nash_price_and_quantity(constraint=1000):
    """this will import the GNEP solver and calculate the Nash equilibrium price for that setting.
    For now, assumes equal inventory sizes & thus equilibrium prices
    Manually adjust for now:
    - Unconstrained: 1.471, 470 | reward 221
    - Constrained at (420*T): 1.7588, 420 | reward 318"""
    if constraint > 470:
        price_nash = 1.471
        # quantity_nash = 470
    elif constraint == 455:
        price_nash = 1.617
    elif constraint == 440:
        price_nash = 1.693
    elif constraint == 425:
        price_nash = 1.74
    elif constraint == 420:
        price_nash = 1.7588
    elif constraint == 410:
        price_nash = 1.795
    elif constraint == 395:
        price_nash = 1.843
    elif constraint == 380:
        price_nash = 1.885
    elif constraint == 365:
        price_nash = 1.925
    elif constraint == 230:
        price_nash = 2.213
    else:
        print("constraint not found")
        exit()

    quantity_nash = int(min(470, constraint))
    return price_nash, quantity_nash


def calc_monopolistic_price_and_quantity():
    """this will import the GNEP solver and calculate the monopolistic price for that setting
    reward: 337"""
    price_monopolistic = 1.925  # Calvano setting placeholder
    quantity_monopolistic = 365
    return price_monopolistic, quantity_monopolistic


def calc_rewards_range(which_price_grid, constraint):
    """returns the lowest and highest reward achievable for an agent with inventory constraint, depending on the price grid
    output is meant to be used for setting the normalizing_rewards_min and normalizing_rewards_max in the config
    """
    if which_price_grid == "unconstrained":
        if constraint > 470:
            lowest_reward = 63
            highest_reward = 445
        elif constraint == 455:
            lowest_reward = 68
            highest_reward = 400
        elif constraint == 440:
            lowest_reward = 68
            highest_reward = 387
        elif constraint == 425:
            lowest_reward = 68
            highest_reward = 379
        elif constraint == 420:
            lowest_reward = 68
            highest_reward = 379
        elif constraint == 410:
            lowest_reward = 68
            highest_reward = 379
        elif constraint == 395:
            lowest_reward = 68
            highest_reward = 365
        elif constraint == 380:
            lowest_reward = 68
            highest_reward = 356
        elif constraint == 365:
            lowest_reward = 68
            highest_reward = 354
        elif constraint == 230:
            lowest_reward = 68
            highest_reward = 234

    if which_price_grid == "constrained":
        if constraint > 470:
            lowest_reward = 63
            highest_reward = 445
        elif constraint == 455:
            lowest_reward = 129
            highest_reward = 393
        elif constraint == 440:
            lowest_reward = 172
            highest_reward = 381
        elif constraint == 425:
            lowest_reward = 205
            highest_reward = 372
        elif constraint == 420:
            lowest_reward = 218
            highest_reward = 368
        elif constraint == 410:
            lowest_reward = 243
            highest_reward = 363
        elif constraint == 395:
            lowest_reward = 279
            highest_reward = 356
        elif constraint == 380:
            lowest_reward = 309
            highest_reward = 347
        elif constraint == 365:
            lowest_reward = 337
            highest_reward = 337
        elif constraint == 230:
            lowest_reward = 279
            highest_reward = 279

    return lowest_reward, highest_reward


def update_dict_recursively(cfg, update):
    for k, v in update.items():
        if v is not None and isinstance(v, collections.abc.Mapping):
            cfg[k] = update_dict_recursively(cfg.get(k, {}), v)
        else:
            cfg[k] = v
    return cfg


@partial(jax.jit, static_argnums=(1,))
def update_pytree_recursively(cfg, update):
    def update_leaf(cfg_leaf, update_leaf):
        return jnp.where(update_leaf is None, cfg_leaf, update_leaf)

    return jax.tree.map(update_leaf, cfg, update, is_leaf=lambda x: not isinstance(x, dict))


# modify config: pass --config-name or -cn.
@hydra.main(config_path=config_path, config_name="config", version_base=None)
def main(args):
    config_name = HydraConfig.get().job.config_name
    print(f"Config name: {config_name}")
    args.num_inner_steps = args.time_horizon
    if args.agent1 == "DQN" and args.agent2 == "DQN":
        args.normalizing_rewards_gamma = args.dqn_default.discount
    elif args.agent1 == "PPO" and args.agent2 == "PPO":
        args.normalizing_rewards_gamma = args.ppo_default.gamma
    elif not hasattr(args, "normalizing_rewards_gamma"):
        args.normalizing_rewards_gamma = None

    # get the collusive and competitive price. for now, assumes equal inventory sizes & thus equilibrium prices

    args.normalizing_rewards_min, args.normalizing_rewards_max = calc_rewards_range(
        args.which_price_grid, args.initial_inventories[0]
    )
    price_nash, quantity_nash = calc_nash_price_and_quantity(args.initial_inventories[0])
    price_monopolistic, quantity_monopolistic = calc_monopolistic_price_and_quantity()
    # Scale (per-step) initial inventories with time horizon
    args.initial_inventories = [inv * args.time_horizon for inv in args.initial_inventories]

    # adjust run name
    args.wandb.name = get_unique_run_name(args.wandb.name, args.wandb.project, args.wandb.entity)

    # calc price vector and set it in args:
    if args.possible_prices == None:
        (
            args.possible_prices,
            args.min_price,
            args.max_price,
            args.competitive_action,
            args.collusive_action,
            args.competitive_price,
            args.collusive_price,
        ) = possible_prices_func(
            price_nash,
            price_monopolistic,
            num_price_steps=args.num_prices,
            xi=args.xi,
            which_price_grid=args.which_price_grid,
        )

    # use the env params to calculate equilibrium profits
    (
        args.competitive_profits,
        args.competitive_profits_episodetotal,
        args.collusive_profits,
        args.collusive_profits_episodetotal,
    ) = equilibrium_profits(
        args.num_inner_steps,  # "episode length"
        args.competitive_price,
        quantity_nash,
        args.collusive_price,
        quantity_monopolistic,
        args.marginal_costs,
    )

    args.dqn_default.epsilon_anneal_time = int(
        args.dqn_default.epsilon_anneal_duration
        * args.num_iters
        * args.num_envs
        * args.num_outer_steps
        * args.num_inner_steps
    )
    args.ppo_default.entropy_coeff_horizon = int(
        args.ppo_default.entropy_anneal_duration
        * args.num_iters
        * args.num_envs
        * args.num_outer_steps
        * args.num_inner_steps
    )

    os.environ["HYDRA_FULL_ERROR"] = "0"
    jax.config.update("jax_traceback_filtering", "off")
    # jax.config.update("jax_default_device", jax.devices("cpu")[0])
    print(f"Jax backend: {jax.extend.backend.get_backend().platform}")
    print(omegaconf.OmegaConf.to_yaml(args))

    """Set up main"""
    logger = logging.getLogger()

    save_dir = global_setup(args)
    print(save_dir)
    # Strip './exp/' from the beginning of save_dir and add './conf/archive/'
    config_save_path = os.path.join("./conf/archive/", save_dir[6:])
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    shutil.copy2(f"conf/{config_name}.yaml", config_save_path)

    print(f"Config file saved to: {config_save_path}")
    env, env_params = env_setup(args, logger)
    assert args.time_horizon == args.num_inner_steps

    print(f"num actions: {env.num_actions}")

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, env.observation_space(env_params))
    obs_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, env.observation_space(env_params))
    print(f"obs shape: {obs_shape}")
    print(f"obs dtypes: {obs_dtypes}")

    watchers = watcher_setup(args, logger)
    if not args.wandb.log:
        watchers = False
    print(f"Watchers: {watchers}")

    agent_list = agent_setup(args, env, env_params, logger)
    # print(agent_list[0]._state.shape)
    runner = runner_setup(args, env, agent_list, save_dir, logger)

    print(f"Number of Training Iterations: {args.num_iters}")

    """Uses the runner with gridsearch."""

    # load the update dict
    update_dict = omegaconf.OmegaConf.to_container(args.gridsearch)

    doing_gridsearch = (
        any(leaf is not None for leaf in jax.tree_util.tree_leaves(update_dict))
        or args.get("num_seeds") > 1
    )
    print(f"Doing gridsearch: {doing_gridsearch}")

    # make a meshgrid
    update_dict = jax.tree_util.tree_map(
        lambda x: jnp.array(x), update_dict, is_leaf=lambda x: isinstance(x, list)
    )
    leaves, treedef = jax.tree_util.tree_flatten(
        update_dict, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    leaves_idx = [jnp.arange(len(leaf)) for leaf in leaves]
    meshgrid = jnp.meshgrid(*leaves_idx)  # , indexing='ij')
    update_dict = jax.tree_util.tree_map(
        lambda idx, x: x[idx.reshape(-1), ...],
        jax.tree_util.tree_unflatten(treedef, meshgrid),
        update_dict,
    )
    print(f"Update dict: {update_dict}")

    # args must be a regular dict if it's to be updated within vmap scope
    args = omegaconf.OmegaConf.to_container(args, resolve=True)

    # dump the args to a file in the save_dir
    args_path = os.path.join(save_dir, "args.pkl")
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
        print(f"--- Args saved to {args_path} ---")

    def run_experiment(seed: int, update_dict: Dict[str, Any]):
        """Input: (vmapped) rng and update_dict, each value should be a scalar since it's vmapped.
        Output:
        - agents: 2-tuple of agents. each leaf has leading dims [num_rngs, num_configs].
        - log_data: 5-tuple of env_stats, rewards1/2, agent_metrics1/2. each of those has
        extra leading dims [num_seeds, num_configs]."""
        # with profiler.StepTraceAnnotation("setup"):
        args_exp = deepcopy(args)
        # could try the update_pytree_recursively here for JIT
        # would need to first do args_exp_jax = jax.tree.map(jnp.array, args_exp)
        # and update_dict_jax = jax.tree.map(lambda x: jnp.array(x) if x is not None else None, update)
        args_exp = update_dict_recursively(args_exp, update_dict)
        args_exp["seed"] = seed  # works unless JITted
        args_exp["dqn_default"]["epsilon_anneal_time"] = int(
            args_exp["dqn_default"]["epsilon_anneal_duration"]
            * args_exp["num_iters"]
            * args_exp["num_envs"]
            * args_exp["num_outer_steps"]
            * args_exp["num_inner_steps"]
        )
        args_exp["ppo_default"]["entropy_coeff_horizon"] = int(
            args_exp["ppo_default"]["entropy_anneal_duration"]
            * args_exp["num_iters"]
            * args_exp["num_envs"]
            * args_exp["num_outer_steps"]
            * args_exp["num_inner_steps"]
        )
        # epsi = args_exp["dqn_default"]["epsilon_finish"]
        # tie = args_exp["dqn_default"]["target_update_interval_episodes"]
        # lr = args_exp["dqn_default"]["learning_rate"]
        # tui = args_exp["dqn_default"]["training_interval_episodes"]
        agent_list = agent_setup(args_exp, env, env_params, logger)
        runner = runner_setup(args_exp, env, agent_list, save_dir, logger)
        # agent1 = agent_list[0]
        # ag1_epsi = agent1._epsilon_finish
        # ag1_tui = agent1._target_update_interval_episodes
        # return epsi, tie, lr, tui, ag1_epsi, ag1_tui
        # with profiler.StepTraceAnnotation("run_loop"):
        agents, log_data, init_rng = runner.run_loop(
            args_exp["seed"], env_params, agent_list, args_exp["num_iters"]
        )
        return agents, log_data, init_rng

    rng = jax.random.PRNGKey(args.get("seed"))

    print(f"--- Starting training loop ---")
    if doing_gridsearch:
        print(f"--- Running gridsearch ---")
        # vmap run_experiment over updated config (inner) and seeds (outer)
        run_experiment_vmap = jax.vmap(
            jax.vmap(
                run_experiment,
                in_axes=(None, jax.tree.map(lambda x: 0, update_dict)),
            ),
            in_axes=(0, None),
        )

        # if num_seeds >1, create list of seeds [args.seed, args.seed+num_players, ...] with len = num_seeds
        if args.get("num_seeds") > 1:
            seeds = jnp.array(
                [
                    args.get("seed") + i * args.get("num_players")
                    for i in range(args.get("num_seeds"))
                ]
            )
            print(f"seeds: {seeds}")
        else:
            seeds = jnp.array([args.get("seed")])

        # rngs = jax.random.split(rng, args.get("num_seeds"))
        run_time = time.time()

        # with profiler.trace("./jax-profile"):
        agents, log_data, init_rng = run_experiment_vmap(seeds, update_dict)
        jax.block_until_ready(agents)
        jax.block_until_ready(log_data)
        jax.block_until_ready(init_rng)

        update_dict_path = os.path.join(save_dir, "update_dict.pkl")
        with open(update_dict_path, "wb") as f:
            pickle.dump(update_dict, f)
            print(f"--- Update dict saved to {update_dict_path} ---")
        # dump seeds to a file in the save_dir
        seeds_path = os.path.join(save_dir, "seeds.pkl")
        with open(seeds_path, "wb") as f:
            pickle.dump(seeds, f)
            print(f"--- Seeds saved to {seeds_path} ---")

        # Flatten the update_dict: {"dqn_default" : {"learning_rate": [0.01, 0.02]}} -> {"dqn_default/learning_rate": [0.01, 0.02]}
        flattened_update_dict = flatten_dict(update_dict)

        # Create a mapping of flattened indices to original hyperparameter values
        hyperparam_mapping = []
        num_configs = len(next(iter(flattened_update_dict.values())))
        for idx in range(num_configs):
            config = {}
            for key, values in flattened_update_dict.items():
                config[key] = values[idx]
            hyperparam_mapping.append(config)
            # result: list of dicts, where j-th dict is a hyperparameter setting

        # Save the hyperparam_mapping to a file
        hyperparam_mapping_path = os.path.join(save_dir, "hyperparam_mapping.pkl")
        with open(hyperparam_mapping_path, "wb") as f:
            pickle.dump(hyperparam_mapping, f)
            print(f"--- Hyperparameter mapping saved to {hyperparam_mapping_path} ---")

        agent1, agent2 = agents
        agent1.save_state(os.path.join(save_dir, "agent_1_state.pkl"))
        agent2.save_state(os.path.join(save_dir, "agent_2_state.pkl"))
        print(f"--- Agents saved to {save_dir} ---")

    else:
        print(f"--- Running single training ---")
        run_time = time.time()
        # rng = jax.random.fold_in(rng, 1) # ensures same key as gridsearch with num_seeds=1
        seed = args.get("seed")
        agents, log_data, init_rng = runner.run_loop(
            seed, env_params, agent_list, args.get("num_iters")
        )
    print(f"--- Train time: {time.time() - run_time:.3f} seconds ---")

    # Dump log_data to a file in the save_dir
    log_data_path = os.path.join(save_dir, "log_data.pkl")
    with open(log_data_path, "wb") as f:
        pickle.dump(log_data, f)
        print(f"--- Log data saved to {log_data_path} ---")

    # if we have a saved log_data file, load it
    if os.path.exists(log_data_path):
        with open(log_data_path, "rb") as f:
            log_data = pickle.load(f)
            print(f"--- Log data loaded from {log_data_path} ---")

    # log the data
    log_time = time.time()
    print(f"log_data file size: {get_size_in_megabytes(log_data):.2f} MB")
    print(f"--- Starting logging ---")
    if doing_gridsearch:
        print(f"not logging gridsearch, use notebook for that")
        pass
        # runner.log_data(log_data, agents, args.get("num_iters"), watchers)
    else:
        runner.log_data(log_data, agents, args.get("num_iters"), watchers)

    print(f"--- Logging time: {time.time() - log_time:.3f} seconds ---")
    wandb.finish()


# endregion

if __name__ == "__main__":
    # print JAX device used
    print(f"jax devices: {jax.devices()}")
    start_time = time.time()
    main()
    print(f"--- Runtime: {time.time() - start_time:.3f} seconds ---")
