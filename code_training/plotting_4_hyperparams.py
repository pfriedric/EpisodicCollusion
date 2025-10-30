# %%
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
from utils import flatten_dict
from typing import Dict, Any

from plotting_utils import display_single_plot

### Alter this for different runs. Options "DQN", "PPO"
DQN_or_PPO = "DQN"

plot_new = True
gen_mean_p = 0.5

save_dir = f"exp/fig4_{DQN_or_PPO}_plots"
os.makedirs(save_dir, exist_ok=True)


def retrieve_log_data_from_save_dir(dir_name):
    log_data = None
    try:
        with open(f"{dir_name}/log_data.pkl", "rb") as file:
            log_data = pickle.load(file)
        if isinstance(log_data, tuple):
            if len(log_data) == 2:
                log_data, _ = log_data
            elif len(log_data) == 3:
                log_data, _, _ = log_data
            else:
                raise ValueError("log_data tuple has an unexpected length")
        exists_log_data = True
        print(f"Loaded log_data from {dir_name}.")
        with open(f"{dir_name}/update_dict.pkl", "rb") as file:
            update_dict = pickle.load(file)
        with open(f"{dir_name}/args.pkl", "rb") as file:
            args = pickle.load(file)
    except FileNotFoundError:
        print(f"No log_data.pkl file found in {dir_name}.")
        exists_log_data = False
        update_dict = {}
        args = {}
    return log_data, update_dict, args, exists_log_data


# goal: for every hyperparameter, make a plot with 2 subplots:
# left subplot: boxplots. x: different values of hyperparam in question. y: boxplot (over seeds) of  convergence metric
# middle subplot: boxplots. x: different values of hyperparam in question. y: boxplot (over seeds) of collusion index with arithmetic mean
# right subplot: boxplots. x: different values of hyperparam in question. y: boxplot (over seeds) of collusion index with generalized mean

# gather log_data from the run:
try:
    eps_dir = os.path.join(
        f"exp/fig4_{DQN_or_PPO}_eps/",
        next(os.walk(f"exp/fig4_{DQN_or_PPO}_eps/"))[1][0],
    )
    print(f"found epsilon dir: {eps_dir}")
except Exception as e:
    print(f"No directory found for eps in exp/fig4_{DQN_or_PPO}_eps/, {e}")
    eps_exists_log_data = False
    eps_dir = ""
try:
    lr_dir = os.path.join(
        f"exp/fig4_{DQN_or_PPO}_lr/", next(os.walk(f"exp/fig4_{DQN_or_PPO}_lr/"))[1][0]
    )
    print(f"found learning_rate dir: {lr_dir}")
except Exception as e:
    print(f"No directory found for lr in exp/fig4_{DQN_or_PPO}_lr/, {e}")
    lr_exists_log_data = False
    lr_dir = ""
try:
    trainint_dir = os.path.join(
        f"exp/fig4_{DQN_or_PPO}_trainint/",
        next(os.walk(f"exp/fig4_{DQN_or_PPO}_trainint/"))[1][0],
    )
    print(f"found training_interval dir: {trainint_dir}")
except Exception as e:
    print(f"No directory found for trainint in exp/fig4_{DQN_or_PPO}_trainint/, {e}")
    trainint_exists_log_data = False
    trainint_dir = ""
try:
    ent_dir = os.path.join(
        f"exp/fig4_{DQN_or_PPO}_ent/",
        next(os.walk(f"exp/fig4_{DQN_or_PPO}_ent/"))[1][0],
    )
except Exception as e:
    print(f"No directory found for ent in exp/fig4_{DQN_or_PPO}_ent/, {e}")
    ent_exists_log_data = False
    ent_dir = ""

eps_log_data, eps_update_dict, eps_args, eps_exists_log_data = retrieve_log_data_from_save_dir(
    eps_dir
)
lr_log_data, lr_update_dict, lr_args, lr_exists_log_data = retrieve_log_data_from_save_dir(lr_dir)
trainint_log_data, trainint_update_dict, trainint_args, trainint_exists_log_data = (
    retrieve_log_data_from_save_dir(trainint_dir)
)
ent_log_data, ent_update_dict, ent_args, ent_exists_log_data = retrieve_log_data_from_save_dir(
    ent_dir
)


def generalized_mean(x, p=0.5, ax=1):
    """calculates an altered version of the generalized mean that is meant to better take into account negative values. input x is a multidimensional array."""
    ## altered version for multidimensional input
    res = np.mean(np.sign(x) * np.abs(x) ** p, axis=ax)
    res = np.maximum(res, 0)
    return res ** (1 / p)


def format_float(x):
    """Format float without trailing zeros after the decimal point."""
    try:
        return f"{x:.5f}".rstrip("0").rstrip(".")
    except:
        return str(x)


def rename_hyperparam(key):
    return (
        key.replace("dqn_default", "DQN")
        .replace("epsilon_finish", "epsilon")
        .replace("target_update_interval", "target_update")
        .replace("training_interval_episodes", "train_eps_interval")
        .replace("/", "_")
        .replace("ppo_default", "PPO")
        .replace("entropy_coeff_start", "entropy_start")
        .replace("num_minibatches", "minibatches")
        .replace("num_epochs", "epochs")
    )


def plot_hyperparam_boxplots(all_env_stats, param, param_values, param_args):
    log_interval = max(param_args["num_iters"] // 1000, 5 if param_args["num_iters"] > 1000 else 1)
    x_axis = np.arange(0, param_args["num_iters"], log_interval)
    last_ten_percent_episodes = param_args["num_iters"] // 10
    # print(f"last_ten_percent_episodes: {last_ten_percent_episodes}")
    # set up a plot, empty for now. 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"hspace": 0.33})

    # for each value of the hyperparam:
    for j, value in enumerate(param_values):
        # print(f"j: {j}, value: {value}")

        agent_profit_gains_seeds_timesteps = np.zeros(
            (
                param_args["num_seeds"],  # 100
                param_args["num_players"],  # 2
                all_env_stats["train/collusion_index/mean_player_1"].shape[2],  # 6k
            )
        )
        for agent_idx in range(param_args["num_players"]):
            agent_profit_gains_seeds_timesteps[:, agent_idx, :] = all_env_stats[
                f"train/collusion_index/mean_player_{agent_idx + 1}"
            ][:, j, :]  # [seeds, T]

        coll_idx_seeds_timesteps_arith = agent_profit_gains_seeds_timesteps.mean(
            axis=1
        )  # [seeds, T]
        coll_idx_seeds_timesteps_gen = generalized_mean(
            agent_profit_gains_seeds_timesteps, p=gen_mean_p
        )  # [seeds, T]

        coll_idx_per_seed_arith = np.mean(
            coll_idx_seeds_timesteps_arith[:, -last_ten_percent_episodes:], axis=1
        )
        coll_idx_per_seed_gen = np.mean(
            coll_idx_seeds_timesteps_gen[:, -last_ten_percent_episodes:], axis=1
        )

        # Using prices as mean over episode. shape [seeds, T]
        agent1_prices = all_env_stats[f"train/all_envs/mean_action/price_player_1"][:, j, :]
        agent2_prices = all_env_stats[f"train/all_envs/mean_action/price_player_2"][:, j, :]

        quotient_price_diff = np.abs(agent1_prices - agent2_prices) / (
            param_args["collusive_price"] - param_args["competitive_price"]
        )  # shape [seeds, T]
        convergence_vs_dispersion_metric_per_seed = np.mean(
            quotient_price_diff[:, -last_ten_percent_episodes:], axis=1
        )  # shape [seeds]

        # # Calculate training-run-based CI
        # (
        #     avg_training_collusion_index_per_seed,
        #     _,
        #     _,
        # ) = get_training_run_based_collusion_index(
        #     all_env_stats, agents, last_ten_percent_episodes, j
        # )

        # print(
        #     f"convergence_vs_dispersion_metric_per_seed stdev: {convergence_vs_dispersion_metric_per_seed.std()}"
        # )
        # Add boxplots to each subplot
        axs[0].boxplot(convergence_vs_dispersion_metric_per_seed, positions=[j])
        # Add a horizontal line at y=0.2
        axs[1].boxplot(coll_idx_per_seed_gen, positions=[j])
        axs[2].boxplot(coll_idx_per_seed_arith, positions=[j])
        # print(f"added subplots for index {j}")
        # Set x-ticks and labels
        for ax in axs:
            ax.set_xticks(range(len(param_values)))
            ax.set_xticklabels([format_float(value) for value in param_values])

    axs[0].axhline(
        y=0.2,
        color="blue",
        linestyle="dotted",
        linewidth=1,
        label="Convergence",
    )
    axs[1].axhline(
        y=0.05,
        color="blue",
        linestyle="dotted",
        linewidth=1,
        label="Collusion",
    )
    axs[2].axhline(
        y=0.05,
        color="blue",
        linestyle="dotted",
        linewidth=1,
        label="Collusion",
    )

    # Set labels and titles for each subplot
    metrics = [
        "Convergence vs Dispersion",
        f"Collusion index (generalized mean, p={gen_mean_p})",
        "Collusion index (arithmetic mean)",
    ]
    axs[0].set_ylabel("convergence metric")
    axs[1].set_ylabel("collusion index (gen. mean)")
    axs[2].set_ylabel("collusion index (arith. mean)")
    for i, ax in enumerate(axs):
        ax.set_title(f"{metrics[i]}")
        ax.set_xlabel(rename_hyperparam(param))
        ax.legend()

    # Add fixed hyperparameters information
    title = f"Comparisons for {rename_hyperparam(param)}. Distribution over {param_args['num_seeds']} seeds."
    plt.suptitle(title)
    plt.savefig(os.path.join(save_dir, f"{rename_hyperparam(param)}_boxplots.png"))
    print(f"Saved plot to {os.path.join(save_dir, f'{rename_hyperparam(param)}_boxplots.png')}")
    plt.close()
    # plt.show()

    return fig, axs


# lr
try:
    lr_env_stats = lr_log_data[0]
    lr_flattened_update_dict = flatten_dict(lr_update_dict)
    for key, value in lr_flattened_update_dict.items():
        lr_flattened_update_dict[key] = np.round(value, 5)
    lr_name = next(iter(lr_flattened_update_dict))
    lr_values = {
        param: list(dict.fromkeys(str(v) if isinstance(v, list) else v for v in values))
        for param, values in lr_flattened_update_dict.items()
    }
    print(f"lr_values: {lr_values}")

    # plot_hyperparam_boxplots(lr_env_stats, lr_name, lr_values[lr_name], lr_args)
except Exception as e:
    print(f"No log_data.pkl file found in {lr_dir}, {e}")

# eps
try:
    eps_env_stats = eps_log_data[0]
    eps_flattened_update_dict = flatten_dict(eps_update_dict)
    for key, value in eps_flattened_update_dict.items():
        eps_flattened_update_dict[key] = np.round(value, 5)
    eps_name = next(iter(eps_flattened_update_dict))
    eps_values = {
        param: list(dict.fromkeys(str(v) if isinstance(v, list) else v for v in values))
        for param, values in eps_flattened_update_dict.items()
    }
    print(f"eps_values: {eps_values}")

    # plot_hyperparam_boxplots(eps_env_stats, eps_name, eps_values[eps_name], eps_args)
except Exception as e:
    print(f"No log_data.pkl file found in {eps_dir}, {e}")

# trainint
try:
    trainint_env_stats = trainint_log_data[0]
    trainint_flattened_update_dict = flatten_dict(trainint_update_dict)
    trainint_name = next(iter(trainint_flattened_update_dict))
    trainint_values = {
        param: list(dict.fromkeys(str(v) if isinstance(v, list) else v for v in values))
        for param, values in trainint_flattened_update_dict.items()
    }
    print(f"trainint_values: {trainint_values}")

    # plot_hyperparam_boxplots(
    #     trainint_env_stats, trainint_name, trainint_values[trainint_name], trainint_args
    # )
except Exception as e:
    print(f"No log_data.pkl file found in {trainint_dir}, {e}")

# entropy
try:
    ent_env_stats = ent_log_data[0]
    ent_flattened_update_dict = flatten_dict(ent_update_dict)
    for key, value in ent_flattened_update_dict.items():
        ent_flattened_update_dict[key] = np.round(value, 5)
    ent_name = next(iter(ent_flattened_update_dict))
    ent_values = {
        param: list(dict.fromkeys(str(v) if isinstance(v, list) else v for v in values))
        for param, values in ent_flattened_update_dict.items()
    }
    print(f"ent_values: {ent_values}")

    # plot_hyperparam_boxplots(ent_env_stats, ent_name, ent_values[ent_name], ent_args)
except Exception as e:
    print(f"No log_data.pkl file found in {ent_dir}, {e}")


#### MULTIRUNS -- data spread across multiple sub-directories that are each their own run
def gather_spread_data(base_dir, param):
    """
    Gathers data for a hyperparameter from a directory of runs.

    Args:
        base_dir (str): The base directory containing subdirectories for each run.
        param (str): The hyperparameter to gather data for.

    Returns:
        dict: A dictionary with keys as the values of the hyperparameter and values as tuples of log_data, update_dict, and args.
    """
    all_data = {}
    for subdir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_path):
            log_data, update_dict, args, exists_log_data = retrieve_log_data_from_save_dir(
                full_path
            )
            if exists_log_data:
                if param == "inventory":
                    inventory = args.get("initial_inventories", None)
                    # print(f"inventory: {inventory}")
                    inventory = inventory[0] // args["time_horizon"]
                    # print(f"inventory: {inventory}")
                    # all_data is a dict, each key a value of the hyperparam that gets assigned that value's log_data
                    if inventory is not None:
                        all_data[inventory] = (log_data, update_dict, args)
                elif param == "time_horizon":
                    time_horizon = args.get("time_horizon", None)
                    if time_horizon is not None:
                        all_data[time_horizon] = (log_data, update_dict, args)
                elif param == "num_minibatches":
                    num_minibatches = args["ppo_default"]["num_minibatches"]
                    if num_minibatches is not None:
                        all_data[num_minibatches] = (log_data, update_dict, args)
                elif param == "num_epochs":
                    num_epochs = args["ppo_default"]["num_epochs"]
                    if num_epochs is not None:
                        all_data[num_epochs] = (log_data, update_dict, args)
                elif param == "buffer_size":
                    buffer_size = args["dqn_default"]["buffer_size"]
                    if buffer_size is not None:
                        all_data[buffer_size] = (log_data, update_dict, args)
                else:
                    raise ValueError(f"Invalid parameter: {param}")
    return all_data


# Add this code after the existing hyperparameter plots
try:
    inventory_base_dir = f"exp/fig4_{DQN_or_PPO}_inv_size"
    # inventory_data = gather_inventory_data(inventory_base_dir)
    inventory_data = gather_spread_data(inventory_base_dir, "inventory")
    # print(f"inventory_data: {inventory_data}")
except Exception as e:
    inventory_data = None
    print(f"Couldn't find inventory data {e}")

if inventory_data:
    inventory_values = sorted(inventory_data.keys())
    first_data = next(iter(inventory_data.values()))
    inventory_args = first_data[2]
    first_env_stats = first_data[0][0]

    # Prepare data for plotting
    # [0][0] for ->log_data -> env_stats
    inv_all_env_stats = {}
    for key in first_env_stats.keys():
        first_array = first_env_stats[key]
        inv_all_env_stats[key] = np.zeros(
            (first_array.shape[0], len(inventory_values), first_array.shape[2])
        )

    # don't want to plot the first inventory value -- interval too small, unstable learning
    inventory_values = inventory_values[1:]

    # Fill inv_all_env_stats
    for i, inventory in enumerate(inventory_values):
        log_data, _, _ = inventory_data[inventory]
        for key in inv_all_env_stats:
            inv_all_env_stats[key][:, i, :] = log_data[0][key].squeeze(axis=1)

    # Plot the inventory hyperparameter
    # plot_hyperparam_boxplots(
    #     inv_all_env_stats, "initial_inventories", inventory_values, inventory_args
    # )
else:
    print("No inventory data found.")

# for the hyperparams initial_inventories and time_horizon, we need to gather log_data from a bunch of different runs
# to do this, for each such run, load log_data, extract the necessary values, dump them in a csv to disk

try:
    time_horizon_base_dir = f"exp/fig4_{DQN_or_PPO}_time_horizon_multirun"
    time_horizon_data = gather_spread_data(time_horizon_base_dir, "time_horizon")
except Exception as e:
    time_horizon_data = None
    print(f"Couldn't find time horizon data {e}")

if time_horizon_data:
    time_horizon_values = sorted(time_horizon_data.keys())
    first_data = next(iter(time_horizon_data.values()))
    time_horizon_args = first_data[2]
    first_env_stats = first_data[0][0]

    # Prepare data for plotting
    # [0][0] for ->log_data -> env_stats
    th_all_env_stats = {}
    for key in first_env_stats.keys():
        first_array = first_env_stats[key]
        th_all_env_stats[key] = np.zeros(
            (first_array.shape[0], len(time_horizon_values), first_array.shape[2])
        )

    # Fill th_all_env_stats
    for i, time_horizon in enumerate(time_horizon_values):
        log_data, _, _ = time_horizon_data[time_horizon]
        for key in th_all_env_stats:
            th_all_env_stats[key][:, i, :] = log_data[0][key].squeeze(axis=1)

    # Plot the time_horizon hyperparameter
    # plot_hyperparam_boxplots(
    #     th_all_env_stats, "time_horizon", time_horizon_values, time_horizon_args
    # )
else:
    print("Plotting time horizon failed :(")


try:
    num_minibatches_base_dir = f"exp/fig4_{DQN_or_PPO}_mb_multirun"
    num_minibatches_data = gather_spread_data(num_minibatches_base_dir, "num_minibatches")
except Exception as e:
    num_minibatches_data = None
    print(f"Couldn't find minibatches data {e}")

if num_minibatches_data:
    num_minibatches_values = sorted(num_minibatches_data.keys())
    first_data = next(iter(num_minibatches_data.values()))
    num_minibatches_args = first_data[2]
    first_env_stats = first_data[0][0]

    # Prepare data for plotting
    # [0][0] for ->log_data -> env_stats
    num_minibatches_all_env_stats = {}
    for key in first_env_stats.keys():
        first_array = first_env_stats[key]
        num_minibatches_all_env_stats[key] = np.zeros(
            (first_array.shape[0], len(num_minibatches_values), first_array.shape[2])
        )

    # Fill num_minibatches_all_env_stats
    for i, num_minibatches in enumerate(num_minibatches_values):
        log_data, _, _ = num_minibatches_data[num_minibatches]
        for key in num_minibatches_all_env_stats:
            num_minibatches_all_env_stats[key][:, i, :] = log_data[0][key].squeeze(axis=1)

    # Plot the num_minibatches hyperparameter
    # plot_hyperparam_boxplots(
    #     num_minibatches_all_env_stats,
    #     "PPO_num_minibatches",
    #     num_minibatches_values,
    #     num_minibatches_args,
    # )
else:
    print("Plotting minibatches failed :(")


try:
    num_epochs_base_dir = f"exp/fig4_{DQN_or_PPO}_ep_multirun"
    num_epochs_data = gather_spread_data(num_epochs_base_dir, "num_epochs")
except Exception as e:
    num_epochs_data = None
    print(f"Couldn't find epochs data {e}")

if num_epochs_data:
    num_epochs_values = sorted(num_epochs_data.keys())
    first_data = next(iter(num_epochs_data.values()))
    num_epochs_args = first_data[2]
    first_env_stats = first_data[0][0]

    # Prepare data for plotting
    # [0][0] for ->log_data -> env_stats
    num_epochs_all_env_stats = {}
    for key in first_env_stats.keys():
        first_array = first_env_stats[key]
        num_epochs_all_env_stats[key] = np.zeros(
            (first_array.shape[0], len(num_epochs_values), first_array.shape[2])
        )

    # Fill num_epochs_all_env_stats
    for i, num_epochs in enumerate(num_epochs_values):
        log_data, _, _ = num_epochs_data[num_epochs]
        for key in num_epochs_all_env_stats:
            num_epochs_all_env_stats[key][:, i, :] = log_data[0][key].squeeze(axis=1)

    # Plot the num_epochs hyperparameter
    # plot_hyperparam_boxplots(
    #     num_epochs_all_env_stats, "PPO_num_epochs", num_epochs_values, num_epochs_args
    # )
else:
    print("Plotting num_epochs failed :(")

try:
    buffer_size_base_dir = f"exp/fig4_{DQN_or_PPO}_buffer_size_multirun"
    buffer_size_data = gather_spread_data(buffer_size_base_dir, "buffer_size")
except Exception as e:
    buffer_size_data = None
    print(f"Couldn't find buffer size data {e}")

if buffer_size_data:
    buffer_size_values = sorted(buffer_size_data.keys())
    first_data = next(iter(buffer_size_data.values()))
    buffer_size_args = first_data[2]
    first_env_stats = first_data[0][0]

    # Prepare data for plotting
    # [0][0] for ->log_data -> env_stats
    buffer_size_all_env_stats = {}
    for key in first_env_stats.keys():
        first_array = first_env_stats[key]
        buffer_size_all_env_stats[key] = np.zeros(
            (first_array.shape[0], len(buffer_size_values), first_array.shape[2])
        )

    # Fill buffer_size_all_env_stats
    for i, buffer_size in enumerate(buffer_size_values):
        log_data, _, _ = buffer_size_data[buffer_size]
        for key in buffer_size_all_env_stats:
            buffer_size_all_env_stats[key][:, i, :] = log_data[0][key].squeeze(axis=1)

    # Plot the buffer_size hyperparameter
    # plot_hyperparam_boxplots(
    #     buffer_size_all_env_stats, "buffer_size", buffer_size_values, buffer_size_args
    # )
else:
    print("Plotting buffer size failed :(")


def display_plots(plot_pattern):
    # sort by t
    for plot_path in sorted(glob.glob(plot_pattern)):
        display_single_plot(plot_path)


def load_and_display_plots():
    plot_pattern = os.path.join(save_dir, f"*_boxplots.png")
    display_plots(plot_pattern)


load_and_display_plots()

# %%
paper_dir = os.path.join(save_dir, "paper_plots")
os.makedirs(paper_dir, exist_ok=True)


def plot_paper_boxplots(all_env_stats, param, param_values, param_args):
    log_interval = max(param_args["num_iters"] // 1000, 5 if param_args["num_iters"] > 1000 else 1)
    x_axis = np.arange(0, param_args["num_iters"], log_interval)
    last_ten_percent_episodes = param_args["num_iters"] // 10

    # Set up a plot with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.3})

    # for each value of the hyperparam:
    for j, value in enumerate(param_values):
        agent_profit_gains_seeds_timesteps = np.zeros(
            (
                param_args["num_seeds"],
                param_args["num_players"],
                all_env_stats["train/collusion_index/mean_player_1"].shape[2],
            )
        )
        for agent_idx in range(param_args["num_players"]):
            agent_profit_gains_seeds_timesteps[:, agent_idx, :] = all_env_stats[
                f"train/collusion_index/mean_player_{agent_idx + 1}"
            ][:, j, :]

        coll_idx_seeds_timesteps_gen = generalized_mean(
            agent_profit_gains_seeds_timesteps, p=gen_mean_p
        )

        coll_idx_per_seed_gen = np.mean(
            coll_idx_seeds_timesteps_gen[:, -last_ten_percent_episodes:], axis=1
        )

        # Using prices as mean over episode. shape [seeds, T]
        agent1_prices = all_env_stats[f"train/all_envs/mean_action/price_player_1"][:, j, :]
        agent2_prices = all_env_stats[f"train/all_envs/mean_action/price_player_2"][:, j, :]

        # Metric 1: convergence vs dispersion.
        quotient_price_diff = np.abs(agent1_prices - agent2_prices) / (
            param_args["collusive_price"] - param_args["competitive_price"]
        )
        convergence_vs_dispersion_metric_per_seed = np.mean(
            quotient_price_diff[:, -last_ten_percent_episodes:], axis=1
        )

        # Add boxplots to each subplot with wider boxes
        box_width = 0.3
        axs[0].boxplot(convergence_vs_dispersion_metric_per_seed, positions=[j], widths=box_width)
        axs[1].boxplot(coll_idx_per_seed_gen, positions=[j], widths=box_width)

    # Set x-ticks and labels
    for ax in axs:
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([format_float(value) for value in param_values])

    axs[0].axhline(
        y=0.2,
        color="blue",
        linestyle="dotted",
        linewidth=1,
        label="convergence threshold",
    )
    axs[1].axhline(
        y=0.05,
        color="blue",
        linestyle="dotted",
        linewidth=1,
        label="collusive threshold",
    )

    # Set labels and titles for each subplot
    metrics = [
        "Convergence vs Dispersion",
        f"Collusion index (generalized mean, p={gen_mean_p})",
    ]
    axs[0].set_ylabel("Convergence Metric")
    axs[1].set_ylabel(f"Collusion Index (Gen. Mean, p={gen_mean_p})")
    for i, ax in enumerate(axs):
        ax.set_title(f"{metrics[i]}")
        ax.set_xlabel(rename_hyperparam(param))
        ax.legend()

    # Add fixed hyperparameters information
    title = f"Comparisons for {rename_hyperparam(param)}. Distribution over {param_args['num_seeds']} seeds."
    plt.suptitle(title)
    plt.savefig(
        os.path.join(paper_dir, f"fig4_{rename_hyperparam(param)}_boxplots.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Saved plot to {os.path.join(paper_dir, f'fig4_{rename_hyperparam(param)}_boxplots.png')}"
    )
    plt.close()

    return fig, axs


def plot_paper_boxplots_fancy(all_env_stats, param, param_values, param_args):
    param_without_spaces = rename_hyperparam(param).replace(" ", "_")
    param_with_spaces = (
        " ".join(
            word.capitalize() if word.lower() != "of" else word
            for word in rename_hyperparam(param).replace("_", " ").split()
        )
        .replace("Dqn", "DQN")
        .replace("Ppo", "PPO")
    )

    # Apply elegant plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#1f77b4", "#ff7f0e"])

    log_interval = max(param_args["num_iters"] // 1000, 5 if param_args["num_iters"] > 1000 else 1)
    last_ten_percent_episodes = param_args["num_iters"] // 10

    # Set up a plot with 2 subplots, much narrower horizontally
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300, facecolor="white")
    plt.subplots_adjust(wspace=0.4)

    # Set boxplot color based on agent type
    boxplot_color = "#8B4B8B" if param_args["agent_default"] == "PPO" else "#1A5F7A"

    for j, value in enumerate(param_values):
        agent_profit_gains_seeds_timesteps = np.zeros(
            (
                param_args["num_seeds"],
                param_args["num_players"],
                all_env_stats["train/collusion_index/mean_player_1"].shape[2],
            )
        )
        for agent_idx in range(param_args["num_players"]):
            agent_profit_gains_seeds_timesteps[:, agent_idx, :] = all_env_stats[
                f"train/collusion_index/mean_player_{agent_idx + 1}"
            ][:, j, :]

        coll_idx_seeds_timesteps_gen = generalized_mean(
            agent_profit_gains_seeds_timesteps, p=gen_mean_p
        )

        coll_idx_per_seed_gen = np.mean(
            coll_idx_seeds_timesteps_gen[:, -last_ten_percent_episodes:], axis=1
        )

        agent1_prices = all_env_stats[f"train/all_envs/mean_action/price_player_1"][:, j, :]
        agent2_prices = all_env_stats[f"train/all_envs/mean_action/price_player_2"][:, j, :]

        quotient_price_diff = np.abs(agent1_prices - agent2_prices) / (
            param_args["collusive_price"] - param_args["competitive_price"]
        )
        convergence_vs_dispersion_metric_per_seed = np.mean(
            quotient_price_diff[:, -last_ten_percent_episodes:], axis=1
        )

        # Add boxplots to each subplot with narrower boxes
        box_width = 0.3
        for i, ax in enumerate(axs):
            data = convergence_vs_dispersion_metric_per_seed if i == 0 else coll_idx_per_seed_gen
            bp = ax.boxplot(data, positions=[j], widths=box_width, patch_artist=True)

            for element in [
                "whiskers",
                "fliers",
                "means",
                "medians",
                "caps",
            ]:  # "boxes",
                plt.setp(bp[element], color=boxplot_color, linewidth=1)
            for box in bp["boxes"]:
                box_coords = box.get_path().vertices
                patch = mpatches.PathPatch(
                    box.get_path(),
                    facecolor=boxplot_color,
                    edgecolor=boxplot_color,
                    alpha=0.3,
                    linewidth=2,
                )
                box.set_fill(False)
                ax.add_patch(patch)
                box.set_edgecolor(boxplot_color)
                box.set_linewidth(1)
            plt.setp(bp["fliers"], markeredgecolor=boxplot_color)

            # # Set the color for all parts of the boxplot
            # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            #     plt.setp(bp[element], color=boxplot_color)

    # Set x-ticks and labels
    for ax in axs:
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([format_float(value) for value in param_values], rotation=45, ha="right")

    # Add threshold lines
    axs[0].axhline(
        y=0.2,
        color="#888888",
        linestyle="--",
        linewidth=2,
        label="Convergence",
    )
    axs[1].axhline(
        y=0.05,
        color="#2E8B57",
        linestyle="--",
        linewidth=2,
        label="Collusion",
    )

    # Set labels and titles
    metrics = [
        "Convergence Metric",
        f"Collusion Index",
    ]
    axs[0].set_ylabel("Convergence Metric", fontsize=15, labelpad=5)
    axs[1].set_ylabel("Collusion Index", fontsize=15, labelpad=5)
    for i, ax in enumerate(axs):
        ax.set_xlabel(param_with_spaces, fontsize=15, labelpad=5)
        legend = ax.legend(
            fontsize=12,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
        )
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(paper_dir, f"fig4_{param_without_spaces}_boxplots.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    return fig, axs


try:
    plot_paper_boxplots_fancy(lr_env_stats, lr_name, lr_values[lr_name], lr_args)
except Exception as e:
    print(f"Failed plotting learning rate: {e}")
try:
    plot_paper_boxplots_fancy(eps_env_stats, eps_name, eps_values[eps_name], eps_args)
except Exception as e:
    print(f"Failed plotting epsilon: {e}")
try:
    plot_paper_boxplots_fancy(
        trainint_env_stats, trainint_name, trainint_values[trainint_name], trainint_args
    )
except Exception as e:
    print(f"Failed plotting training iterations: {e}")
try:
    plot_paper_boxplots_fancy(ent_env_stats, ent_name, ent_values[ent_name], ent_args)
except Exception as e:
    print(f"Failed plotting entropy: {e}")
try:
    plot_paper_boxplots_fancy(
        num_minibatches_all_env_stats,
        "PPO_num_minibatches",
        num_minibatches_values,
        num_minibatches_args,
    )
except Exception as e:
    print(f"Failed plotting num_minibatches: {e}")
try:
    plot_paper_boxplots_fancy(
        num_epochs_all_env_stats,
        "PPO_num_epochs",
        num_epochs_values,
        num_epochs_args,
    )
except Exception as e:
    print(f"Failed plotting num_epochs: {e}")
try:
    plot_paper_boxplots_fancy(
        buffer_size_all_env_stats,
        "DQN_buffer_size",
        buffer_size_values,
        buffer_size_args,
    )
except Exception as e:
    print(f"Failed plotting buffer_size: {e}")
# try:
#     plot_paper_boxplots_fancy(
#         inv_all_env_stats,
#         f"{DQN_or_PPO}_initial_inventories",
#         inventory_values,
#         inventory_args,
#     )
# except Exception as e:
#     print(f"Failed plotting initial inventories: {e}")
# try:
#     plot_paper_boxplots_fancy(
#         th_all_env_stats,
#         f"{DQN_or_PPO}_time_horizon",
#         time_horizon_values,
#         time_horizon_args,
#     )
# except Exception as e:
#     print(f"Failed plotting time horizon: {e}")
# Display plots
display_plots(os.path.join(paper_dir, "fig4_*_boxplots.png"))
# %%


def make_DQN_and_PPO_boxplots():
    """This function creates fancy paper boxplots for DQN and PPO, where they share the same hyperparams and values.
    This can be done for inventory and time horizon.
    For each hyperparameter value, there are two boxplot candles next to each other, one for DQN and one for PPO.
    This way, we can compress two boxplots into one, and fit more on the page.
    It works in exactly the same way and the same design as the plot_paper_boxplots_fancy() function.
    """
    # Set up plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    # Define colors for DQN and PPO
    dqn_color = "#1A5F7A"  # Soft blue
    ppo_color = "#8B4B8B"  # Soft orange

    # Create output directory
    output_dir = "exp/fig4_combined_boxplots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots for inventory and time horizon
    for param in ["inventory", "time_horizon"]:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300, facecolor="white")
        plt.subplots_adjust(wspace=0.3)

        for algo in ["DQN", "PPO"]:
            if param == "inventory":
                base_dir = f"exp/fig4_{algo}_inv_size"
            else:
                base_dir = f"exp/fig4_{algo}_time_horizon_multirun"

            param_data = gather_spread_data(base_dir, param)

            param_values = sorted(param_data.keys())
            first_data = next(iter(param_data.values()))
            param_args = first_data[2]
            first_env_stats = first_data[0][0]

            # Prepare data for plotting
            # [0][0] for ->log_data -> env_stats
            param_all_env_stats = {}
            for key in first_env_stats.keys():
                first_array = first_env_stats[key]
                param_all_env_stats[key] = np.zeros(
                    (first_array.shape[0], len(param_values), first_array.shape[2])
                )

            # if param == "inventory":
            # don't want to plot the first inventory value -- interval too small, unstable learning
            # param_values = param_values[1:]

            # Fill param_all_env_stats
            for i, param_value in enumerate(param_values):
                log_data, _, _ = param_data[param_value]
                for key in param_all_env_stats:
                    param_all_env_stats[key][:, i, :] = log_data[0][key].squeeze(axis=1)

            plot_boxplots(
                axs,
                param_all_env_stats,
                algo,
                param,
                param_values,
                param_args,
                dqn_color if algo == "DQN" else ppo_color,
            )

        # Set labels and titles
        metrics = ["Convergence Metric", "Collusion Index"]
        for i, ax in enumerate(axs):
            ax.set_ylabel(metrics[i], fontsize=15, labelpad=5)
            ax.set_xlabel(
                "Inventory Size" if param == "inventory" else "Time Horizon",
                fontsize=15,
                labelpad=5,
            )

            convergence_line = ax.axhline(
                y=0.2 if i == 0 else 0.05,
                color="#888888" if i == 0 else "#2E8B57",
                linestyle="--",
                linewidth=2,
            )

            handles = [
                mpatches.Patch(color=dqn_color, label="DQN", alpha=0.3),
                mpatches.Patch(color=ppo_color, label="PPO", alpha=0.3),
                convergence_line,
            ]

            # legend = ax.legend(
            hline_label = "Convergence" if i == 0 else "Collusion"
            labels = ["DQN", "PPO", hline_label]
            legend = ax.legend(
                handles=handles,
                labels=labels,
                fontsize=12,
                frameon=True,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
            )
            legend.get_frame().set_edgecolor("#333333")
            legend.get_frame().set_linewidth(0.8)

        # # Add threshold lines
        # axs[0].axhline(
        #     y=0.2,
        #     color="#007BA7",
        #     linestyle="--",
        #     linewidth=1.5,
        #     label="convergence threshold",
        # )
        # axs[1].axhline(
        #     y=0.05,
        #     color="#2E8B57",
        #     linestyle="--",
        #     linewidth=1.5,
        #     label="collusive threshold",
        # )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"fig4_{param}_combined_boxplots.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()


def plot_boxplots(
    axs,
    data: Dict[int, Any],
    algo: str,
    param: str,
    param_values,
    param_args,
    color: str,
):
    """
    Plots boxplots for a given algorithm and parameter.
    """
    last_ten_percent_episodes = param_args["num_iters"] // 10

    for j, value in enumerate(param_values):
        agent_profit_gains = np.zeros(
            (
                param_args["num_seeds"],
                param_args["num_players"],
                data["train/collusion_index/mean_player_1"].shape[2],
            )
        )
        for agent_idx in range(param_args["num_players"]):
            agent_profit_gains[:, agent_idx, :] = data[
                f"train/collusion_index/mean_player_{agent_idx + 1}"
            ][:, j, :]

        coll_idx_gen = generalized_mean(agent_profit_gains, p=0.5)
        coll_idx_per_seed_gen = np.mean(coll_idx_gen[:, -last_ten_percent_episodes:], axis=1)

        agent1_prices = data[f"train/all_envs/mean_action/price_player_1"][:, j, :]
        agent2_prices = data[f"train/all_envs/mean_action/price_player_2"][:, j, :]

        quotient_price_diff = np.abs(agent1_prices - agent2_prices) / (
            param_args["collusive_price"] - param_args["competitive_price"]
        )
        convergence_metric = np.mean(quotient_price_diff[:, -last_ten_percent_episodes:], axis=1)

        box_width = 0.3
        position = j + (0.2 if algo == "PPO" else -0.2)
        for i, ax in enumerate(axs):
            values = convergence_metric if i == 0 else coll_idx_per_seed_gen
            bp = ax.boxplot(
                values,
                positions=[position],
                widths=box_width,
                patch_artist=True,  # patch_artist controls the filling?
            )
            for element in [
                "whiskers",
                "fliers",
                "means",
                "medians",
                "caps",
            ]:  # "boxes",
                plt.setp(bp[element], color=color, linewidth=1)
            for box in bp["boxes"]:
                box_coords = box.get_path().vertices
                patch = mpatches.PathPatch(
                    box.get_path(),
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.3,
                    linewidth=2,
                )
                box.set_fill(False)
                ax.add_patch(patch)
                box.set_edgecolor(color)
                box.set_linewidth(1)
            plt.setp(bp["fliers"], markeredgecolor=color)

    for ax in axs:
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([str(value) for value in param_values], rotation=45, ha="right")


# def generalized_mean(x, p=0.5, ax=1):
#     """Calculates the generalized mean."""
#     res = np.mean(np.sign(x) * np.abs(x) ** p, axis=ax)
#     res = np.maximum(res, 0)
#     return res ** (1 / p)

# Call the function to generate the plots
make_DQN_and_PPO_boxplots()


display_plots(os.path.join("exp/fig4_combined_boxplots", "fig4_*_combined_boxplots.png"))
# %%
