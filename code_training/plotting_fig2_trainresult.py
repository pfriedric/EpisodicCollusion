# %%
import pickle
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import glob

from plotting_utils import (
    overall_mean_stdev_from_seed_means_variances,
    plot_agent_metrics_shaded,
    plot_agent_metrics_indiv_seeds,
    display_single_plot,
)

### Alter this for different runs. Options for the algorithm: "DQN", "PPO", "compPPO", "unconstDQN"
save_dir = "exp/DQN"


plot_new = True
plot_shaded_or_individual = "shaded"
generalized_mean_p = 0.5

run_name = save_dir.split("/")[-1]
print(f"run_name: {run_name}")

print(f"--- LOADING FROM {save_dir} ---")
### LOAD DATA ###
# load all files
with open(f"{save_dir}/args.pkl", "rb") as file:
    args = pickle.load(file)
    try:
        foo = args["deviation_times"]
    except:
        args["deviation_times"] = jnp.arange(args["num_inner_steps"])

try:
    with open(f"{save_dir}/log_data.pkl", "rb") as file:
        log_data = pickle.load(file)
    if isinstance(log_data, tuple):
        if len(log_data) == 2:
            log_data, eval_log_data = log_data
            forced_deviation_log_data = (
                None  # has shape [num_seeds, num_configs, num_deviations, num_timesteps]
            )
        elif len(log_data) == 3:
            log_data, eval_log_data, forced_deviation_log_data = log_data
            forced_deviation_log_data_unstacked = [
                jax.tree.map(lambda v: v[:, :, i, ...], forced_deviation_log_data)
                for i in range(args["deviation_times"].shape[0])
            ]
        else:
            raise ValueError("log_data tuple has an unexpected length")
    exists_log_data = True

    plot_dir = glob.glob(f"{save_dir}/plots_*")
    if not plot_dir:
        print(f"No plots_ directory found in {save_dir}. Proceeding to plot.")
        plot_new = True
    else:
        print(f"Found plot directory: {plot_dir[0]}")

except FileNotFoundError:
    print(f"No log_data.pkl file found in {save_dir}. Proceeding directly to plots.")
    exists_log_data = False

with open(f"{save_dir}/update_dict.pkl", "rb") as file:
    update_dict = pickle.load(file)

with open(f"{save_dir}/seeds.pkl", "rb") as file:
    seeds = pickle.load(file)


all_env_stats, all_a1_metrics, all_a2_metrics = log_data
all_agent_metrics = [all_a1_metrics, all_a2_metrics]

num_iters = args["num_iters"]
num_seeds = args["num_seeds"]
time_horizon = args["time_horizon"]
num_envs = args["num_envs"]
agents = [args[f"agent{i + 1}"] for i in range(args["num_players"])]


log_interval = max(num_iters // 1000, 5 if num_iters > 1000 else 1)
x_axis = np.arange(0, num_iters, log_interval)
last_ten_percent_episodes = len(x_axis) // 10
dqn_training_starts = args["dqn_default"]["initial_exploration_episodes"]

competitive_demand = min(args["initial_inventories"][0], 470)
collusive_demand = 365

competitive_profits_episodetotal = args["competitive_profits_episodetotal"]
# print(f"competitive_profits_episodetotal: {competitive_profits_episodetotal}")
collusive_profits_episodetotal = args["collusive_profits_episodetotal"]
# print(f"collusive_profits_episodetotal: {collusive_profits_episodetotal}")


print(f"--- SETUP ---")
print(f"  run name: {save_dir.split('/')[-1]}")
print(f"  normalized rewards: {args['normalize_rewards_manually']}")
inv_per_T = [inv / args["time_horizon"] for inv in args["initial_inventories"]]
print(f"  inventories (/T): {inv_per_T}")
print(f"  price grid: {args['which_price_grid']}")
if args["agent_default"] == "DQN":
    print(f"DQN setup:")
    print(
        f"  LR {args['dqn_default']['learning_rate']} {'annealing to 0' if args['dqn_default']['lr_scheduling'] else 'fixed'} {'over ' + str(args['dqn_default']['lr_anneal_duration'] * 100) + '% of run' if args['dqn_default']['lr_scheduling'] else '(flat)'}"
    )
    try:
        print(
            f"  Epsilon anneals {'linearly' if args['dqn_default']['epsilon_anneal_type'] == 'linear' else 'exponentially'} from {args['dqn_default']['epsilon_start']} to {args['dqn_default']['epsilon_finish']} over {args['dqn_default']['epsilon_anneal_duration'] * 100:.2f}% of run"
        )
    except:
        print(
            f"  Epsilon anneals linearly from {args['dqn_default']['epsilon_start']} to {args['dqn_default']['epsilon_finish']} over {args['dqn_default']['epsilon_anneal_duration'] * 100:.2f}% of run"
        )
    print(
        f"  buffer size: {args['dqn_default']['buffer_size']}, batch size: {args['dqn_default']['buffer_batch_size']}"
    )
    print(
        f"  discount: {args['dqn_default']['discount']}, max grad norm: {args['dqn_default']['max_gradient_norm']}"
    )
    print(f"  hidden sizes: {args['dqn_default']['hidden_sizes']}")
    print(
        f"  training starts after {args['dqn_default']['initial_exploration_episodes']} eps; then trains every {args['dqn_default']['training_interval_episodes']} eps; target update every {args['dqn_default']['target_update_interval_episodes']} eps"
    )

elif args["agent_default"] == "PPO":
    print(f"PPO agent (setup printing not implemented)")


def generalized_mean(x, p=0.5, ax=1):
    """calculates an altered version of the generalized mean that is meant to better take into account negative values. input x is a multidimensional array."""
    ## altered version for multidimensional input
    res = np.mean(np.sign(x) * np.abs(x) ** p, axis=ax)
    res = np.maximum(res, 0)

    return res ** (1 / p)


def plot_main(env_metrics, x_axis, gen_mean_p):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    env_metrics_sliced = {k: v[:, 0, x_axis, ...] for k, v in env_metrics.items()}
    title = "4 metrics"

    # filter x-axis if DQN agents were used so we don't plot non-training eps at the start
    if all(agent == "DQN" for agent in agents):
        start_index = next((i for i, x in enumerate(x_axis) if x > dqn_training_starts), 0)
        filtered_x_axis = x_axis[start_index:]
        env_metrics_sliced = {k: v[:, start_index:] for k, v in env_metrics_sliced.items()}
    else:
        start_index = 0
        filtered_x_axis = x_axis

    ## plot total profit of all agents
    total_profit_means = np.zeros((num_seeds, len(filtered_x_axis)))
    for agent_idx, agent in enumerate(agents):
        agent_seed_means = env_metrics_sliced[
            f"vmap_metrics/total_profit_mean_player_{agent_idx + 1}"
        ].astype(np.float32)
        # print(f"shape of agent_seed_means: {agent_seed_means.shape}")
        total_profit_means += agent_seed_means  # [seeds, T]

    # print(f"shape of total_profit_means: {total_profit_means.shape}")
    total_profit_mean = np.mean(total_profit_means, axis=0)
    total_profit_std = np.std(total_profit_means, axis=0)

    ## top left: plot total profit
    axs[0, 0].plot(filtered_x_axis, total_profit_mean, linewidth=0.75)
    axs[0, 0].fill_between(
        filtered_x_axis,
        total_profit_mean - total_profit_std,
        total_profit_mean + total_profit_std,
        alpha=0.3,
    )
    axs[0, 0].set_title("Total Profit")
    # horizontal line at competitive and collusive profits
    axs[0, 0].axhline(np.sum(competitive_profits_episodetotal), color="r", linestyle="--")
    axs[0, 0].axhline(np.sum(collusive_profits_episodetotal), color="g", linestyle="--")
    axs[0, 0].set_xlabel("episodes")

    for agent_idx, agent in enumerate(agents):
        agent_seed_means = env_metrics_sliced[
            f"train/all_envs/mean_action/action_player_{agent_idx + 1}"
        ]
        agent_seed_vars = env_metrics_sliced[f"vmap_metrics/action_var_player_{agent_idx + 1}"]
        if plot_shaded_or_individual == "individual":
            axs[0, 1], line, _, _ = plot_agent_metrics_indiv_seeds(
                axs[0, 1],
                agent_seed_means,
                agent_seed_vars,
                filtered_x_axis,
                agent_idx,
                "Average Action per Episode",
                num_envs,
            )
        elif plot_shaded_or_individual == "shaded":
            axs[0, 1], line, _, _ = plot_agent_metrics_shaded(
                axs[0, 1],
                agent_seed_means,
                agent_seed_vars,
                filtered_x_axis,
                agent_idx,
                "Average Action per Episode",
                num_envs,
            )
        axs[0, 1].legend()
        # add horizontal lines for competitive and collusive actions
        competitive_action = args["competitive_action"]
        collusive_action = args["collusive_action"]
        axs[0, 1].axhline(
            competitive_action,
            color="r",
            linestyle="--",
            # label="Competitive Action",
        )
        axs[0, 1].axhline(
            collusive_action,
            color="g",
            linestyle="--",
            # label="Collusive Action",
        )

        # Set y-axis limits and ticks
        num_prices = args["num_prices"]
        axs[0, 1].set_ylim(0, num_prices)
        tick_interval = max(1, num_prices // 7)  # Ensure at least 7-8 ticks
        axs[0, 1].set_yticks(range(0, num_prices + 1, tick_interval))
        # legend in the top left corner
        axs[0, 1].legend(loc="upper left")
        axs[0, 1].set_xlabel("episodes")

    ## bottom left: collusion index
    # overall collidx can be geom mean or average of individual agents' profit gains
    agent_profit_gains_seeds = np.zeros((num_seeds, len(agents), len(filtered_x_axis)))
    for agent_idx, agent in enumerate(agents):
        agent_profit_gains_seeds[:, agent_idx, :] = env_metrics_sliced[
            f"train/collusion_index/mean_player_{agent_idx + 1}"
        ]  # [seeds, N, T]

    coll_idx_seeds_arith = agent_profit_gains_seeds.mean(axis=1)

    def safe_geom_mean(x, axis, offset):
        """safe geometric mean along given axis"""
        x = np.maximum(x, 1e-10)
        return np.exp(np.mean(np.log(x), axis=axis))

    coll_idx_seeds_gen = generalized_mean(agent_profit_gains_seeds, p=gen_mean_p)

    coll_idx_arith_mean = coll_idx_seeds_arith.mean(axis=0)
    coll_idx_arith_var = coll_idx_seeds_arith.var(axis=0)
    coll_idx_gen_mean = coll_idx_seeds_gen.mean(axis=0)
    coll_idx_gen_var = coll_idx_seeds_gen.var(axis=0)

    # test: print geometric mean for last 100 timesteps
    # print(f"coll_idx_geom_mean[-100:]: {coll_idx_geom_mean[-100:]}")
    if plot_shaded_or_individual == "individual":
        for seed in range(num_seeds):
            axs[1, 0].plot(filtered_x_axis, coll_idx_seeds_arith[seed], alpha=0.1, linewidth=0.75)
            axs[1, 1].plot(filtered_x_axis, coll_idx_seeds_gen[seed], alpha=0.1, linewidth=0.75)

        axs[1, 0].plot(filtered_x_axis, coll_idx_arith_mean, linewidth=1.5, color="black")
        axs[1, 1].plot(filtered_x_axis, coll_idx_gen_mean, linewidth=1.5, color="black")
    elif plot_shaded_or_individual == "shaded":
        axs[1, 0].plot(filtered_x_axis, coll_idx_arith_mean, linewidth=0.75)
        axs[1, 0].fill_between(
            filtered_x_axis,
            coll_idx_arith_mean - np.sqrt(coll_idx_arith_var),
            coll_idx_arith_mean + np.sqrt(coll_idx_arith_var),
            alpha=0.3,
        )

        axs[1, 1].plot(filtered_x_axis, coll_idx_gen_mean, linewidth=0.75)
        axs[1, 1].fill_between(
            filtered_x_axis,
            coll_idx_gen_mean - np.sqrt(coll_idx_gen_var),
            coll_idx_gen_mean + np.sqrt(coll_idx_gen_var),
            alpha=0.3,
        )

    axs[1, 0].set_title("Collusion Index (Arithmetic Mean)")
    axs[1, 0].axhline(0, color="r", linestyle="--")
    axs[1, 0].axhline(1, color="g", linestyle="--")
    axs[1, 0].set_xlabel("episodes")

    axs[1, 1].set_title(f"Collusion Index (Generalized Mean, p={generalized_mean_p})")
    axs[1, 1].axhline(0, color="r", linestyle="--")
    axs[1, 1].axhline(1, color="g", linestyle="--")
    axs[1, 1].set_xlabel("episodes")

    plot_dir = os.path.join(save_dir, "paper_plots")
    os.makedirs(plot_dir, exist_ok=True)
    # plt.show()

    print(
        f"gen_mean last 10%: {coll_idx_gen_mean[-int(len(coll_idx_gen_mean) * 0.1) :].mean():.2f}"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"all_plots.png"))
    print(f"saved fig2_{plot_shaded_or_individual}.png")
    plt.close()
    return fig, axs, title


def display_plots(plot_pattern):
    for plot_path in sorted(glob.glob(plot_pattern)):
        display_single_plot(plot_path)


def load_and_display_plots():
    plot_dir = os.path.join(save_dir, "paper_plots")
    plot_pattern = os.path.join(plot_dir, "all_plots.png")
    display_plots(plot_pattern)


plot_main(all_env_stats, x_axis, generalized_mean_p)
load_and_display_plots()


# %%
def plot_vert(env_metrics, x_axis, gen_mean_p):
    # pretty plots
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.rcParams["text.usetex"] = True

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    env_metrics_sliced = {k: v[:, 0, x_axis, ...] for k, v in env_metrics.items()}
    # title = f"{args["agent1"]} Collusion"

    # filter x-axis if DQN agents were used so we don't plot non-training eps at the start
    if all(agent == "DQN" for agent in agents):
        start_index = next((i for i, x in enumerate(x_axis) if x > dqn_training_starts), 0)
        filtered_x_axis = x_axis[start_index:]
        env_metrics_sliced = {k: v[:, start_index:] for k, v in env_metrics_sliced.items()}
    else:
        start_index = 0
        filtered_x_axis = x_axis

    ## plot total profit of all agents
    total_profit_means = np.zeros((num_seeds, len(filtered_x_axis)))
    for agent_idx, agent in enumerate(agents):
        agent_seed_means = env_metrics_sliced[
            f"vmap_metrics/total_profit_mean_player_{agent_idx + 1}"
        ].astype(np.float32)
        # print(f"shape of agent_seed_means: {agent_seed_means.shape}")
        total_profit_means += agent_seed_means  # [seeds, T]

    # print(f"shape of total_profit_means: {total_profit_means.shape}")
    total_profit_mean = np.mean(total_profit_means, axis=0)
    total_profit_std = np.std(total_profit_means, axis=0)

    agent_colors = plt.cm.tab10.colors  #  "#1f77b4", "#ff7f0e"]
    for agent_idx, agent in enumerate(agents):
        agent_seed_means = env_metrics_sliced[
            f"train/all_envs/mean_action/action_player_{agent_idx + 1}"
        ]
        agent_seed_vars = env_metrics_sliced[f"vmap_metrics/action_var_player_{agent_idx + 1}"]
        # axs[0], line, _, _ = plot_agent_metrics_shaded_selectable_color(
        #     axs[0],
        #     agent_seed_means,
        #     agent_seed_vars,
        #     filtered_x_axis,
        #     agent_idx,
        #     "Average Action per Episode",
        #     num_envs,
        #     agent_colors[agent_idx],
        # )

        metric_mean, metric_std = overall_mean_stdev_from_seed_means_variances(
            agent_seed_means, agent_seed_vars, num_envs
        )

        # Plot the shaded area first
        axs[0].fill_between(
            filtered_x_axis,
            metric_mean - metric_std,
            metric_mean + metric_std,
            alpha=0.3,
            color=agent_colors[agent_idx],
            linewidth=0,  # this is key!! otherwise, edge line with custom colors
        )

        # Plot the line on top of the shaded area
        axs[0].plot(
            filtered_x_axis,
            metric_mean,
            label=f"Agent {agent_idx + 1}",
            linewidth=2,
            color=agent_colors[agent_idx],
        )
        # axs.set_title(f"{metric_name}", fontsize=16)

        if agent == "DQN":
            greedy_action_mean = env_metrics_sliced[
                f"vmap_metrics/greedy_action_mean_player_{agent_idx + 1}"
            ]
            greedy_action_var = env_metrics_sliced[
                f"vmap_metrics/greedy_action_var_player_{agent_idx + 1}"
            ]
            axs[0].plot(
                filtered_x_axis,
                greedy_action_mean.mean(axis=0),
                linestyle=":",
                color=agent_colors[agent_idx],
                linewidth=2,
                alpha=0.6,
            )

    # add horizontal lines for competitive and collusive actions
    competitive_action = args["competitive_action"]
    collusive_action = args["collusive_action"]
    axs[0].axhline(
        competitive_action,
        color="#d62728",
        linestyle="--",
        # label="Competitive Action",
        linewidth=2,
    )
    axs[0].axhline(
        collusive_action,
        color="#2ca02c",
        linestyle="--",
        # label="Collusive Action",
        linewidth=2,
    )

    # Set y-axis limits and ticks
    num_prices = args["num_prices"]
    axs[0].set_ylim(0, num_prices)
    tick_interval = max(1, num_prices // 7)  # Ensure at least 7-8 ticks
    axs[0].set_yticks(range(0, num_prices + 1, tick_interval))
    # legend in the top left corner
    # axs[0].legend(loc="upper left")
    # axs[0].set_xlabel("Episodes")
    axs[0].set_xlabel("Episodes", fontsize=15, labelpad=10)
    axs[0].set_ylabel("Action", fontsize=15, labelpad=10)
    axs[0].set_title("Average Action per Episode", fontsize=16, pad=10)
    legend = axs[0].legend(fontsize=12, frameon=True, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)

    ## bottom collusion index
    agent_profit_gains_seeds = np.zeros((num_seeds, len(agents), len(filtered_x_axis)))
    for agent_idx, agent in enumerate(agents):
        agent_profit_gains_seeds[:, agent_idx, :] = env_metrics_sliced[
            f"train/collusion_index/mean_player_{agent_idx + 1}"
        ]  # [seeds, N, T]

    # coll_idx_seeds_arith = agent_profit_gains_seeds.mean(axis=1)

    def safe_geom_mean(x, axis, offset):
        """safe geometric mean along given axis"""
        x = np.maximum(x, 1e-10)
        return np.exp(np.mean(np.log(x), axis=axis))

    coll_idx_seeds_gen = generalized_mean(agent_profit_gains_seeds, p=gen_mean_p)

    # coll_idx_arith_mean = coll_idx_seeds_arith.mean(axis=0)
    # coll_idx_arith_var = coll_idx_seeds_arith.var(axis=0)
    coll_idx_gen_mean = coll_idx_seeds_gen.mean(axis=0)
    coll_idx_gen_var = coll_idx_seeds_gen.var(axis=0)

    # test: print geometric mean for last 100 timesteps
    # collusion_color = "#9B7ED9"  # Dark Orchid
    # collusion_color = "#5DADE2"  # Soft Teal
    # collusion_color = "#D7837F"  # Dusty Rose
    # collusion_color = "#7DCEA0"  # Sage Green
    # collusion_color = "#F1948A"  # Muted Coral

    if "DQN" in run_name:
        collusion_color = "#1A5F7A"
    elif "PPO" in run_name:
        collusion_color = "#8B4B8B"
    # collusion_color = (
    #     "#007BA7"  # Cerulean # "#4682B4" Steel Blue # "#9999CC"  # Pastel Indigo
    # )
    axs[1].plot(filtered_x_axis, coll_idx_gen_mean, linewidth=2, color=collusion_color)
    axs[1].fill_between(
        filtered_x_axis,
        coll_idx_gen_mean - np.sqrt(coll_idx_gen_var),
        coll_idx_gen_mean + np.sqrt(coll_idx_gen_var),
        alpha=0.3,
        color=collusion_color,
        linewidth=0,
    )

    # axs[1].set_title(f"Collusion Index (Generalized Mean, p={generalized_mean_p})")
    # axs[1].axhline(0, color="r", linestyle="--")
    # axs[1].axhline(1, color="g", linestyle="--")
    # axs[1].set_xlabel("episodes")
    axs[1].set_title(
        f"Collusion Index (Generalized Mean, $\gamma={generalized_mean_p}$)",
        fontsize=16,
        pad=10,
    )
    axs[1].axhline(0, color="#d62728", linestyle="--", linewidth=2)
    axs[1].axhline(1, color="#2ca02c", linestyle="--", linewidth=2)
    axs[1].set_xlabel("Episodes", fontsize=15, labelpad=10)
    axs[1].set_ylabel("Collusion Index", fontsize=15, labelpad=10)

    # for ax in axs:
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)

    last_10_percent_avg = coll_idx_gen_mean[-int(len(coll_idx_gen_mean) * 0.1) :].mean()
    if run_name == "DQN":
        text_x = 0.946
        text_y = 0.02
    elif run_name == "PPO":
        text_x = 0.954
        text_y = 0.015
    elif run_name == "compPPO":
        text_x = 0.95
        text_y = 0.06
    elif run_name == "unconstDQN":
        text_x = 0.953
        text_y = 0.06
    axs[1].text(
        text_x,  # DQN: 0.946. PPO: 0.954 compPPO
        text_y,  # DQN: 0.02. PPO: 0.015 compPPO
        f"Last 10\% avg: {last_10_percent_avg:.2f}",
        transform=axs[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    plot_dir = os.path.join(save_dir, "paper_plots")
    os.makedirs(plot_dir, exist_ok=True)
    # plt.show()

    print(
        f"gen_mean last 10%: {coll_idx_gen_mean[-int(len(coll_idx_gen_mean) * 0.1) :].mean():.2f}"
    )

    fig.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"fig2_{run_name}_training.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(f"saved fig2_{run_name}_training.png")
    plt.close()
    return fig, axs


plot_vert(all_env_stats, x_axis, generalized_mean_p)
display_single_plot(os.path.join(save_dir, "paper_plots", f"fig2_{run_name}_training.png"))


# %%
def plot_PPO_and_DQN_training_runs():
    # pretty plots
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.rcParams["text.usetex"] = True

    fig, axs = plt.subplots(3, 1, figsize=(6, 10))

    algorithms = ["DQN", "PPO"]
    colors = {"DQN": "#1A5F7A", "PPO": "#8B4B8B"}  # Using the colors we defined earlier

    all_data = {}
    max_episodes = 0

    # first, load data for DQN and PPO
    for algo in algorithms:
        save_dir = f"exp/{algo}"
        with open(f"{save_dir}/args.pkl", "rb") as file:
            args = pickle.load(file)
        with open(f"{save_dir}/log_data.pkl", "rb") as file:
            log_data = pickle.load(file)
        log_data, _, _ = log_data
        all_env_stats, _, _ = log_data  # detangling from agent metrics

        log_interval = max(args["num_iters"] // 1000, 5 if args["num_iters"] > 1000 else 1)
        x_axis = np.arange(0, args["num_iters"], log_interval)
        max_episodes = max(max_episodes, len(x_axis))

        all_data[algo] = {"args": args, "env_stats": all_env_stats, "x_axis": x_axis}

    # plot DQN and PPO training runs
    for i, algo in enumerate(algorithms):
        args = all_data[algo]["args"]
        env_stats = all_data[algo]["env_stats"]
        x_axis = all_data[algo]["x_axis"]

        env_metrics_sliced = {k: v[:, 0, x_axis, ...] for k, v in env_stats.items()}

        agent_colors = ["#1f77b4", "#ff7f0e"]
        for agent_idx in range(args["num_players"]):
            agent_seed_means = env_metrics_sliced[
                f"train/all_envs/mean_action/action_player_{agent_idx + 1}"
            ]
            agent_seed_vars = env_metrics_sliced[f"vmap_metrics/action_var_player_{agent_idx + 1}"]

            metric_mean, metric_std = overall_mean_stdev_from_seed_means_variances(
                agent_seed_means, agent_seed_vars, args["num_envs"]
            )

            axs[i].fill_between(
                x_axis,
                metric_mean - metric_std,
                metric_mean + metric_std,
                alpha=0.3,
                color=agent_colors[agent_idx],
                linewidth=0,
            )

            axs[i].plot(
                x_axis,
                metric_mean,
                label=f"{algo} {agent_idx + 1}",
                linewidth=2,
                color=agent_colors[agent_idx],
            )
        axs[i].axhline(args["competitive_action"], color="#d62728", linestyle="--", linewidth=2)
        axs[i].axhline(args["collusive_action"], color="#2ca02c", linestyle="--", linewidth=2)

        # axs[i].set_ylim(0, args["num_prices"])
        tick_interval = max(1, args["num_prices"] // 7)
        axs[i].set_yticks(range(0, args["num_prices"] + 1, tick_interval))
        axs[i].set_xlabel("Episodes", fontsize=15, labelpad=5)
        axs[i].set_ylabel("Action", fontsize=15, labelpad=5)
        # axs[i].set_title(f"{algo} Average Action per Episode", fontsize=16, pad=5)
        legend = axs[i].legend(
            fontsize=14, frameon=True, loc="upper left", bbox_to_anchor=(0.02, 0.98)
        )
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(0.8)

    # Collusion Index subplot
    for algo in algorithms:
        args = all_data[algo]["args"]
        env_stats = all_data[algo]["env_stats"]
        x_axis = all_data[algo]["x_axis"]

        env_metrics_sliced = {k: v[:, 0, x_axis, ...] for k, v in env_stats.items()}

        agent_profit_gains_seeds = np.zeros((args["num_seeds"], args["num_players"], len(x_axis)))
        for agent_idx in range(args["num_players"]):
            agent_profit_gains_seeds[:, agent_idx, :] = env_metrics_sliced[
                f"train/collusion_index/mean_player_{agent_idx + 1}"
            ]

        coll_idx_seeds_gen = generalized_mean(agent_profit_gains_seeds, p=0.5)
        coll_idx_gen_mean = coll_idx_seeds_gen.mean(axis=0)
        coll_idx_gen_var = coll_idx_seeds_gen.var(axis=0)

        normalized_x = np.linspace(0, 100, len(x_axis))

        axs[2].plot(normalized_x, coll_idx_gen_mean, linewidth=2, color=colors[algo], label=algo)
        axs[2].fill_between(
            normalized_x,
            coll_idx_gen_mean - np.sqrt(coll_idx_gen_var),
            coll_idx_gen_mean + np.sqrt(coll_idx_gen_var),
            alpha=0.3,
            color=colors[algo],
            linewidth=0,
        )

    # axs[2].set_title("Collusion Index (Generalized Mean, $\gamma=0.5$)", fontsize=16, pad=5)
    axs[2].axhline(0, color="#d62728", linestyle="--", linewidth=2)
    axs[2].axhline(1, color="#2ca02c", linestyle="--", linewidth=2)
    axs[2].set_xlabel(f"Training Progress (\%)", fontsize=15, labelpad=5)
    axs[2].set_ylabel("Collusion Index", fontsize=15, labelpad=5)
    axs[2].legend(fontsize=14, frameon=True, loc="upper left", bbox_to_anchor=(0.02, 0.96))

    fig.tight_layout()
    plot_dir = os.path.join("exp", "fig2_combined_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(
        os.path.join(plot_dir, "fig2_DQN_PPO_training.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("Saved fig2_DQN_PPO_training.png")
    plt.close()
    return fig, axs


# Call the function
plot_dir = os.path.join("exp", "fig2_combined_plots")
os.makedirs(plot_dir, exist_ok=True)
fig, axs = plot_PPO_and_DQN_training_runs()
display_single_plot(os.path.join(plot_dir, "fig2_DQN_PPO_training.png"))

# %%
