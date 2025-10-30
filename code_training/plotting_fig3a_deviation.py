# %%
import pickle
import os
import jax
import numpy as np
import matplotlib.pyplot as plt
import glob
from plotting_utils import display_single_plot

### Alter this for different runs. Options for the algorithm: "DQN", "PPO", "compPPO", "unconstDQN"
save_dir = "exp/DQN"


########################################################
plot_new = False
plot_shaded_or_individual = "shaded"
generalized_mean_p = 0.5
shaded_alpha = 0.15

run_name = save_dir.split("/")[-1]

print(f"--- LOADING FROM {save_dir} ---")
### LOAD DATA ###
# load all files
with open(f"{save_dir}/args.pkl", "rb") as file:
    args = pickle.load(file)
    deviation_times = np.arange(args["num_inner_steps"])

try:
    with open(f"{save_dir}/log_data.pkl", "rb") as file:
        log_data = pickle.load(file)
    if isinstance(log_data, tuple):
        if len(log_data) == 2:
            log_data, eval_log_data = log_data
            forced_deviation_log_data = (
                None  # has shape [num_seeds, num_configs, num_deviations, num_timesteps]
            )
            forced_deviation_log_data_unstacked = None
            print(f"no forced deviation log data!")
        elif len(log_data) == 3:
            log_data, eval_log_data, forced_deviation_log_data = log_data
            forced_deviation_log_data_unstacked = [
                jax.tree.map(lambda v: v[:, :, i, ...], forced_deviation_log_data)
                for i in range(args["num_inner_steps"])
            ]
        else:
            raise ValueError("log_data tuple has an unexpected length")
    exists_log_data = True

    plot_dir = glob.glob(f"{save_dir}/forced_deviation_")
    if not plot_dir:
        print(f"No forced deviation plots directory found in {save_dir}. Proceeding to plot.")
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
num_actions = args["num_prices"]


log_interval = max(num_iters // 1000, 5 if num_iters > 1000 else 1)
x_axis = np.arange(0, num_iters, log_interval)
last_ten_percent_episodes = len(x_axis) // 10
dqn_training_starts = args["dqn_default"]["initial_exploration_episodes"]

competitive_demand = min(args["initial_inventories"][0], 470)
collusive_demand = 365

competitive_profits_episodetotal = args["competitive_profits_episodetotal"]
print(f"competitive_profits_episodetotal: {competitive_profits_episodetotal}")
collusive_profits_episodetotal = args["collusive_profits_episodetotal"]
print(f"collusive_profits_episodetotal: {collusive_profits_episodetotal}")


print(f"--- SETUP ---")
print(f"  run name: {save_dir.split('/')[-1]}")
print(f"  normalized rewards: {args['normalize_rewards_manually']}")
inv_per_T = [inv / args["time_horizon"] for inv in args["initial_inventories"]]
print(f"  inventories (/T): {inv_per_T}")
print(f"  price grid: {args['which_price_grid']}")
print(f"  env used: {args['env_id']}")
print(f"  number of iters: {num_iters}, seeds: {num_seeds}")
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


def plot_per_timestep_metrics(per_timestep_metrics, forced_deviation=False):
    """expects a single eval_log_data shaped input"""
    num_agents = len(agents)

    if not isinstance(forced_deviation, (int, np.integer)):
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # Increased width from 18 to 24
    else:
        fig, axs = plt.subplots(1, 4, figsize=(24, 5))  # Increased width from 24 to 32

    # Verify the length of data is time_horizon
    time_horizon = per_timestep_metrics[f"actions_1"].shape[2]
    x_range = range(time_horizon)
    discount_factor = args["normalizing_rewards_gamma"]
    is_forced_deviation_plot = isinstance(forced_deviation, (int, np.integer))

    if is_forced_deviation_plot:
        title = f"Per-Timestep Metrics from forced episode at t={forced_deviation}"
    else:
        title = f"Per-Timestep Metrics from eval episode"

    if is_forced_deviation_plot:
        metrics_to_plot = ["actions", "rewards"]  # "rewards_rescaled", "inventories"
    else:
        metrics_to_plot = ["actions", "rewards", "inventories"]

    # Plot Actions, Rewards, Inventories (, Rescaled Rewards)
    for i, metric in enumerate(metrics_to_plot):
        agent_colors = plt.cm.tab10.colors
        for agent_idx, agent in enumerate(agents):
            data = per_timestep_metrics[f"{metric}_{agent_idx + 1}"][:, 0, :]
            # print(f"data name: {metric}_{agent_idx+1}, shape: {data.shape}")
            axs[i].plot(
                x_range,
                data.mean(axis=0),
                label=f"Agent {agent_idx + 1}",
                color=agent_colors[agent_idx],
            )

            if is_forced_deviation_plot:
                eval_data = eval_log_data[f"{metric}_{agent_idx + 1}"][:, 0, :]
                axs[i].plot(
                    x_range,
                    eval_data.mean(axis=0),
                    # label=f"Agent {agent_idx+1} (Eval)",
                    color=agent_colors[agent_idx],
                    linestyle=":",
                )
            if plot_shaded_or_individual == "shaded":
                axs[i].fill_between(
                    x_range,
                    data.mean(axis=0) - data.std(axis=0),
                    data.mean(axis=0) + data.std(axis=0),
                    alpha=shaded_alpha,
                )
            elif plot_shaded_or_individual == "individual":
                for seed in range(data.shape[0]):
                    axs[i].plot(
                        x_range,
                        data[seed, :],
                        alpha=0.1,
                        linewidth=0.75,
                        color=agent_colors[agent_idx],
                    )
        axs[i].set_title(f"{metric.capitalize()} per Timestep")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend(loc="upper left")
        if metric == "rewards":
            competitive_profits = args["competitive_profits"]
            collusive_profits = args["collusive_profits"]
            for agent_idx in range(len(agents)):
                axs[i].axhline(
                    competitive_profits[agent_idx],
                    color="r",
                    linestyle="--",
                )
                axs[i].axhline(
                    collusive_profits[agent_idx],
                    color="g",
                    linestyle="--",
                )
            axs[i].legend()
        elif metric == "rewards_rescaled":
            for agent_idx in range(len(agents)):
                axs[i].axhline(
                    rescaled_competitive_reward,
                    color="r",
                    linestyle="--",
                )
                axs[i].axhline(
                    rescaled_collusive_reward,
                    color="g",
                    linestyle="--",
                )
            axs[i].legend()

        elif metric == "actions":
            competitive_action = args["competitive_action"]
            collusive_action = args["collusive_action"]
            axs[i].axhline(
                competitive_action,
                color="r",
                linestyle="--",
            )
            axs[i].axhline(
                collusive_action,
                color="g",
                linestyle="--",
            )
            axs[i].legend()

    if is_forced_deviation_plot:
        # plot a comparison of profit vs non-deviation rewards
        for agent_idx in range(len(agents)):
            numerator = per_timestep_metrics[f"rewards_{agent_idx + 1}"][
                :, 0, :
            ]  # shape (num_seeds, time_horizon)
            denominator = eval_log_data[f"rewards_{agent_idx + 1}"][
                :, 0, :
            ]  # shape (num_seeds, time_horizon)
            epsilon = 1e-10
            denominator_safe = denominator + epsilon
            reward_ratio = numerator / denominator_safe

            reward_ratio = np.where(
                np.isclose(numerator, 0) & np.isclose(denominator, 0), 1, reward_ratio
            )
            reward_ratio = np.clip(reward_ratio, 0, 2)
            axs[2].plot(
                x_range,
                reward_ratio.mean(axis=0),
                label=f"Agent {agent_idx + 1}",
                color=agent_colors[agent_idx],
            )
            if plot_shaded_or_individual == "shaded":
                axs[2].fill_between(
                    x_range,
                    reward_ratio.mean(axis=0) - reward_ratio.std(axis=0),
                    reward_ratio.mean(axis=0) + reward_ratio.std(axis=0),
                    alpha=shaded_alpha,
                )
            elif plot_shaded_or_individual == "individual":
                for seed in range(reward_ratio.shape[0]):
                    axs[2].plot(
                        x_range,
                        reward_ratio[seed, :],
                        alpha=0.1,
                        linewidth=0.75,
                        color=agent_colors[agent_idx],
                    )
            # Start of Selection
        axs[2].axhline(1, color="b", linestyle="--", label="Non-deviation episode")
        axs[2].set_title("Ratio of per-timestep profit, deviation/non-deviation ep.")
        axs[2].set_ylabel("Per-Timestep Profit Ratio")
        axs[2].legend()

        # plot the ratio of cumulative profit so far to the non-deviation episode
        total_cumulative_profit = np.zeros(per_timestep_metrics[f"rewards_1"][:, 0, :].shape)
        total_non_deviation_cumulative_profit = np.zeros(eval_log_data[f"rewards_1"][:, 0, :].shape)

        for agent_idx in range(len(agents)):
            cumulative_profit = np.cumsum(
                per_timestep_metrics[f"rewards_{agent_idx + 1}"][:, 0, :], axis=1
            )
            non_deviation_cumulative_profit = np.cumsum(
                eval_log_data[f"rewards_{agent_idx + 1}"][:, 0, :], axis=1
            )
            profit_ratio = cumulative_profit / non_deviation_cumulative_profit
            axs[3].plot(
                x_range,
                profit_ratio.mean(axis=0),
                label=f"Agent {agent_idx + 1}",
                color=agent_colors[agent_idx],
            )
            if plot_shaded_or_individual == "shaded":
                axs[3].fill_between(
                    x_range,
                    profit_ratio.mean(axis=0) - profit_ratio.std(axis=0),
                    profit_ratio.mean(axis=0) + profit_ratio.std(axis=0),
                    alpha=shaded_alpha,
                )
            elif plot_shaded_or_individual == "individual":
                for seed in range(profit_ratio.shape[0]):
                    axs[3].plot(
                        x_range,
                        profit_ratio[seed, :],
                        alpha=0.1,
                        linewidth=0.75,
                        color=agent_colors[agent_idx],
                    )
            total_cumulative_profit += cumulative_profit
            total_non_deviation_cumulative_profit += non_deviation_cumulative_profit

        # Plot total cumulative profit ratio
        total_profit_ratio = total_cumulative_profit / total_non_deviation_cumulative_profit
        axs[3].plot(
            x_range,
            total_profit_ratio.mean(axis=0),
            label="Total Agents",
            color="k",
            linestyle="--",
        )
        if plot_shaded_or_individual == "shaded":
            axs[3].fill_between(
                x_range,
                total_profit_ratio.mean(axis=0) - total_profit_ratio.std(axis=0),
                total_profit_ratio.mean(axis=0) + total_profit_ratio.std(axis=0),
                alpha=shaded_alpha,
                color="k",
            )
        elif plot_shaded_or_individual == "individual":
            for seed in range(total_profit_ratio.shape[0]):
                axs[3].plot(
                    x_range,
                    total_profit_ratio[seed, :],
                    alpha=0.1,
                    linewidth=0.75,
                    color="k",
                )

        axs[3].axhline(1, color="b", linestyle="--", label="Non-deviation episode")
        axs[3].set_title("Ratio of cumulative profit, deviation/non-deviation ep.")
        axs[3].set_ylabel("Cumulative Profit Ratio")
        axs[3].legend()
        title += f"\nEnd-of-episode total profit vs non-deviation: {total_profit_ratio.mean(axis=0)[-1] * 100:.2f}%"

    fig.suptitle(title)  # Increased fontsize and y position
    # plt.subplots_adjust(top=0.85)  # Adjust top margin and increase space between subplots
    plt.subplots_adjust(
        left=0.05, right=0.98, top=0.85, bottom=0.15, wspace=0.2
    )  # Adjusted margins and spacing

    if is_forced_deviation_plot:
        # put a vertical dotted line at the deviation point in each subplot
        for ax in axs:
            ax.axvline(forced_deviation, color="k", linestyle="--")

    for ax in axs:
        ax.set_xlabel("Timestep")
        # ax.set_xticks(
        #     np.linspace(0, time_horizon - 1, num=min(7, time_horizon), dtype=int)
        # )
        # ax.set_xticklabels(
        #     np.linspace(0, time_horizon - 1, num=min(7, time_horizon), dtype=int)
        # )
        # set xticks per integer
        ax.set_xticks(np.arange(0, time_horizon, step=1))
        ax.tick_params(axis="x", rotation=45)  # Rotated x-tick labels for better fit

    # plt.tight_layout()
    plt.close()
    return fig, title


def plot_main(eval_metrics, forced_deviation_metrics_unstacked, x_axis):
    plot_dir = os.path.join(save_dir, "forced_deviation_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # plot the regular eval metrics first
    fig, title = plot_per_timestep_metrics(eval_metrics, forced_deviation="nope")

    # save eval metrics plot
    fig.savefig(os.path.join(plot_dir, f"fig3_{plot_shaded_or_individual}_nodeviation.png"))
    print(f"saved fig3_{plot_shaded_or_individual}_nodeviation.png")
    # now plot the forced deviation metrics
    for i in deviation_times:
        fig, title = plot_per_timestep_metrics(
            forced_deviation_log_data_unstacked[i], forced_deviation=i
        )

        # save deviation plot
        fig.savefig(
            os.path.join(
                plot_dir,
                f"fig3_{plot_shaded_or_individual}_deviation_{i}.png",
            )
        )
        print(f"saved fig3_{plot_shaded_or_individual}_deviation_{i}.png")


def display_plots(plot_pattern):
    # if plot_pattern is for deviation plots, sort by deviation time
    if "_deviation_" in plot_pattern and "_nodeviation" not in plot_pattern:
        for plot_path in sorted(
            glob.glob(plot_pattern),
            key=lambda x: int(x.split("_deviation_")[-1].replace(".png", "")),
        ):
            display_single_plot(plot_path)
    else:
        for plot_path in sorted(glob.glob(plot_pattern)):
            display_single_plot(plot_path)


def load_and_display_plots():
    plot_dir = os.path.join(save_dir, "forced_deviation_plots")
    # first display no deviation plot
    plot_pattern = os.path.join(plot_dir, "fig3_*_nodeviation.png")
    display_plots(plot_pattern)
    # then display deviation plots
    plot_pattern = os.path.join(plot_dir, "fig3_*_deviation_*.png")
    display_plots(plot_pattern)

    # plot_pattern = os.path.join(plot_dir, "fig3_best_response_surface.png")
    # display_plots(plot_pattern)

    display_single_plot(os.path.join(save_dir, "paper_plots", f"fig3a_{run_name}_forced_dev.png"))


if plot_new and exists_log_data:
    try:
        plot_main(eval_log_data, forced_deviation_log_data_unstacked, x_axis)
        # load_and_display_plots()
    except Exception as e:
        print(f"Error plotting main: {e}")


# %%
def plot_forced_dev():
    """expects a single eval_log_data shaped input"""
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
    plt.rcParams["text.usetex"] = True

    num_agents = len(agents)

    fig, axs = plt.subplots(
        2, 1, figsize=(6, 8), dpi=300, facecolor="white"
    )  # gridspec_kw={"hspace": 0.25})

    # Verify the length of data is time_horizon
    time_horizon = forced_deviation_log_data_unstacked[1][f"actions_1"].shape[2]
    x_range = range(time_horizon)
    discount_factor = args["normalizing_rewards_gamma"]
    title = f"Evolution of actions after forced deviation of agent 1"

    # agent_colors = plt.cm.tab10.colors
    agent_colors = ["#1f77b4", "#ff7f0e"]
    for agent_idx, agent in enumerate(agents):
        per_timestep_metrics = forced_deviation_log_data_unstacked[1]
        data = per_timestep_metrics[f"actions_{agent_idx + 1}"][:, 0, :]
        axs[0].plot(
            x_range,
            data.mean(axis=0),
            label=f"Agent {agent_idx + 1}",
            color=agent_colors[agent_idx],
            linewidth=2,
        )
        eval_data = eval_log_data[f"actions_{agent_idx + 1}"][:, 0, :]
        axs[0].plot(
            x_range,
            eval_data.mean(axis=0),
            # label=f"Agent {agent_idx+1} (Eval)",
            color=agent_colors[agent_idx],
            linestyle=":",
            linewidth=2,
        )
        axs[0].fill_between(
            x_range,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=shaded_alpha,
            color=agent_colors[agent_idx],
            linewidth=0,
        )

        axs[0].set_title(f"Deviation at $t = 1$", fontsize=16, pad=10)
        axs[0].set_ylabel("Actions", fontsize=15, labelpad=5)

        ## plot 2
        per_timestep_metrics = forced_deviation_log_data_unstacked[9]
        data = per_timestep_metrics[f"actions_{agent_idx + 1}"][:, 0, :]
        axs[1].plot(
            x_range,
            data.mean(axis=0),
            label=f"Agent {agent_idx + 1}",
            color=agent_colors[agent_idx],
            linewidth=2,
        )
        eval_data = eval_log_data[f"actions_{agent_idx + 1}"][:, 0, :]
        axs[1].plot(
            x_range,
            eval_data.mean(axis=0),
            # label=f"Agent {agent_idx+1} (Eval)",
            color=agent_colors[agent_idx],
            linestyle=":",
            linewidth=2,
        )
        axs[1].fill_between(
            x_range,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=shaded_alpha,
            color=agent_colors[agent_idx],
            linewidth=0,
        )

        axs[1].set_title(f"Deviation at $t = 9$", fontsize=16, pad=10)
        axs[1].set_ylabel("Actions", fontsize=15, labelpad=5)

    axs[0].axvline(1, color="#888888", linestyle="--", linewidth=2)
    # Add horizontal lines for competitive and collusive actions using fancy colors
    axs[0].axhline(2, color="#d62728", linestyle="--", linewidth=2, label="Competitive action")
    axs[0].axhline(12, color="#2ca02c", linestyle="--", linewidth=2, label="Collusive action")
    axs[0].legend(fontsize=12, frameon=True, loc="upper right", bbox_to_anchor=(1, 0.95))

    # Add horizontal lines for competitive and collusive actions using fancy colors for axs[1]
    axs[1].axhline(2, color="#d62728", linestyle="--", linewidth=2, label="Competitive action")
    axs[1].axhline(12, color="#2ca02c", linestyle="--", linewidth=2, label="Collusive action")
    axs[1].axvline(9, color="#888888", linestyle="--", linewidth=2)
    # Update legend for axs[1] to include new lines
    # axs[1].legend(loc="upper left", fontsize=8)

    # fig.suptitle(
    #     title, y=0.98, fontsize=12
    # )  # Adjusted y position and increased font size for the title

    for ax in axs:
        ax.set_xlabel("Timestep", fontsize=15, labelpad=5)
        ax.set_xticks(np.arange(0, time_horizon, step=1))
        ax.tick_params(axis="x", rotation=45)  # Rotated x-tick labels for better fit

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    # plt.subplots_adjust(top=0.93)  # Adjust top margin to accommodate the main title
    # plt.subplots_adjust(bottom=0.06)  # Reduce the bottom margin
    os.makedirs(os.path.join(save_dir, "paper_plots"), exist_ok=True)
    plt.savefig(os.path.join(save_dir, "paper_plots", f"fig3a_{run_name}_forced_dev.png"))
    plt.close()
    return fig, title


def plot_actions_no_deviation():
    """expects a single eval_log_data shaped input"""
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

    num_agents = len(agents)

    # Create a single figure with one subplot
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300, facecolor="white")

    # Verify the length of data is time_horizon
    time_horizon = eval_log_data[f"actions_1"].shape[2]
    x_range = range(time_horizon)
    discount_factor = args["normalizing_rewards_gamma"]
    title = f"Evolution of actions after forced deviation of agent 1"

    agent_colors = ["#1f77b4", "#ff7f0e"]
    for agent_idx, agent in enumerate(agents):
        per_timestep_metrics = eval_log_data
        data = per_timestep_metrics[f"actions_{agent_idx + 1}"][:, 0, :]
        ax.plot(
            x_range,
            data.mean(axis=0),
            label=f"Agent {agent_idx + 1}",
            color=agent_colors[agent_idx],
            linewidth=2,
        )
        ax.fill_between(
            x_range,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=shaded_alpha,
            color=agent_colors[agent_idx],
            linewidth=0,
        )

    ax.set_ylabel("Actions", fontsize=15, labelpad=5)

    # Add horizontal lines for competitive and collusive actions using fancy colors
    ax.axhline(2, color="#d62728", linestyle="--", linewidth=2, label="Competitive action")
    ax.axhline(12, color="#2ca02c", linestyle="--", linewidth=2, label="Collusive action")
    ax.legend(fontsize=12, frameon=True, loc="upper right", bbox_to_anchor=(1, 0.95))

    ax.set_xlabel("Timestep", fontsize=15, labelpad=5)
    ax.set_xticks(np.arange(0, time_horizon, step=1))
    ax.tick_params(axis="x", rotation=45)  # Rotated x-tick labels for better fit

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    os.makedirs(os.path.join(save_dir, "paper_plots"), exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "paper_plots", f"fig3a_{run_name}_evalep.png"),
        facecolor="white",
    )
    plt.close()
    return fig, title


try:
    plot_forced_dev()
    display_single_plot(os.path.join(save_dir, "paper_plots", f"fig3a_{run_name}_forced_dev.png"))
except Exception as e:
    print(f"Error plotting forced deviation: {e}")

try:
    plot_actions_no_deviation()
    display_single_plot(os.path.join(save_dir, "paper_plots", f"fig3a_{run_name}_evalep.png"))
except Exception as e:
    print(f"Error plotting actions: {e}")

# %%
