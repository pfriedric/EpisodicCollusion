import numpy as np
import os
import matplotlib.pyplot as plt


def rescale_zero_to_one(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def rename_hyperparam(key):
    return (
        key.replace("dqn_default", "DQN")
        .replace("epsilon_finish", "epsilon")
        .replace("target_update_interval", "target_update")
        .replace("training_interval_episodes", "train_eps_interval")
        .replace("/", "_")
        .replace("ppo_default", "PPO")
    )


def get_training_run_based_collusion_index(
    all_env_stats, agents, last_ten_percent_episodes, hyperparam_combo: int = 0
):
    j = hyperparam_combo
    avg_collusion_index = np.zeros((num_seeds, len(x_axis_long)))
    for agent_idx, agent in enumerate(agents):
        agent_seed_means = all_env_stats[f"train/collusion_index/mean_player_{agent_idx + 1}"][
            :, j, :
        ]  # shape [seeds, T], slicing jth hyperparam combo
        avg_collusion_index += agent_seed_means
    avg_collusion_index /= len(agents)  # shape [seeds, T]

    # each seed: average over last 10% of training run
    avg_training_collusion_index_per_seed = np.mean(
        avg_collusion_index[:, last_ten_percent_episodes:], axis=1
    )  # [seeds]
    avg_training_collusion_index_mean = np.mean(avg_training_collusion_index_per_seed)  # scalar
    avg_training_collusion_index_var = np.var(avg_training_collusion_index_per_seed)  # scalar

    return (
        avg_training_collusion_index_per_seed,
        avg_training_collusion_index_mean,
        avg_training_collusion_index_var,
    )


def apply_moving_average(data, x_axis, window_size):
    if data.ndim == 2:
        ma = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window_size) / window_size, mode="valid"),
            axis=1,
            arr=data,
        )
    else:
        ma = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    # Adjust x-axis to match the moving average data
    adjusted_x_axis = x_axis[window_size - 1 :]
    return ma, adjusted_x_axis


def overall_mean_stdev_from_seed_means_variances(seed_means, seed_variances, num_envs):
    """
    Calculate the overall mean and standard deviation across seeds and environments.

    :param seed_means: numpy array of shape (num_seeds, num_time_steps)
    :param seed_variances: numpy array of shape (num_seeds, num_time_steps)
    :return: tuple of (overall_mean, overall_std), each of shape (num_time_steps,)
    """
    num_seeds = seed_means.shape[0]
    overall_mean = np.mean(seed_means, axis=0)
    between_seed_var = np.sum(num_envs * (seed_means - overall_mean) ** 2, axis=0)
    within_seed_var = np.sum((num_envs - 1) * seed_variances, axis=0)
    overall_variance = (between_seed_var + within_seed_var) / (num_seeds * num_envs - 1)
    overall_std = np.sqrt(overall_variance)
    return overall_mean, overall_std


def plot_agent_metrics_shaded(
    axs,
    agent_seed_means,
    agent_seed_vars,
    x_axis,
    agent_idx,
    metric_name,
    num_envs,
):
    """Adds line for mean and std of any agent metric over seeds to provided plot axs."""
    # provided means/vars are per seed, over a group of environments -- we want the mean/var over all seeds/envs
    metric_mean, metric_std = overall_mean_stdev_from_seed_means_variances(
        agent_seed_means, agent_seed_vars, num_envs
    )
    (line,) = axs.plot(
        x_axis,
        metric_mean,
        label=f"Agent {agent_idx + 1}",
        linewidth=0.75,
    )
    axs.fill_between(
        x_axis,
        metric_mean - metric_std,
        metric_mean + metric_std,
        alpha=0.3,
    )
    axs.set_title(f"{metric_name}")
    return axs, line, metric_mean, metric_std


def plot_agent_metrics_shaded_selectable_color(
    axs,
    agent_seed_means,
    agent_seed_vars,
    x_axis,
    agent_idx,
    metric_name,
    num_envs,
    agent_color=None,
):
    """Adds line for mean and std of any agent metric over seeds to provided plot axs."""
    # provided means/vars are per seed, over a group of environments -- we want the mean/var over all seeds/envs
    metric_mean, metric_std = overall_mean_stdev_from_seed_means_variances(
        agent_seed_means, agent_seed_vars, num_envs
    )

    # Plot the shaded area first
    axs.fill_between(
        x_axis,
        metric_mean - metric_std,
        metric_mean + metric_std,
        alpha=0.3,
        color=agent_color,
        linewidth=0,  # this is key!! otherwise, edge line with custom colors
    )

    # Plot the line on top of the shaded area
    (line,) = axs.plot(
        x_axis,
        metric_mean,
        label=f"Agent {agent_idx + 1}",
        linewidth=0.75,
        color=agent_color,
    )
    axs.set_title(f"{metric_name}", fontsize=16)
    return axs, line, metric_mean, metric_std


def plot_agent_metrics_indiv_seeds(
    axs, agent_seed_means, agent_seed_vars, x_axis, agent_idx, metric_name, num_envs
):
    """Adds line for mean and individual seed lines of any agent metric to provided plot axs."""
    # provided means/vars are per seed, over a group of environments -- we want the mean/var over all seeds/envs
    metric_mean, metric_std = overall_mean_stdev_from_seed_means_variances(
        agent_seed_means, agent_seed_vars, num_envs
    )
    window_size = 10
    metric_mean_ma, ma_x_axis = apply_moving_average(metric_mean, x_axis, window_size)
    (line,) = axs.plot(ma_x_axis, metric_mean_ma, label=f"Agent {agent_idx + 1}", linewidth=0.75)
    color = line.get_color()
    alpha = 0.05
    num_seeds = agent_seed_means.shape[0]
    if num_seeds > 10 and num_seeds <= 50:
        alpha = 0.03
    elif num_seeds > 50:
        alpha = 0.01
    for seed_mean in agent_seed_means:
        seed_mean_ma, _ = apply_moving_average(seed_mean, x_axis, window_size)
        axs.plot(ma_x_axis, seed_mean_ma, color=color, alpha=alpha, linewidth=0.75)
    axs.set_title(f"{metric_name}")
    return axs, line, metric_mean, metric_std


def display_single_plot(plot_path):
    print(f"Displaying {os.path.basename(plot_path)}")
    img = plt.imread(plot_path)
    dpi = plt.rcParams["figure.dpi"]
    height, width, _ = img.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    plt.show()
