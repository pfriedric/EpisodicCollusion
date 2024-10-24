# %%
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import display_single_plot


def safe_geom_mean(x, y):
    x = np.maximum(x, 1e-10)
    y = np.maximum(y, 1e-10)
    return np.sqrt(x * y)


def geom_mean_shifted(x, y, offset):
    return np.sqrt((x + offset) * (y + offset)) - offset


def arithmetic_mean(x, y):
    return (x + y) / 2


def generalized_mean(x, p=0.5):
    ## normal version, keep for archive
    # x = np.maximum(x, 0)
    # y = np.maximum(y, 0)
    # return ((x**p + y**p) / 2) ** (1 / p)

    ## altered version
    res = np.mean(np.sign(x) * np.abs(x) ** p, axis=0)
    res = np.maximum(res, 0)

    return res ** (1 / p)


def make_paper_plot():
    # Set up the plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["text.usetex"] = True

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor="white")

    x = np.linspace(-0.3, 0.3, 100)
    y = np.full_like(x, 0.3)
    z = np.stack([x, y], axis=0)

    arithmetic_mean_values = arithmetic_mean(x, y)
    safe_geom_mean_values = safe_geom_mean(x, y)

    # Plot generalized means
    # colors = ['#ff9e6d', '#f57f5b', '#e55c5e', '#cb4679', '#a82296']  # Custom color gradient
    coolwarm_colors = [
        "#3b4cc0",
        "#6788ee",
        "#9bbcff",
        "#c6d7f2",
        "#dddcdc",
        "#f1c1bd",
        "#f88b8d",
        "#e35a59",
        "#b40426",
    ]

    cmap = plt.cm.coolwarm

    # Plot arithmetic and geometric means
    ax.plot(
        x,
        arithmetic_mean_values,
        label="Arithmetic Mean",
        color=cmap(1.0),
        linewidth=2,
        linestyle=":",
    )  # color='#d62728', linewidth=2)
    ax.plot(
        x,
        safe_geom_mean_values,
        label="Geometric Mean",
        color=cmap(0.0),
        linewidth=2,
        linestyle=":",
    )  # color='#1f77b4', linewidth=2)

    ps = [0.9, 0.7, 0.5, 0.3, 0.1]
    colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(ps))]
    for p, color in zip(ps, colors[::-1]):
        generalized_mean_values = generalized_mean(z, p)
        if p == 0.5:
            ax.plot(
                x,
                generalized_mean_values,
                label=f"Generalized Mean ($\gamma={p}$)",
                color="#888888",
                linewidth=3,
            )
        else:
            ax.plot(
                x,
                generalized_mean_values,
                label=f"Generalized Mean ($\gamma={p}$)",
                color=color,
                linewidth=3,
            )

    # Set labels and title
    ax.set_xlabel(r"Agent 1's profit gain $\Delta_e$", fontsize=16, labelpad=10)
    ax.set_ylabel("Collusion Index", fontsize=16, labelpad=10)
    # ax.set_title("Comparison of Different Means", fontsize=16, pad=20)

    # Configure the legend
    handles, labels = ax.get_legend_handles_labels()
    geometric_mean_index = labels.index("Geometric Mean")
    handles.append(handles.pop(geometric_mean_index))
    labels.append(labels.pop(geometric_mean_index))
    legend = ax.legend(
        handles,
        labels,
        fontsize=12,
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
    )
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.8)

    # Add a text annotation explaining y
    ax.text(
        0.02,
        0.9,
        r"Agent 2's profit gain $\Delta_{e,2}$ = 0.3 (constant)",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
        color="#555555",
    )

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(
        "means_comparison_plot.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


make_paper_plot()


display_single_plot("means_comparison_plot.png")
