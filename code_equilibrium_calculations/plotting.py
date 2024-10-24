# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set up minimalistic, high-quality plot style
plt.style.use("seaborn-v0_8-whitegrid")
rcParams["font.family"] = "Helvetica"
rcParams["font.size"] = 11
rcParams["axes.linewidth"] = 0.8
rcParams["axes.edgecolor"] = "#333333"
rcParams["xtick.major.width"] = 0.8
rcParams["ytick.major.width"] = 0.8
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["axes.prop_cycle"] = plt.cycler(color=["#1f77b4", "#ff7f0e"])

# Load the data from the uploaded CSV file
file_path = "results.csv"
data = pd.read_csv(file_path)

# Split the data into segments for clarity in plotting
data_main = data[data["inventory"] <= 470]
data_extension = data[data["inventory"] >= 470]


# Function to rescale the x-axis values
def rescale_x(x):
    if x <= 365:
        return x * 0.8  # Compress values below 365
    elif 365 < x <= 470:
        return 292 + (x - 365) * 2  # Stretch values between 365 and 470
    else:
        return 502 + (x - 470) * 0.8  # Compress values above 470


# Add transformed inventory column
data_main.insert(
    0, "transformed_inventory", data_main["inventory"].apply(rescale_x), True
)
data_extension.insert(
    0, "transformed_inventory", data_extension["inventory"].apply(rescale_x), True
)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor="white")

# Plot for 'nash' and 'monop'
colors = {"monop": "#2ca02c", "nash": "#d62728"}
labels = {"monop": "Collusive", "nash": "Nash"}

for column in ["nash", "monop"]:
    color = colors[column]
    label = labels[column]
    line = ax.plot(
        data_main["transformed_inventory"],
        data_main[column],
        label=label,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=2,
        color=color,
    )
    ax.plot(
        data_extension["transformed_inventory"],
        data_extension[column],
        linestyle="-",
        linewidth=2,
        color=color,
    )

# Set x-ticks manually
original_ticks = [50, 150, 250, 365, 470]
transformed_ticks = [rescale_x(tick) for tick in original_ticks]
ax.set_xticks(transformed_ticks)
ax.set_xticklabels(original_ticks)

# Labeling and layout
# ax.set_title("Equilibrium Price for Two Equally Inventory Constrained Agents", fontsize=16, pad=20)
ax.set_xlabel("Inventory Capacity", fontsize=14, labelpad=10)
ax.set_ylabel("Equilibrium Price", fontsize=14, labelpad=10)

# Customize the legend
legend = ax.legend(
    fontsize=12, frameon=True, loc="upper right", bbox_to_anchor=(0.98, 0.98)
)
legend.get_frame().set_edgecolor("#333333")
legend.get_frame().set_linewidth(0.8)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add constraint region labels
ax.axvline(x=rescale_x(365), color="#888888", linestyle="--", linewidth=1)
ax.axvline(x=rescale_x(470), color="#888888", linestyle="--", linewidth=1)

ax.text(
    rescale_x(182.5),
    ax.get_ylim()[0],
    "Overconstrained",
    ha="center",
    va="bottom",
    fontsize=12,
    rotation=0,
    color="#555555",
)
ax.text(
    rescale_x(417.5),
    ax.get_ylim()[0],
    "Constrained",
    ha="center",
    va="bottom",
    fontsize=12,
    rotation=0,
    color="#555555",
)
ax.text(
    rescale_x(520),
    ax.get_ylim()[0],
    "Unconstrained",
    ha="center",
    va="bottom",
    fontsize=12,
    rotation=0,
    color="#555555",
)

# Add some padding to the plot
ax.set_xlim(ax.get_xlim()[0] - 10, ax.get_xlim()[1] + 10)
ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)

# Adjust layout and save
plt.tight_layout()
plt.savefig(
    "equilibria.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)


# # Display the saved figure
# plt.figure(figsize=(10, 6))
# img = plt.imread("equilibria_minimalist.png")
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# %%
