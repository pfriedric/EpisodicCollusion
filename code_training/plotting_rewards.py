# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import seaborn as sns

from environment.market_env import MarketEnv, EnvState
from main import possible_prices_func

# %%
# defining params
num_prices = 15
xi = 0.2
qualities = jnp.array([2, 2])
marginal_costs = jnp.array([1, 1])
horizontal_diff = 0.25
demand_scaling_factor = 1000
p_N_unconstrained = 1.471  # demand: 470
p_N_455 = 1.617  # demand: 455
p_N_440 = 1.693  # demand: 440
p_N_425 = 1.74  # demand: 425
p_N_420 = 1.7588  # demand: 420
p_N_410 = 1.795  # demand: 410
p_N_395 = 1.843  # demand: 395
p_N_380 = 1.885  # demand: 380
p_N_365 = 1.925  # demand: 365
p_M_365 = 1.925  # demand: 365


env = MarketEnv(num_agents=2, num_actions=num_prices, time_horizon=1)

# calc unconstrained demand using the unconstrained competitive & collusive prices
_, _, _, _, _, unconstrained_competitive_price, unconstrained_collusive_price = (
    possible_prices_func(
        p_N=p_N_unconstrained, p_M=p_M_365, num_price_steps=num_prices, xi=xi
    )
)

# calc constrained demand at 420 inventory level
_, _, _, _, _, constrained_competitive_price, constrained_collusive_price = (
    possible_prices_func(p_N=p_N_420, p_M=p_M_365, num_price_steps=num_prices, xi=xi)
)

# Create a dummy state for demand calculation
dummy_state = EnvState(
    inventories=jnp.array([100, 100]),  # Assuming large inventory
    last_prices=jnp.array([1.0, 1.0]),  # Dummy values
    last_actions=jnp.array([0, 0]),  # Dummy values
    t=jnp.array([0]),  # Dummy value
)

print("USAGE")
print("price_constraint determines the price grid's width")
print(
    "reward_constraint is the actual inventory constraint used in the calculation of agent rewards"
)
print(
    "reward_constraint_for_comparison_grid is the reward constraint used for the 2nd scenario that's constructed and can be used for rescaling"
)
# demand_unconstrained_competitive = env.MNL_demand(
#     dummy_state,
#     jnp.array([unconstrained_competitive_price, unconstrained_competitive_price]),
#     qualities,
#     horizontal_diff,
#     demand_scaling_factor,
# )

# demand_constrained_competitive = env.MNL_demand(
#     dummy_state,
#     jnp.array([constrained_competitive_price, constrained_competitive_price]),
#     qualities,
#     horizontal_diff,
#     demand_scaling_factor,
# )
# print(f"demand_unconstrained_competitive: {demand_unconstrained_competitive}")
# print(f"demand_constrained_competitive: {demand_constrained_competitive}")


def calculate_reward(price, demand):
    return (price - 1) * demand


def calculate_reward_matrices(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    # data structure that reads the Nash price from the price_constraint level.
    # this is used to determine p_N for the price grid, and the Nash price from the reward_constraint level
    # to color the "true" NE in the heatmap
    possible_nash_prices_dict = {
        "unconstrained": 1.471,
        "470": 1.471,
        "455": 1.617,
        "440": 1.693,
        "425": 1.74,
        "420": 1.7588,
        "410": 1.795,
        "395": 1.843,
        "380": 1.885,
        "365": 1.925,
        "230": 2.213,
    }
    possible_monop_prices_dict = {
        "unconstrained": 1.925,
        "470": 1.925,
        "455": 1.925,
        "440": 1.925,
        "425": 1.925,
        "420": 1.925,
        "410": 1.925,
        "395": 1.925,
        "380": 1.925,
        "365": 1.925,
        "230": 2.213,
    }
    try:
        p_N = possible_nash_prices_dict[price_constraint]
        p_M = possible_monop_prices_dict[price_constraint]
    except KeyError:
        raise ValueError(
            f"Invalid price_constraint: {price_constraint}. possible values: {possible_nash_prices_dict.keys()} "
        )

    # find the price in the dict that is closest to the reward_constraint, excluding "unconstrained"
    closest_price = min(
        (k for k in possible_nash_prices_dict if k != "unconstrained"),
        key=lambda x: abs(float(x) - reward_constraint),
    )
    reward_constraint_induced_nash_price = possible_nash_prices_dict[closest_price]

    # Create a grid of possible prices -- uses the price_constraint
    (
        prices,
        _,
        _,
        _,
        collusive_action,
        _,
        _,
    ) = possible_prices_func(p_N=p_N, p_M=p_M, num_price_steps=num_prices, xi=xi)
    print(f"prices: {prices}")

    # Calculate individual rewards -- these use the reward constraint
    average_reward_matrix = jnp.zeros((num_prices, num_prices))
    reward_matrix_agent1 = np.zeros((num_prices, num_prices))
    reward_matrix_agent2 = np.zeros((num_prices, num_prices))

    compared_average_reward_matrix = jnp.zeros((num_prices, num_prices))
    compared_reward_matrix_agent1 = jnp.zeros((num_prices, num_prices))
    compared_reward_matrix_agent2 = jnp.zeros((num_prices, num_prices))

    for i, price1 in enumerate(prices):
        for j, price2 in enumerate(prices):
            actions = jnp.array([price1, price2])
            demands = env.MNL_demand(
                dummy_state, actions, qualities, horizontal_diff, demand_scaling_factor
            )
            demand1 = jnp.minimum(demands[0], reward_constraint)
            demand2 = jnp.minimum(demands[1], reward_constraint)
            reward1 = calculate_reward(price1, demand1)
            reward2 = calculate_reward(price2, demand2)
            average_reward = (reward1 + reward2) / 2
            reward_matrix_agent1[i, j] = reward1
            reward_matrix_agent2[i, j] = reward2
            average_reward_matrix = average_reward_matrix.at[i, j].set(average_reward)

            # By passing reward_constraint_for_comparison_grid (default: unconstrained), we can rescale the output using the
            # lowest / highest rewards across both unconstrained & constrained setting.
            # b/c default is unconstrained, `reward_constraint_for_comparison_grid` only needs to be passed if calculating unconstrained rewards
            compared_demand1 = jnp.minimum(
                reward_constraint_for_comparison_grid, demands[0]
            )
            compared_demand2 = jnp.minimum(
                reward_constraint_for_comparison_grid, demands[1]
            )
            compared_reward1 = calculate_reward(price1, compared_demand1)
            compared_reward2 = calculate_reward(price2, compared_demand2)
            compared_average_reward = (compared_reward1 + compared_reward2) / 2
            compared_reward_matrix_agent1 = compared_reward_matrix_agent1.at[i, j].set(
                compared_reward1
            )
            compared_average_reward_matrix = compared_average_reward_matrix.at[
                i, j
            ].set(compared_average_reward)

            ## tracking the lowest/highest rewards for scaling.
            # we can
            # - constrain price grid ("zooming in" on the heatmap)
            # - constrain rewards (capping rewards if demand is too big).
            # This creates 4 scenarios where prices and rewards can each be capped/uncapped.
            # Ultimately we want to decide on what the prices are, and then compare how rewards change from unconstrained->constrained
            #   so both these should be rescaled with the same numbers (as in the experiment, we'll only apply a single rescaling to all rewards over all time)
            # But: which number should be used as the lower & upper range?
            # Note: moving from unconstrained->constrained rewards,
            #   1) for average reward the low-end will go down (case ag1=max, ag2=min, but ag2 can't capitalize anymore)
            #      while top end stays same (monopoly)
            #      --> use constrained min, either max
            #   2) for individual reward, low-end stays same (case ag1=max, ag2=min),
            #      but top-end goes down (ag1 can't capitalize if ag2 increases their price above ag1's)
            #      --> use either min, unconstrained max

    lowest_average_reward = min(
        jnp.min(average_reward_matrix), jnp.min(compared_average_reward_matrix)
    )
    highest_average_reward = max(
        jnp.max(average_reward_matrix), jnp.max(compared_average_reward_matrix)
    )
    lowest_individual_reward = min(
        jnp.min(reward_matrix_agent1), jnp.min(compared_reward_matrix_agent1)
    )
    highest_individual_reward = max(
        jnp.max(reward_matrix_agent1), jnp.max(compared_reward_matrix_agent1)
    )

    print(f"lowest_average_reward: {lowest_average_reward}")

    avg_scaling_min = jnp.min(average_reward_matrix)
    avg_scaling_max = jnp.max(average_reward_matrix)
    indiv_scaling_min = jnp.min(reward_matrix_agent1)
    indiv_scaling_max = jnp.max(reward_matrix_agent1)
    if rescale_reward == "min_max_over_computed_grid":
        # Rescale each matrix to [0, 1] using min-max normalization
        average_reward_matrix = (average_reward_matrix - avg_scaling_min) / (
            avg_scaling_max - avg_scaling_min
        )
        reward_matrix_agent1 = (reward_matrix_agent1 - indiv_scaling_min) / (
            indiv_scaling_max - indiv_scaling_min
        )
        reward_matrix_agent2 = (reward_matrix_agent2 - indiv_scaling_min) / (
            indiv_scaling_max - indiv_scaling_min
        )
    elif rescale_reward == "only_max_over_computed_and_compared_grid":
        # Rescale each matrix by dividing by THE UNCONSTRAINED maximum value
        # reason: even in a constrained setting, beginning of episode is likely unconstrained
        avg_scaling_max = highest_average_reward
        indiv_scaling_max = highest_individual_reward
        average_reward_matrix = average_reward_matrix / avg_scaling_max
        reward_matrix_agent1 = reward_matrix_agent1 / indiv_scaling_max
        reward_matrix_agent2 = reward_matrix_agent2 / indiv_scaling_max

    elif rescale_reward == "min_max_over_computed_and_compared_grid":
        # Rescale each matrix to [0, 1] using min-max normalization, but using the lowest min / highest max among constrained & unconstrained rewards
        avg_scaling_max = highest_average_reward
        avg_scaling_min = lowest_average_reward
        indiv_scaling_max = highest_individual_reward
        indiv_scaling_min = lowest_individual_reward
        # Rescale each matrix to [0, 1] using min-max normalization, but using the unconstrained max
        average_reward_matrix = (average_reward_matrix - avg_scaling_min) / (
            avg_scaling_max - avg_scaling_min
        )
        reward_matrix_agent1 = (reward_matrix_agent1 - indiv_scaling_min) / (
            indiv_scaling_max - indiv_scaling_min
        )
        reward_matrix_agent2 = (reward_matrix_agent2 - indiv_scaling_min) / (
            indiv_scaling_max - indiv_scaling_min
        )

    elif rescale_reward == "min_max_over_compared_grid":
        avg_scaling_max = jnp.max(compared_average_reward_matrix)
        avg_scaling_min = jnp.min(compared_average_reward_matrix)
        indiv_scaling_max = jnp.max(compared_reward_matrix_agent1)
        indiv_scaling_min = jnp.min(compared_reward_matrix_agent1)
        average_reward_matrix = (average_reward_matrix - avg_scaling_min) / (
            avg_scaling_max - avg_scaling_min
        )
        reward_matrix_agent1 = (reward_matrix_agent1 - indiv_scaling_min) / (
            indiv_scaling_max - indiv_scaling_min
        )
        reward_matrix_agent2 = (reward_matrix_agent2 - indiv_scaling_min) / (
            indiv_scaling_max - indiv_scaling_min
        )

    scales = (
        avg_scaling_min,
        avg_scaling_max,
        indiv_scaling_min,
        indiv_scaling_max,
    )

    # find the price in `prices` that is closest to the reward_constraint_induced_nash_price
    # this is used to mark the "true" (according to agent's inventory constraint) NE in the heatmap
    closest_price_index = min(
        range(len(prices)),
        key=lambda i: abs(prices[i] - reward_constraint_induced_nash_price),
    )
    competitive_action = closest_price_index

    return (
        average_reward_matrix,
        reward_matrix_agent1,
        reward_matrix_agent2,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    )


# %%
def create_average_reward_heatmap(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    (
        average_reward_matrix,
        _,
        _,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(
        price_constraint,
        reward_constraint,
        rescale_reward,
        reward_constraint_for_comparison_grid,
    )

    # Flip the reward matrix vertically to match the reversed prices
    average_reward_matrix = jnp.flipud(average_reward_matrix)
    prices1 = prices[::-1]
    prices2 = prices

    # print the min- and max rewards
    print(f"min reward: {average_reward_matrix.min()}")
    print(f"max reward: {average_reward_matrix.max()}")

    # Create heatmap
    plt.figure(figsize=(10, 8))

    # Create a mask for highlighting specific cells
    highlight_mask = np.zeros_like(average_reward_matrix, dtype=bool)
    highlight_mask[num_prices - competitive_action - 1, competitive_action] = True
    highlight_mask[num_prices - collusive_action - 1, collusive_action] = True

    # format of the numbers: if no rescaling: integer. if rescaling: 2 decimal digits
    fmt = ".0f" if not rescale_reward else ".2f"

    # Plot the main heatmap
    ax = sns.heatmap(
        average_reward_matrix,
        xticklabels=prices2,
        yticklabels=prices1,
        cmap="YlOrRd",
        annot=True,
        fmt=fmt,
        annot_kws={"size": 6},
    )

    # Highlight the specific cells
    for i, j in zip(*np.where(highlight_mask)):
        text = ax.texts[i * average_reward_matrix.shape[1] + j]
        text.set_weight("bold")
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))

    # a string that adds the min and max rewards to the title, if we're scaling, and is aware of which matrix is being plotted
    (
        avg_scaling_min,
        avg_scaling_max,
        _,
        _,
    ) = scales
    title = "Average rewards: "
    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "prices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += "prices constrained"
        else:
            title += "prices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"
        else:
            title += ", inventory unconstrained"

    if rescale_reward in [
        "min_max_over_computed_grid",
        "min_max_over_computed_and_compared_grid",
    ]:
        # scaling source min: it's the reward constrained-min, unless we're doing situation specific and asking for unconstrained
        scaling_source_min = (
            "(unconstrained)"
            if rescale_reward == "min_max_over_computed_grid"
            and reward_constraint >= 1000
            else "(constrained)"
        )
        title += f"\nScaling: [0, 1]-trafo with min={avg_scaling_min:.0f} {scaling_source_min}, max={avg_scaling_max:.0f}"

    elif rescale_reward == "only_max_over_computed_and_compared_grid":
        title += f"\nScaling: divided by max of {avg_scaling_max:.0f}"

    elif rescale_reward == "min_max_over_compared_grid":
        # if reward_constraint_for_comparison_grid < 1000, pass constrained.
        scaling_source = (
            "(constrained)"
            if reward_constraint_for_comparison_grid < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: [0, 1]-trafo with min={avg_scaling_min:.0f} {scaling_source}, max={avg_scaling_max:.0f}"

    plt.xlabel("Agent 2 Price")
    plt.ylabel("Agent 1 Price")
    plt.title(title)
    plt.show()

    #


def make_four_average_reward_heatmaps(
    rescale_reward=False, price_constraint="unconstrained", reward_constraint=1e6
):

    ## UNCONSTRAINED PRICES
    print(f"unconstrained setting")
    create_average_reward_heatmap(
        price_constraint="unconstrained",
        reward_constraint=1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print(f"unconstrained prices, but constrained rewards:")
    create_average_reward_heatmap(
        price_constraint="unconstrained",
        reward_constraint=reward_constraint,
        rescale_reward=rescale_reward,
    )

    ## CONSTRAINED PRICES
    print(f"constrained prices, but unconstrained rewards: (at beginning of episode)")
    create_average_reward_heatmap(
        price_constraint=price_constraint,
        reward_constraint=1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print(f"constrained prices and rewards")
    create_average_reward_heatmap(
        price_constraint=price_constraint,
        reward_constraint=reward_constraint,
        rescale_reward=rescale_reward,
    )


make_four_average_reward_heatmaps(
    rescale_reward=False,
    price_constraint="440",
    reward_constraint=440,
)


# %%
### INDIVIDUAL REWARD HEATMAP ###
def create_individual_reward_heatmap(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    (
        _,
        reward_matrix_agent1,
        reward_matrix_agent2,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(
        price_constraint,
        reward_constraint,
        rescale_reward,
        reward_constraint_for_comparison_grid,
    )

    # Flip matrices vertically to match reversed prices
    reward_matrix_agent1 = np.flipud(reward_matrix_agent1)
    reward_matrix_agent2 = np.flipud(reward_matrix_agent2)

    plt.figure(figsize=(12, 10))

    # Use agent 1's rewards for coloring
    # Create a mask for competitive and collusive equilibria
    mask = np.zeros_like(reward_matrix_agent1)
    mask[num_prices - competitive_action - 1, competitive_action] = 1
    mask[num_prices - collusive_action - 1, collusive_action] = 1

    # Plot the main heatmap
    ax = sns.heatmap(
        reward_matrix_agent1,
        xticklabels=prices,
        yticklabels=prices[::-1],
        cmap="YlOrRd",
        annot=False,
        cbar_kws={"label": "Agent 1 Reward"},
    )

    # Highlight the specific cells
    for i, j in zip(*np.where(mask)):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))

    # format of the numbers: if no rescaling: integer. if rescaling: 2 decimal digits
    fmt = ".0f" if not rescale_reward else ".2f"

    # Annotate with both agents' rewards
    for i in range(num_prices):
        for j in range(num_prices):
            plt.text(
                j + 0.5,
                i + 0.5,
                f"{reward_matrix_agent1[i,j]:{fmt}}\n{reward_matrix_agent2[i,j]:{fmt}}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    # # Add legend for competitive and collusive equilibria
    # plt.plot([], [], 's', color='black', alpha=0.3, label='Equilibria')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.xlabel("Agent 2 Price")
    plt.ylabel("Agent 1 Price")

    indiv_scaling_min, indiv_scaling_max = scales[2:4]

    # if reward_constraint < 1000:
    #     title = f"Individual Rewards (Agent 1 color, Agent 1 top / Agent 2 bottom) - constrained to {reward_constraint} inventory level"
    # else:
    #     title = f"Individual Rewards (Agent 1 color, Agent 1 top / Agent 2 bottom) - unconstrained"
    title = f"Individual Rewards (Agent 1 color, Agent 1 top / Agent 2 bottom)"

    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "\nPrices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += "\nPrices constrained"
        else:
            title += "\nPrices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"
        else:
            title += ", inventory unconstrained"

    if rescale_reward in [
        "min_max_over_computed_grid",
        "min_max_over_computed_and_compared_grid",
    ]:
        scaling_source = (
            "(constrained)"
            if rescale_reward == "min_max_over_computed_grid"
            and reward_constraint < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f} {scaling_source}"

    if rescale_reward == "min_max_over_compared_grid":
        # if reward_constraint_for_comparison_grid < 1000, pass constrained.
        scaling_source = (
            "(constrained)"
            if reward_constraint_for_comparison_grid < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f} {scaling_source}"

    plt.title(title)
    plt.show()


def make_four_individual_reward_heatmaps(
    rescale_reward=False, price_constraint="unconstrained", reward_constraint=1e6
):
    """If using 'min_max_over_computed_and_compared_grid' or 'min_max_over_compared_grid':
    reward_constraint_for_comparison_grid defaults to 1e6 (unconstrained)
    So:
    - if 'min_max_over_computed_and_compared_grid', must pass =reward_constraint for the unconstrained-price case
    - if 'min_max_over_compared_grid', must pass the same value for both cases (probably want 'constrained')
    """

    ## UNCONSTRAINED PRICES
    print(f"unconstrained setting")
    create_individual_reward_heatmap(
        "unconstrained",
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print(f"unconstrained prices, but constrained rewards:")
    create_individual_reward_heatmap(
        "unconstrained",
        reward_constraint,
        rescale_reward=rescale_reward,
    )

    ## CONSTRAINED PRICES
    print(f"constrained prices, but unconstrained rewards: (at beginning of episode)")
    create_individual_reward_heatmap(
        price_constraint,
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print(f"constrained prices and rewards")
    create_individual_reward_heatmap(
        price_constraint,
        reward_constraint,
        rescale_reward=rescale_reward,
    )

    ## UNCONSTRAINED PRICES, NORMALIZED ACCORDING TO CONSTRAINED REWARDS
    print(f"unconstrained setting, normalized according to constrained rewards")
    create_individual_reward_heatmap(
        "unconstrained",
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )

    print(
        f"unconstrained prices, but constrained rewards, normalized according to constrained rewards"
    )
    create_individual_reward_heatmap(
        "unconstrained",
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )


make_four_individual_reward_heatmaps(
    rescale_reward="min_max_over_computed_and_compared_grid",  # "min_max_over_compared_grid",  # "min_max_over_computed_and_compared_grid",
    price_constraint="420",
    reward_constraint=420,
)


# %%
### REWARD DIFFERENCE HEATMAP ###
def create_reward_difference_heatmap(
    price_constraint, reward_constraint, rescale_reward=False
):
    (
        _,
        reward_matrix_agent1,
        reward_matrix_agent2,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(price_constraint, reward_constraint, rescale_reward)

    # Calculate reward differences
    reward_diff_matrix = reward_matrix_agent1 - reward_matrix_agent2

    # Flip matrix vertically to match reversed prices
    reward_diff_matrix = np.flipud(reward_diff_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        reward_diff_matrix,
        xticklabels=prices,
        yticklabels=prices[::-1],
        cmap="RdBu",
        center=0,
        annot=True,
        fmt=".2f" if rescale_reward else ".0f",
    )

    plt.xlabel("Agent 2 Price")
    plt.ylabel("Agent 1 Price")
    title = f"Reward Difference (Agent 1 - Agent 2)"
    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "\nPrices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += "\nPrices constrained"
        else:
            title += "\nPrices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"

    plt.title(title)
    plt.show()


def make_four_reward_difference_heatmaps(rescale_reward=False):
    print("Unconstrained setting")
    create_reward_difference_heatmap(
        "unconstrained", 1e6, rescale_reward=rescale_reward
    )
    print("Constrained prices and rewards")
    create_reward_difference_heatmap("420", 420, rescale_reward=rescale_reward)


make_four_reward_difference_heatmaps(rescale_reward=False)

# %%
from mpl_toolkits.mplot3d import Axes3D


def create_3d_reward_surface(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    (
        average_reward_matrix,
        _,
        _,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(
        price_constraint,
        reward_constraint,
        rescale_reward,
        reward_constraint_for_comparison_grid,
    )

    avg_scaling_min, avg_scaling_max = scales[:2]

    # Flip matrix vertically to match reversed prices
    # average_reward_matrix = np.flipud(average_reward_matrix)

    X, Y = np.meshgrid(prices, prices)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, average_reward_matrix, cmap="viridis")

    ax.set_xlabel("Agent 2 Price")
    ax.set_ylabel("Agent 1 Price")
    ax.set_zlabel("Average Reward")
    title = "3D Reward Surface of average reward"
    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "\nPrices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += "\nPrices constrained"
        else:
            title += "\nPrices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"
        else:
            title += ", inventory unconstrained"

    if rescale_reward in ["min_max", "min_max_unconstrained"]:
        scaling_source = (
            "(constrained)"
            if rescale_reward == "min_max" and reward_constraint < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={avg_scaling_min:.0f}, max={avg_scaling_max:.0f} {scaling_source}"

    ax.set_title(title)

    # ax.set_xticks(prices)
    # ax.set_yticks(prices)

    fig.colorbar(surf)
    plt.show()


def make_four_3d_reward_surfaces(
    rescale_reward=False, price_constraint="unconstrained", reward_constraint=1e6
):
    print("Unconstrained setting")
    create_3d_reward_surface(
        "unconstrained",
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print(f"unconstrained prices, but constrained rewards:")
    create_3d_reward_surface(
        "unconstrained", reward_constraint, rescale_reward=rescale_reward
    )
    print(f"constrained prices, but unconstrained rewards: (at beginning of episode)")
    create_3d_reward_surface(
        price_constraint,
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices and rewards")
    create_3d_reward_surface(
        price_constraint, reward_constraint, rescale_reward=rescale_reward
    )


make_four_3d_reward_surfaces(
    rescale_reward="min_max_over_computed_and_compared_grid",
    price_constraint="420",
    reward_constraint=420,
)


# %%
### CONTOUR PLOTS ###
def create_average_reward_contour_plot(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    (
        average_reward_matrix,
        _,
        _,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(
        price_constraint,
        reward_constraint,
        rescale_reward,
        reward_constraint_for_comparison_grid,
    )

    avg_scaling_min, avg_scaling_max = scales[:2]

    X, Y = np.meshgrid(prices, prices)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, average_reward_matrix, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Average Reward")

    # Mark competitive and collusive points
    plt.plot(
        prices[competitive_action],
        prices[competitive_action],
        "ro",
        markersize=10,
        label="Competitive",
    )
    plt.plot(
        prices[collusive_action],
        prices[collusive_action],
        "go",
        markersize=10,
        label="Collusive",
    )

    plt.xlabel("Agent 2 Price")
    plt.ylabel("Agent 1 Price")
    title = "Average Reward Contour Plot"
    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "\nPrices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += f"\nPrices constrained at {price_constraint}"
        else:
            title += "\nPrices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"
        else:
            title += ", inventory unconstrained"

    if rescale_reward in [
        "min_max_over_computed_grid",
        "min_max_over_computed_and_compared_grid",
    ]:
        scaling_source = (
            "(constrained)"
            if rescale_reward == "min_max_over_computed_grid"
            and reward_constraint < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={avg_scaling_min:.0f}, max={avg_scaling_max:.0f} {scaling_source}"
    elif rescale_reward == "only_max_over_computed_and_compared_grid":
        title += f"\nScaling: divided by max of {avg_scaling_max:.0f}"
    elif rescale_reward == "min_max_over_compared_grid":
        scaling_source = (
            "(constrained)"
            if reward_constraint_for_comparison_grid < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={avg_scaling_min:.0f}, max={avg_scaling_max:.0f} {scaling_source}"

    plt.title(title)
    plt.legend()
    plt.show()


def make_four_average_reward_contour_plots(
    rescale_reward=False, price_constraint="unconstrained", reward_constraint=1e6
):
    print("Unconstrained setting")
    create_average_reward_contour_plot(
        "unconstrained",
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Unconstrained prices, but constrained rewards:")
    create_average_reward_contour_plot(
        "unconstrained",
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices, but unconstrained rewards: (at beginning of episode)")
    create_average_reward_contour_plot(
        price_constraint,
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices and rewards")
    create_average_reward_contour_plot(
        price_constraint,
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )


# Usage
make_four_average_reward_contour_plots(
    rescale_reward="min_max_over_compared_grid",
    price_constraint="420",
    reward_constraint=420,
)


# %%
### INDIVIDUAL REWARD CONTOUR PLOTS ###
def create_individual_reward_contour_plot(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    (
        _,
        reward_matrix_agent1,
        reward_matrix_agent2,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(
        price_constraint,
        reward_constraint,
        rescale_reward,
        reward_constraint_for_comparison_grid,
    )

    indiv_scaling_min, indiv_scaling_max = scales[2:4]

    X, Y = np.meshgrid(prices, prices)

    plt.figure(figsize=(12, 10))
    contour = plt.contourf(X, Y, reward_matrix_agent1, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Agent 1 Reward")

    # Mark competitive and collusive points
    plt.plot(
        prices[competitive_action],
        prices[competitive_action],
        "ro",
        markersize=10,
        label="Competitive",
    )
    plt.plot(
        prices[collusive_action],
        prices[collusive_action],
        "go",
        markersize=10,
        label="Collusive",
    )

    # Add text annotations for both agents' rewards
    for i in range(num_prices):
        for j in range(num_prices):
            fmt = ".0f" if not rescale_reward else ".2f"
            plt.text(
                prices[j],
                prices[i],
                f"{reward_matrix_agent1[i,j]:{fmt}}\n{reward_matrix_agent2[i,j]:{fmt}}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    plt.xlabel("Agent 2 Price")
    plt.ylabel("Agent 1 Price")
    title = (
        "Individual Reward Contour Plot (Agent 1 color, Agent 1 top / Agent 2 bottom)"
    )
    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "\nPrices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += f"\nPrices constrained at {price_constraint}"
        else:
            title += "\nPrices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"
        else:
            title += ", inventory unconstrained"

    if rescale_reward in [
        "min_max_over_computed_grid",
        "min_max_over_computed_and_compared_grid",
    ]:
        scaling_source = (
            "(constrained)"
            if rescale_reward == "min_max_over_computed_grid"
            and reward_constraint < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f} {scaling_source}"
    elif rescale_reward == "only_max_over_computed_and_compared_grid":
        title += f"\nScaling: divided by max of {indiv_scaling_max:.0f}"
    elif rescale_reward == "min_max_over_compared_grid":
        scaling_source = (
            "(constrained)"
            if reward_constraint_for_comparison_grid < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f} {scaling_source}"
    elif rescale_reward == False:
        title += (
            f"\nNo rescaling. min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f}"
        )

    plt.title(title)
    plt.legend()
    plt.show()


def make_four_individual_reward_contour_plots(
    rescale_reward=False, price_constraint="unconstrained", reward_constraint=1e6
):
    print("Unconstrained setting")
    create_individual_reward_contour_plot(
        "unconstrained",
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Unconstrained prices, but constrained rewards:")
    create_individual_reward_contour_plot(
        "unconstrained",
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices, but unconstrained rewards: (at beginning of episode)")
    create_individual_reward_contour_plot(
        price_constraint,
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices and rewards")
    create_individual_reward_contour_plot(
        price_constraint,
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )


# Usage
make_four_individual_reward_contour_plots(
    rescale_reward=False,  # "min_max_over_compared_grid",
    price_constraint="455",
    reward_constraint=455,
)

# make_four_individual_reward_contour_plots()

# %%


def create_individual_3d_reward_plot(
    price_constraint,
    reward_constraint,
    rescale_reward=False,
    reward_constraint_for_comparison_grid=1e6,
):
    (
        _,
        reward_matrix_agent1,
        reward_matrix_agent2,
        prices,
        num_prices,
        competitive_action,
        collusive_action,
        scales,
    ) = calculate_reward_matrices(
        price_constraint,
        reward_constraint,
        rescale_reward,
        reward_constraint_for_comparison_grid,
    )

    indiv_scaling_min, indiv_scaling_max = scales[2:4]

    X, Y = np.meshgrid(prices, prices)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    mask_agent1_higher = reward_matrix_agent1 > reward_matrix_agent2
    mask_agent2_higher = reward_matrix_agent2 > reward_matrix_agent1
    mask_equal = np.isclose(reward_matrix_agent1, reward_matrix_agent2, rtol=1e-5)

    # Plot Agent 1's surface
    surf1 = ax.plot_surface(
        X,
        Y,
        np.where(mask_agent1_higher | mask_equal, reward_matrix_agent1, np.nan),
        cmap="summer",
        alpha=0.9,
    )
    ax.plot_surface(X, Y, reward_matrix_agent1, cmap="summer", alpha=0.3)

    # Plot Agent 2's surface
    surf2 = ax.plot_surface(
        X,
        Y,
        np.where(mask_agent2_higher | mask_equal, reward_matrix_agent2, np.nan),
        cmap="autumn",
        alpha=0.9,
    )
    ax.plot_surface(X, Y, reward_matrix_agent2, cmap="autumn", alpha=0.3)

    # Plot the intersection
    intersection = ax.plot_surface(
        X,
        Y,
        np.where(mask_equal, reward_matrix_agent1, np.nan),
        color="purple",
        alpha=1,
    )

    ax.invert_xaxis()

    ax.set_xlabel("Agent 2 Price")
    ax.set_ylabel("Agent 1 Price")
    ax.set_zlabel("Agent 1's reward")
    title = (
        "Individual Reward Contour Plot (Agent 1 color, Agent 1 top / Agent 2 bottom)"
    )
    if price_constraint == "unconstrained" and reward_constraint >= 1000:
        title += "\nPrices and inventory unconstrained"
    else:
        if price_constraint != "unconstrained":
            title += f"\nPrices constrained at {price_constraint}"
        else:
            title += "\nPrices unconstrained"

        if reward_constraint < 1000:
            title += f", inventory constrained at I={reward_constraint}"
        else:
            title += ", inventory unconstrained"

    if rescale_reward in [
        "min_max_over_computed_grid",
        "min_max_over_computed_and_compared_grid",
    ]:
        scaling_source = (
            "(constrained)"
            if rescale_reward == "min_max_over_computed_grid"
            and reward_constraint < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f} {scaling_source}"
    elif rescale_reward == "only_max_over_computed_and_compared_grid":
        title += f"\nScaling: divided by max of {indiv_scaling_max:.0f}"
    elif rescale_reward == "min_max_over_compared_grid":
        scaling_source = (
            "(constrained)"
            if reward_constraint_for_comparison_grid < 1000
            else "(unconstrained)"
        )
        title += f"\nScaling: min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f} {scaling_source}"
    elif rescale_reward == False:
        title += (
            f"\nNo rescaling. min={indiv_scaling_min:.0f}, max={indiv_scaling_max:.0f}"
        )

    plt.title(title)
    plt.legend()
    plt.show()


def make_four_individual_3d_reward_plots(
    rescale_reward=False, price_constraint="unconstrained", reward_constraint=1e6
):
    print("Unconstrained setting")
    create_individual_3d_reward_plot(
        "unconstrained",
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Unconstrained prices, but constrained rewards:")
    create_individual_3d_reward_plot(
        "unconstrained",
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices, but unconstrained rewards: (at beginning of episode)")
    create_individual_3d_reward_plot(
        price_constraint,
        1e6,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )
    print("Constrained prices and rewards")
    create_individual_3d_reward_plot(
        price_constraint,
        reward_constraint,
        rescale_reward=rescale_reward,
        reward_constraint_for_comparison_grid=reward_constraint,
    )


make_four_individual_3d_reward_plots(
    rescale_reward=False,  # "min_max_over_compared_grid",
    price_constraint="455",
    reward_constraint=455,
)

# %%
