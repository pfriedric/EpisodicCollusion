# %%
import pickle
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import glob
from agents.dqn.dqn import DQN, make_DQN_agent
from agents.dqn.networks import make_dqn_marketenv_network
from agents.ppo.ppo import PPO, make_agent
from agents.ppo.networks import make_marketenv_network
from environment.market_env import MarketEnv, EnvParams
import optax

from plotting_utils import display_single_plot

### Alter this for different runs. Options "DQN", "PPO", "compPPO", "unconstDQN"
save_dir = "exp/DQN"


########################################################
plot_new = True
inv_scaling = "dynamic"  # "dynamic"  # the fraction of inventory agents have in the observation used for plotting. is a float in [0,1] or "dynamic" (1-t/T)
run_name = save_dir.split("/")[-1]
plot_dir = os.path.join(save_dir, "reaction_surface_plot")

plot_pattern = os.path.join(plot_dir, f"reaction_surface_plot_t_*_inv_{inv_scaling}.png")
# check if such plots exist in plot_dir (return True only if one or more plots with that specific pattern is there, using glob)
if not glob.glob(plot_pattern):
    plot_new = True
    print(f"No plots found for inv={inv_scaling}, plotting new ones")

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
        elif len(log_data) == 3:
            log_data, eval_log_data, forced_deviation_log_data = log_data
            forced_deviation_log_data_unstacked = [
                jax.tree.map(lambda v: v[:, :, i, ...], forced_deviation_log_data)
                for i in range(args["num_inner_steps"])
            ]
        else:
            raise ValueError("log_data tuple has an unexpected length")
    exists_log_data = True

    plot_dir = glob.glob(f"{save_dir}/reaction_surface_plot")
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

if args["agent_default"] == "PPO":
    print(f"PPO setup:")
    print(
        f"  LR {args['ppo_default']['learning_rate']} {'annealing to 0' if args['ppo_default']['lr_scheduling'] else 'fixed'} {'over ' + str(args['ppo_default']['lr_anneal_duration'] * 100) + '% of run' if args['ppo_default']['lr_scheduling'] else '(flat)'}"
    )
    print(f"  discount: {args['ppo_default']['gamma']}")
    print(f"  hidden sizes: {args['ppo_default']['hidden_sizes']}")
    if args["ppo_default"]["anneal_entropy"] == "linear":
        print(
            f"  entropy coeff {args['ppo_default']['entropy_coeff_start']} annealing to {args['ppo_default']['entropy_coeff_end']} over {args['ppo_default']['entropy_anneal_duration'] * 100:.2f}% of run"
        )
    if args["ppo_default"]["anneal_entropy"] == "exponential":
        print(
            f"  entropy coeff annealing from {args['ppo_default']['entropy_coeff_start']} to {args['ppo_default']['entropy_coeff_end']}, hitting the min at {args['ppo_default']['entropy_anneal_duration'] * 100:.2f}% of run"
        )
    else:
        print(f"  entropy coeff fixed at {args['ppo_default']['entropy_coeff_start']}")
    print(
        f"  num minibatches: {args['ppo_default']['num_minibatches']}, epochs: {args['ppo_default']['num_epochs']}"
    )


## let's create a new agent 1
# first define an env
env = MarketEnv(
    num_agents=args["num_players"],
    num_actions=args["num_prices"],
    time_horizon=args["time_horizon"],
)
env_params = EnvParams(
    time_horizon=args["time_horizon"],
    min_price=args["min_price"],
    max_price=args["max_price"],
    num_prices=args["num_prices"],
    possible_prices=jnp.array(args["possible_prices"]),
    qualities=jnp.array(args["qualities"]),
    marginal_costs=jnp.array(args["marginal_costs"]),
    horizontal_diff=args["horizontal_diff"],
    demand_scaling_factor=args["demand_scaling_factor"],
    initial_inventories=jnp.array(args["initial_inventories"]),
    initial_prices=jnp.array(args["initial_prices"]),
    initial_actions=jnp.array(args["initial_actions"]),
)

obs_shape = jax.tree_util.tree_map(lambda x: x.shape, env.observation_space(env_params))
obs_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, env.observation_space(env_params))


if args["agent_default"] == "DQN":
    ### Looking into agent
    dqn_network = make_dqn_marketenv_network(
        num_actions=num_actions, hidden_sizes=args["dqn_default"]["hidden_sizes"]
    )
    dummy_learning_rate = args["dqn_default"]["learning_rate"]
    scale = optax.inject_hyperparams(optax.scale)(step_size=-dummy_learning_rate)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args["dqn_default"]["max_gradient_norm"]),
        optax.scale_by_adam(eps=args["dqn_default"]["adam_epsilon"]),
        scale,
    )

    loaded_agent_1 = DQN.load_state(
        f"{save_dir}/agent_1_state.pkl", network=dqn_network, optimizer=optimizer
    )
    loaded_agent_2 = DQN.load_state(
        f"{save_dir}/agent_2_state.pkl", network=dqn_network, optimizer=optimizer
    )
    # these all have shape with leading dims (seeds, cfg=1). so e.g., params!

    default_player_args = args.get("dqn_default")
    player_key_1 = f"dqn{loaded_agent_1.player_id}"
    player_args_1 = args.get(player_key_1, default_player_args)
    player_key_2 = f"dqn{loaded_agent_2.player_id}"
    player_args_2 = args.get(player_key_2, default_player_args)

    agent1 = make_DQN_agent(
        args,
        player_args_1,
        obs_spec=obs_shape,
        obs_dtypes=obs_dtypes,
        action_spec=num_actions,
        seed=seeds[0],
        num_iterations=num_iters,
        player_id=loaded_agent_1.player_id,
    )

    agent2 = make_DQN_agent(
        args,
        player_args_2,
        obs_spec=obs_shape,
        obs_dtypes=obs_dtypes,
        action_spec=num_actions,
        seed=seeds[0],
        num_iterations=num_iters,
        player_id=loaded_agent_2.player_id,
    )

    agent1._state = loaded_agent_1._state
    agent1._mem = loaded_agent_1._mem
    print(f"agent 1 mem shape: {agent1._mem.hidden.shape}")
    print(
        f"agent 1 params shape: {agent1._state.params['market_q_network/~/body/~/linear_0']['b'].shape}"
    )
    agent1_params = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, axis=1), agent1._state.params
    )  # squeezing out config dim
    agent1_mem = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), agent1._mem)
    agent1._eval_policy = jax.jit(agent1._eval_policy)

    # def eval_policy_single_seed_agent1_PPO(params_i, state, obs_i, mem_i):
    #     single_state = state._replace(params=params_i)
    #     return agent1._eval_policy(single_state, obs_i, mem_i)

    # agent1_eval_vmap = jax.vmap(
    #     eval_policy_single_seed_agent1_PPO, in_axes=(0, None, 0, 0)
    # )

    agent2._state = loaded_agent_2._state
    agent2._mem = loaded_agent_2._mem
    print(f"agent 2 mem shape: {agent2._mem.hidden.shape}")
    print(
        f"agent 1 params shape: {agent2._state.params['market_q_network/~/body/~/linear_0']['b'].shape}"
    )
    agent2_params = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, axis=(1, 2)), agent2._state.params
    )  # squeeze out config dim
    agent2_random_key = jnp.squeeze(agent2._state.random_key, axis=(1, 2))
    agent2._state = agent2._state._replace(random_key=agent2_random_key)
    agent2_mem = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), agent2._mem)
    agent2._eval_policy = jax.jit(agent2._eval_policy)

if args["agent_default"] == "PPO":
    ppo_network = make_marketenv_network(
        num_actions=num_actions,
        separate=args["ppo_default"]["separate"],
        hidden_sizes=args["ppo_default"]["hidden_sizes"],
    )
    scale = optax.inject_hyperparams(optax.scale)(step_size=-args["ppo_default"]["learning_rate"])
    optimizer = optax.chain(
        optax.clip_by_global_norm(args["ppo_default"]["max_gradient_norm"]),
        optax.scale_by_adam(eps=args["ppo_default"]["adam_epsilon"]),
        scale,
    )

    loaded_agent_1 = PPO.load_state(
        f"{save_dir}/agent_1_state.pkl", network=ppo_network, optimizer=optimizer
    )
    loaded_agent_2 = PPO.load_state(
        f"{save_dir}/agent_2_state.pkl", network=ppo_network, optimizer=optimizer
    )
    # these all have shape with leading dims (seeds, cfg=1). so e.g., params!

    loaded_agent_1_TS = loaded_agent_1._state  # (seeds, cfg=1)
    loaded_agent_2_TS = loaded_agent_2._state  # (seeds, cfg=1)

    default_player_args = args.get("ppo_default")
    player_key_1 = f"ppo{loaded_agent_1.player_id}"
    player_args_1 = args.get(player_key_1, default_player_args)
    player_key_2 = f"ppo{loaded_agent_2.player_id}"
    player_args_2 = args.get(player_key_2, default_player_args)

    agent1 = make_agent(
        args,
        player_args_1,
        obs_spec=obs_shape,
        # obs_dtypes=obs_dtypes,
        action_spec=num_actions,
        seed=seeds[0],
        num_iterations=num_iters,
        player_id=loaded_agent_1.player_id,
        tabular=False,
    )

    agent2 = make_agent(
        args,
        player_args_2,
        obs_spec=obs_shape,
        # obs_dtypes=obs_dtypes,
        action_spec=num_actions,
        seed=seeds[0],
        num_iterations=num_iters,
        player_id=loaded_agent_2.player_id,
        tabular=False,
    )

    agent1._state = loaded_agent_1._state
    agent1._mem = loaded_agent_1._mem
    agent1_params = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), agent1._state.params)
    agent1_random_key = jnp.squeeze(agent1._state.random_key, axis=1)
    agent1._state = agent1._state._replace(random_key=agent1_random_key)
    agent1_mem = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), agent1._mem)
    agent1._eval_policy = jax.jit(agent1._eval_policy)

    # def eval_policy_single_seed_agent1(params_i, random_key_i, state, obs_i, mem_i):
    #     single_state = state._replace(params=params_i, random_key=random_key_i)
    #     return agent1._eval_policy(single_state, obs_i, mem_i)

    # agent1_eval_vmap = jax.vmap(eval_policy_single_seed_agent1, in_axes=(0, None, 0, 0))

    agent2._state = loaded_agent_2._state
    agent2._mem = loaded_agent_2._mem
    agent2_params = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, axis=(1, 2)), agent2._state.params
    )
    agent2_random_key = jnp.squeeze(agent2._state.random_key, axis=(1, 2))
    agent2._state = agent2._state._replace(random_key=agent2_random_key)
    agent2_mem = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), agent2._mem)
    agent2._eval_policy = jax.jit(agent2._eval_policy)


def plot_agent_reactions(input_time, inv_scaling, vmin, vmax, agent_id):
    # Number of actions available
    num_actions = len(args["possible_prices"])

    # Create action grids
    action_space = jnp.arange(num_actions)
    action1_grid, action2_grid = jnp.meshgrid(
        action_space, action_space, indexing="ij"
    )  # Shape: (num_actions, num_actions)

    # Initialize array to hold mean prices
    mean_prices_grid = np.zeros((num_actions, num_actions))
    mean_actions_grid = np.zeros((num_actions, num_actions))

    possible_prices = jnp.array(args["possible_prices"])

    # Fixed inventories
    inventories_fixed = env_params.initial_inventories.astype(np.int32)  # Shape: (2,)
    inventories_fixed = inventories_fixed * inv_scaling
    # Time step
    t_fixed = jnp.full((1,), input_time, dtype=np.int32)  # Shape: (1,)

    # Prepare the memory state, which is the same for all seeds
    mem = agent1_mem  # Should have appropriate shape

    # Prepare the agent state
    state = agent1._state  # The agent state, with parameters

    @jax.jit
    def eval_action_pair_ag1(action1, action2, agent_state):
        # Prepare last_actions
        last_actions = jnp.array([[action1, action2]], dtype=np.int32)  # Shape: (1, 2)
        last_actions = jnp.tile(last_actions, (num_seeds, 1))  # Shape: (num_seeds, 2)

        # Compute last_prices based on last_actions
        last_prices = possible_prices[last_actions]  # Shape: (num_seeds, 2)

        # Prepare inventories
        inventories = jnp.tile(inventories_fixed, (num_seeds, 1))  # Shape: (num_seeds, 2)

        # Prepare time steps
        t = jnp.tile(t_fixed, (num_seeds, 1))  # Shape: (num_seeds, 1)

        # Create observations dictionary
        observations = {
            "inventories": inventories,
            "last_actions": last_actions,
            "last_prices": last_prices,
            "t": t,
        }

        # Convert observations to JAX arrays
        observations = jax.tree_util.tree_map(jnp.array, observations)

        # Define a function to evaluate the policy for a single seed
        if args["agent_default"] == "PPO":

            def eval_policy_single(params_i, random_key_i, observation_i, mem_i):
                state_i = agent_state._replace(params=params_i, random_key=random_key_i)
                return agent1._eval_policy(state_i, observation_i, mem_i)

                # Vectorize the evaluation over seeds

            actions, _, _, _ = jax.vmap(eval_policy_single)(
                agent1_params, agent1_random_key, observations, mem
            )

        elif args["agent_default"] == "DQN":

            def eval_policy_single(params_i, observation_i, mem_i):
                state_i = agent_state._replace(params=params_i)
                return agent1._eval_policy(state_i, observation_i, mem_i)

            # Vectorize the evaluation over seeds
            actions, _, _, _ = jax.vmap(eval_policy_single)(agent1_params, observations, mem)

        # Map actions to prices
        prices = possible_prices[actions]  # Shape: (num_seeds,)

        # Compute mean price
        mean_price = jnp.mean(prices)
        mean_action = jnp.mean(actions)

        return mean_price, mean_action

    @jax.jit
    def eval_action_pair_ag2(action1, action2, agent_state):
        # Prepare last_actions
        last_actions = jnp.array([[action1, action2]], dtype=np.int32)  # Shape: (1, 2)
        last_actions = jnp.tile(last_actions, (num_seeds, 1))  # Shape: (num_seeds, 2)

        # Compute last_prices based on last_actions
        last_prices = possible_prices[last_actions]  # Shape: (num_seeds, 2)

        # Prepare inventories
        inventories = jnp.tile(inventories_fixed, (num_seeds, 1))  # Shape: (num_seeds, 2)

        # Prepare time steps
        t = jnp.tile(t_fixed, (num_seeds, 1))  # Shape: (num_seeds, 1)

        # Create observations dictionary
        observations = {
            "inventories": inventories,
            "last_actions": last_actions,
            "last_prices": last_prices,
            "t": t,
        }

        # Convert observations to JAX arrays
        observations = jax.tree_util.tree_map(jnp.array, observations)

        # Define a function to evaluate the policy for a single seed
        if args["agent_default"] == "PPO":

            def eval_policy_single(params_i, random_key_i, observation_i, mem_i):
                state_i = agent_state._replace(params=params_i, random_key=random_key_i)
                return agent2._eval_policy(state_i, observation_i, mem_i)

                # Vectorize the evaluation over seeds

            actions, _, _, _ = jax.vmap(eval_policy_single)(
                agent2_params, agent2_random_key, observations, mem
            )
        elif args["agent_default"] == "DQN":

            def eval_policy_single(params_i, observation_i, mem_i):
                state_i = agent_state._replace(params=params_i)
                return agent2._eval_policy(state_i, observation_i, mem_i)

            # Vectorize the evaluation over seeds
            actions, _, _, _ = jax.vmap(eval_policy_single)(agent2_params, observations, mem)

        # Map actions to prices
        prices = possible_prices[actions]  # Shape: (num_seeds,)

        # Compute mean price
        mean_price = jnp.mean(prices)
        mean_action = jnp.mean(actions)

        return mean_price, mean_action

    if agent_id == 1:
        print(f"agent_id is {agent_id}")
        # Loop over action pairs
        for i in range(num_actions):
            for j in range(num_actions):
                action1 = action1_grid[i, j]
                action2 = action2_grid[i, j]

                # Compute the mean reaction price for this action pair
                mean_price, mean_action = eval_action_pair_ag1(action1, action2, agent1._state)

                # Store the mean price in the grid
                mean_prices_grid[i, j] = mean_price
                mean_actions_grid[i, j] = mean_action
    elif agent_id == 2:
        print(f"agent_id is {agent_id}")
        # Loop over action pairs
        for i in range(num_actions):
            for j in range(num_actions):
                action1 = action1_grid[i, j]
                action2 = action2_grid[i, j]

                # Compute the mean reaction price for this action pair
                mean_price, mean_action = eval_action_pair_ag2(action1, action2, agent2._state)

                # Store the mean price in the grid
                mean_prices_grid[i, j] = mean_price
                mean_actions_grid[i, j] = mean_action

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Prepare grid data for plotting
    X = action1_grid
    Y = action2_grid
    Z = mean_actions_grid

    # Create the surface plot with a more subdued color scheme
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Agent 1 Previous Action")
    ax.set_ylabel("Agent 2 Previous Action")
    ax.set_zlabel(f"Mean Reaction of Agent {agent_id}")
    title = f"Agent {agent_id} Reaction at Time {input_time}"
    if args["env_id"] == "MarketEnv-v1":
        title += f", inventories both at {inv_scaling * 100:.0f}% of initial value"
    ax.set_title(title)

    # Invert the x-axis to mirror it
    ax.invert_xaxis()

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label("Mean Reaction Value")

    # Adjust the viewing angle for better visibility
    # ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    # plt.show()
    plt.close()

    return fig, title, Z


def plot_main(eval_metrics, forced_deviation_metrics_unstacked, x_axis, agent_id):
    plot_dir = os.path.join(save_dir, "reaction_surface_plot")
    os.makedirs(plot_dir, exist_ok=True)

    # First, determine the global min and max values
    global_min = float("inf")
    global_max = float("-inf")

    if isinstance(inv_scaling, (float, int)):
        for t in range(20):
            _, _, mean_actions_grid = plot_agent_reactions(
                t, inv_scaling=inv_scaling, vmin=0, vmax=1, agent_id=agent_id
            )
            global_min = min(global_min, np.min(mean_actions_grid))
            global_max = max(global_max, np.max(mean_actions_grid))

        for t in range(20):
            fig, title, _ = plot_agent_reactions(
                t,
                inv_scaling=inv_scaling,
                vmin=global_min,
                vmax=global_max,
                agent_id=agent_id,
            )
            fig.savefig(
                os.path.join(plot_dir, f"reaction_surface_plot_t_{t}_inv_{inv_scaling}.png")
            )
            print(f"Saved plot for t={t}, inv_scaling={inv_scaling}")

    elif inv_scaling == "dynamic":
        for t in range(20):
            inv_scaling_num = np.round(1 - t / 20, 2)
            _, _, mean_actions_grid = plot_agent_reactions(
                t, inv_scaling=inv_scaling_num, vmin=0, vmax=1, agent_id=agent_id
            )
            global_min = min(global_min, np.min(mean_actions_grid))
            global_max = max(global_max, np.max(mean_actions_grid))

        for t in range(20):
            inv_scaling_num = np.round(1 - t / 20, 2)
            fig, title, _ = plot_agent_reactions(
                t,
                inv_scaling=inv_scaling_num,
                vmin=global_min,
                vmax=global_max,
                agent_id=agent_id,
            )
            print(f"Saving plot for t={t}, inv_scaling={inv_scaling_num}")
            fig.savefig(os.path.join(plot_dir, f"reaction_surface_plot_t_{t}_inv_dynamic.png"))
            plt.close(fig)
    else:
        print(f"inv_scaling was {inv_scaling}, not dynamic or int")
    print(f"global min was {global_min} and global max was {global_max}")


def display_plots(plot_pattern):
    # sort by t
    for plot_path in sorted(
        glob.glob(plot_pattern), key=lambda x: int(x.split("_t_")[1].split("_inv_")[0])
    ):
        display_single_plot(plot_path)


def load_and_display_plots():
    plot_dir = os.path.join(save_dir, "reaction_surface_plot")

    plot_pattern = os.path.join(plot_dir, f"reaction_surface_plot_t_*_inv_{inv_scaling}.png")
    display_plots(plot_pattern)


if plot_new and exists_log_data:
    agent_id = 2
    # plot_main(eval_log_data, forced_deviation_log_data_unstacked, x_axis, agent_id)

# load_and_display_plots()

# %%
paper_plot_dir = os.path.join(save_dir, "paper_plots")
os.makedirs(paper_plot_dir, exist_ok=True)


def plot_paper_fig(eval_metrics, forced_deviation_metrics_unstacked, x_axis, agent_id):
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

    # Create a 1x3 subplot for the paper figure
    # fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    fig = plt.figure(figsize=(18, 6.5), dpi=300, facecolor="white")
    axs = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    # fig.suptitle("Learned Best Response Surface at Different Time Steps", fontsize=16)

    # First, determine the global min and max values
    global_min = float("inf")
    global_max = float("-inf")
    plotting_times = np.array([1, 10, 18])

    # Calculate global min and max
    for t in plotting_times:
        inv_scaling_num = np.round(1 - t / 20, 2) if inv_scaling == "dynamic" else inv_scaling
        _, _, mean_actions_grid = plot_agent_reactions(
            t, inv_scaling=inv_scaling_num, vmin=0, vmax=1, agent_id=agent_id
        )
        global_min = min(global_min, np.min(mean_actions_grid))
        global_max = max(global_max, np.max(mean_actions_grid))

    if args["agent_default"] == "PPO":
        # PPO spans such a wide range that left and right are solid red/blue otherwise
        global_min = None
        global_max = None
    # Plot for each time step
    for i, t in enumerate(plotting_times):
        inv_scaling_num = np.round(1 - t / 20, 2) if inv_scaling == "dynamic" else inv_scaling
        _, _, Z = plot_agent_reactions(
            t,
            inv_scaling=inv_scaling_num,
            vmin=global_min,
            vmax=global_max,
            agent_id=agent_id,
        )
        X, Y = np.meshgrid(np.arange(num_actions), np.arange(num_actions), indexing="ij")
        surf = axs[i].plot_surface(
            X, Y, Z, cmap="coolwarm", edgecolor="none", vmin=global_min, vmax=global_max
        )

        if args["agent_default"] == "DQN":
            axs[i].set_xlabel("Agent 1", fontsize=15)  # , labelpad=10)
            axs[i].set_ylabel("Agent 2", fontsize=15)  # , labelpad=10)
            axs[i].set_zlabel(
                f"Reaction of Agent {agent_id}", fontsize=15, labelpad=5
            )  # , labelpad=10)
        if args["agent_default"] == "PPO":
            axs[i].set_xlabel("Agent 1", fontsize=15)  # , labelpad=10)
            axs[i].set_ylabel("Agent 2", fontsize=15)  # , labelpad=10)
            axs[i].set_zlabel(
                f"Reaction of Agent {agent_id}", fontsize=15, labelpad=5
            )  # , labelpad=10)

        axs[i].set_title(f"Timestep $t = {t}$", fontsize=16)
        axs[i].invert_xaxis()

        # Adjust the viewing angle for better visibility
        if args["agent_default"] == "PPO":
            if agent_id == 1:
                axs[i].view_init(elev=25, azim=358)
                # axs[i].view_init(elev=25, azim=358)

            # Add a colorbar for each subplot
            cbar = fig.colorbar(surf, ax=axs[i], shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label("")

    if args["agent_default"] == "DQN":
        # Adjust the layout to make space for the colorbar
        plt.subplots_adjust(left=0.05, right=0.9, top=0.86, bottom=0.17, wspace=0.05)

        # Add a single colorbar to the right of the subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(surf, cax=cbar_ax)
        # cbar.set_label("Mean Reaction Value")

    if args["agent_default"] == "PPO":
        # Adjust subplot layout for PPO
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.1, hspace=0.2)
        for ax in axs:
            ax.title.set_position([0.5, 1.05])

    # Adjust layout and save
    # plt.tight_layout()
    fig.savefig(
        os.path.join(paper_plot_dir, f"fig3b_{run_name}_reaction_surface_{agent_id}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"Saved paper figure: fig3b_{run_name}_reaction_surface_{agent_id}.png")


agent_ids = [1, 2]
for agent_id in agent_ids:
    plot_paper_fig(eval_log_data, forced_deviation_log_data_unstacked, x_axis, agent_id)
    display_single_plot(
        os.path.join(paper_plot_dir, f"fig3b_{run_name}_reaction_surface_{agent_id}.png")
    )

# %%
