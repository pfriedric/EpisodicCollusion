from functools import partial
from typing import Any, NamedTuple
from jaxtyping import Array
from optax import sgd
from pyparsing import sgl_quoted_string
from environment.market_env import EnvState
import jax
import jax.numpy as jnp


def marketenv_stats(
    state: EnvState,
    traj1: NamedTuple,
    traj2: NamedTuple,
    env_traj: NamedTuple,
    info_traj: NamedTuple,
    num_envs: int,
    num_opps: int,
    num_outer: int,
    initial_inventories: Array,
    competitive_profits_onestep: Array,
    collusive_profits_onestep: Array,
) -> dict:
    """Compute statistics for the MarketEnv. Acts on data from one rollout."""
    # env_traj and info_traj: [n_o, n_i, n_opp, n_env, idx_ag]
    # traj1 and traj2: [n_outer, n_inner, n_opp, n_env]

    ## figures as mean over (n_e envs * n_outer episodes * n_opps opponents) episodes
    num_eps = num_envs * num_opps * num_outer
    # calc profit of competitive or collusive price
    episode_length = traj1.rewards.shape[1]
    competitive_profits_episode = competitive_profits_onestep * episode_length
    collusive_profits_episode = collusive_profits_onestep * episode_length

    ## episodic collusion index, for each parallel env:
    # sum rewards over time axis (ax=1):
    ag1_episodic_profits = traj1.unnormalized_rewards.sum(
        axis=1
    )  # -> [n_outer, n_opp, n_env]
    ag2_episodic_profits = traj2.unnormalized_rewards.sum(axis=1)
    # collusion index for each episode, should have shape [n_outer, n_opp, n_env]
    ag1_ep_profit_gain_eps = (ag1_episodic_profits - competitive_profits_episode[0]) / (
        collusive_profits_episode[0] - competitive_profits_episode[0]
    )
    ag2_ep_profit_gain_eps = (ag2_episodic_profits - competitive_profits_episode[1]) / (
        collusive_profits_episode[1] - competitive_profits_episode[1]
    )

    # mean over n_outer, n_opp first (so per-env figures)
    ag1_ep_profit_gain_eps_mean = ag1_ep_profit_gain_eps.mean(axis=(0, 1))
    ag2_ep_profit_gain_eps_mean = ag2_ep_profit_gain_eps.mean(axis=(0, 1))
    ag1_ep_profit_gain_var = ag1_ep_profit_gain_eps_mean.var()
    ag2_ep_profit_gain_var = ag2_ep_profit_gain_eps_mean.var()

    # mean over all parallel envs:
    ag1_ep_profit_gain_mean = ag1_ep_profit_gain_eps.mean()
    ag2_ep_profit_gain_mean = ag2_ep_profit_gain_eps.mean()
    # geometric mean of collusion indices of both agents
    collusion_index_geometricmean_all_envs_all_opps = jnp.sqrt(
        ag1_ep_profit_gain_mean * ag2_ep_profit_gain_mean
    )

    # total episode profits per env (mean over outer, opps)
    total_profit1_per_env = ag1_episodic_profits.mean(axis=(0, 1))
    total_profit2_per_env = ag2_episodic_profits.mean(axis=(0, 1))
    total_profit1_mean = total_profit1_per_env.mean()
    total_profit2_mean = total_profit2_per_env.mean()
    total_profit1_var = total_profit1_per_env.var()
    total_profit2_var = total_profit2_per_env.var()

    # total episode rewards (mean over envs, outer, opps).
    total_rewards1_all_envs_eps_opps = traj1.rewards.sum() / num_eps
    total_rewards2_all_envs_eps_opps = traj2.rewards.sum() / num_eps

    # mean per-timestep rewards
    rewards1_all_envs_eps_opps = traj1.rewards.mean()
    rewards2_all_envs_eps_opps = traj2.rewards.mean()

    # mean per-timestep unnormalized rewards
    rewards1_all_envs_eps_opps_unnormalized = traj1.unnormalized_rewards.mean()
    rewards2_all_envs_eps_opps_unnormalized = traj2.unnormalized_rewards.mean()

    # mean per-timestep actions (over outer, inner, opps, envs)
    actions1_all_envs_eps_opps = traj1.actions.mean()
    actions2_all_envs_eps_opps = traj2.actions.mean()

    # actions per env (over outer, inner, opps)
    actions1_all_eps_opps = traj1.actions.mean(axis=(0, 1, 2))
    actions2_all_eps_opps = traj2.actions.mean(axis=(0, 1, 2))
    # these are the same as actionsi_all_envs_eps_opps.mean()
    # actions1_mean = actions1_all_eps_opps.mean()
    # actions2_mean = actions2_all_eps_opps.mean()
    actions1_var = actions1_all_eps_opps.var()
    actions2_var = actions2_all_eps_opps.var()

    # DQN: greedy actions; PPO: behaviour values per env (over outer, inner, opps)
    greedy_actions1_all_eps_opps = traj1.behavior_values.mean(axis=(0, 1, 2))
    greedy_actions2_all_eps_opps = traj2.behavior_values.mean(axis=(0, 1, 2))
    greedy_actions1_mean = greedy_actions1_all_eps_opps.mean()
    greedy_actions2_mean = greedy_actions2_all_eps_opps.mean()
    greedy_actions1_var = greedy_actions1_all_eps_opps.var()
    greedy_actions2_var = greedy_actions2_all_eps_opps.var()

    # mean per-timestep prices within episodes
    prices1_all_envs_eps_opps = env_traj.env_state.env_state.last_prices[
        :, :, :, :, 0
    ].mean()
    prices2_all_envs_eps_opps = env_traj.env_state.env_state.last_prices[
        :, :, :, :, 1
    ].mean()
    prices1_all_eps_opps = env_traj.env_state.env_state.last_prices[:, :, :, :, 0].mean(
        axis=(0, 1, 2)
    )
    prices2_all_eps_opps = env_traj.env_state.env_state.last_prices[:, :, :, :, 1].mean(
        axis=(0, 1, 2)
    )
    prices1_var = prices1_all_eps_opps.var()
    prices2_var = prices2_all_eps_opps.var()

    # mean per-timestep demands
    demands1_all_envs_eps_opps = info_traj["demands"][:, :, :, :, 0].mean()
    demands2_all_envs_eps_opps = info_traj["demands"][:, :, :, :, 1].mean()

    # mean per-timestep quantities sold
    quantities_sold1_all_envs_eps_opps = info_traj["quantity_sold"][
        :, :, :, :, 0
    ].mean()
    quantities_sold2_all_envs_eps_opps = info_traj["quantity_sold"][
        :, :, :, :, 1
    ].mean()

    ## mean inventory left at end of episode [:, n_inner=-1, :, :, agent_idx=0], in percent
    inventories1_absolute_means_per_env = env_traj.env_state.env_state.inventories[
        :, -1, :, :, 0
    ].mean(axis=(0, 1))
    inventories2_absolute_means_per_env = env_traj.env_state.env_state.inventories[
        :, -1, :, :, 1
    ].mean(axis=(0, 1))

    inventories1_means_per_env = (
        inventories1_absolute_means_per_env / initial_inventories[0]
    )
    inventories2_means_per_env = (
        inventories2_absolute_means_per_env / initial_inventories[1]
    )
    # inventories1_absolute_mean = inventories1_absolute_means_per_env.mean()
    # inventories2_absolute_mean = inventories2_absolute_means_per_env.mean()
    # inventories1_absolute_var = inventories1_absolute_means_per_env.var()
    # inventories2_absolute_var = inventories2_absolute_means_per_env.var()
    inventories1_mean = inventories1_means_per_env.mean()
    inventories2_mean = inventories2_means_per_env.mean()
    inventories1_var = inventories1_means_per_env.var()
    inventories2_var = inventories2_means_per_env.var()

    # inventories1_all_envs_eps_opps = (
    #     env_traj.env_state.env_state.inventories[:, -1, :, :, 0].mean()
    #     / initial_inventories[0]
    # )
    # inventories2_all_envs_eps_opps = (
    #     env_traj.env_state.env_state.inventories[:, -1, :, :, 1].mean()
    #     / initial_inventories[1]
    # )

    # mean of 'reward normalization mean' (over n_out, n_in)
    rewards1_all_envs_eps_opps_mean = env_traj.mean[:, :, 0].mean()
    rewards2_all_envs_eps_opps_mean = env_traj.mean[:, :, 1].mean()
    # mean of 'reward normalization var'
    rewards1_all_envs_eps_opps_var = env_traj.var[:, :, 0].mean()
    rewards2_all_envs_eps_opps_var = env_traj.var[:, :, 1].mean()

    ############################################################

    ### figures for a single (arbitrary) episode: last outer ep., 0th env., 0th opponent
    # [n_outer=last, n_inner=all, n_opp=0, n_envs=0]
    # total episode reward
    # total_profit1 = traj1.unnormalized_rewards[-1, :, 0, 0].sum()
    # total_profit2 = traj2.unnormalized_rewards[-1, :, 0, 0].sum()

    # # collusion index for this episode
    # ag1_collusion_index_singleep = (total_profit1 - competitive_profits_episode[0]) / (
    #     collusive_profits_episode[0] - competitive_profits_episode[0]
    # )
    # ag2_collusion_index_singleep = (total_profit2 - competitive_profits_episode[1]) / (
    #     collusive_profits_episode[1] - competitive_profits_episode[1]
    # )
    # collusion_index_geometricmean_singleep = jnp.sqrt(
    #     ag1_collusion_index_singleep * ag2_collusion_index_singleep
    # )

    # # mean per-timestep reward
    # rewards1 = traj1.rewards[-1, :, 0, 0].mean()
    # rewards2 = traj2.rewards[-1, :, 0, 0].mean()

    # # mean per-timestep actions
    # actions1 = traj1.actions[-1, :, 0, 0].mean()
    # actions2 = traj2.actions[-1, :, 0, 0].mean()

    # # mean per-timestep prices
    # prices1 = env_traj.env_state.env_state.last_prices[-1, :, 0, 0, 0].mean()
    # prices2 = env_traj.env_state.env_state.last_prices[-1, :, 0, 0, 1].mean()

    # # mean per-timestep demands
    # demands1 = info_traj["demands"][-1, :, 0, 0, 0].mean()
    # demands2 = info_traj["demands"][-1, :, 0, 0, 1].mean()

    # # mean per-timestep quantities sold
    # quantities_sold1 = info_traj["quantity_sold"][-1, :, 0, 0, 0].mean()
    # quantities_sold2 = info_traj["quantity_sold"][-1, :, 0, 0, 1].mean()

    # # inventory left at end of episode
    # inventories1 = (
    #     env_traj.env_state.env_state.inventories[-1, -1, 0, 0, 0].mean()
    #     / initial_inventories[0]
    # )
    # inventories2 = (
    #     env_traj.env_state.env_state.inventories[-1, -1, 0, 0, 1].mean()
    #     / initial_inventories[1]
    # )

    return {
        # figures averaged over (n_e envs * n_outer episodes * n_opps opponents) episodes
        "train/collusion_index/mean_player_1": ag1_ep_profit_gain_mean,
        "train/collusion_index/mean_player_2": ag2_ep_profit_gain_mean,
        ## commented out for perf, wandb uses this
        # "train/collusion_index/0th_episode_player_1": ag1_collusion_index_singleep,
        # "train/collusion_index/0th_episode_player_2": ag2_collusion_index_singleep,
        # "train/collusion_index/geometric mean_all_eps": collusion_index_geometricmean_all_envs_all_opps,
        # "train/collusion_index/geometric mean_single_episode": collusion_index_geometricmean_singleep,
        # "train/all_envs/rewards/total_player_1": total_rewards1_all_envs_eps_opps,
        # "train/all_envs/rewards/total_player_2": total_rewards2_all_envs_eps_opps,
        "train/all_envs/rewards/mean_player_1": rewards1_all_envs_eps_opps,
        "train/all_envs/rewards/mean_player_2": rewards2_all_envs_eps_opps,
        ## commented out for perf, wandb uses this
        # "train/all_envs/rewards/mean_player_1_unnormalized": rewards1_all_envs_eps_opps_unnormalized,
        # "train/all_envs/rewards/mean_player_2_unnormalized": rewards2_all_envs_eps_opps_unnormalized,
        "train/all_envs/mean_action/action_player_1": actions1_all_envs_eps_opps,
        "train/all_envs/mean_action/action_player_2": actions2_all_envs_eps_opps,
        "train/all_envs/mean_action/price_player_1": prices1_all_envs_eps_opps,
        "train/all_envs/mean_action/price_player_2": prices2_all_envs_eps_opps,
        ## commented out for perf, wandb uses this
        # "train/all_envs/mean_quantity/demand_player_1": demands1_all_envs_eps_opps,
        # "train/all_envs/mean_quantity/demand_player_2": demands2_all_envs_eps_opps,
        # "train/all_envs/mean_quantity/sold_player_1": quantities_sold1_all_envs_eps_opps,
        # "train/all_envs/mean_quantity/sold_player_2": quantities_sold2_all_envs_eps_opps,
        "train/all_envs/mean_quantity/inv_player_1": inventories1_mean,
        "train/all_envs/mean_quantity/inv_player_2": inventories2_mean,
        ## commented out bc not using reward norm via wrapper
        # "train/all_envs/reward_normalization/mean_player_1": rewards1_all_envs_eps_opps_mean,
        # "train/all_envs/reward_normalization/mean_player_2": rewards2_all_envs_eps_opps_mean,
        # "train/all_envs/reward_normalization/var_player_1": rewards1_all_envs_eps_opps_var,
        # "train/all_envs/reward_normalization/var_player_2": rewards2_all_envs_eps_opps_var,
        # figures for single episode, arbitrarily choosing last outer ep., 0th env., 0th opponent
        # "train/0th_env/rewards/total_player_1": total_profit1,
        # "train/0th_env/rewards/total_player_2": total_profit2,
        # "train/0th_env/rewards/mean_player_1": rewards1,
        # "train/0th_env/rewards/mean_player_2": rewards2,
        # "train/0th_env/mean_action/action_player_1": actions1,
        # "train/0th_env/mean_action/action_player_2": actions2,
        # "train/0th_env/mean_action/price_player_1": prices1,
        # "train/0th_env/mean_action/price_player_2": prices2,
        # "train/0th_env/mean_quantity/demand_player_1": demands1,
        # "train/0th_env/mean_quantity/demand_player_2": demands2,
        # "train/0th_env/mean_quantity/sold_player_1": quantities_sold1,
        # "train/0th_env/mean_quantity/sold_player_2": quantities_sold2,
        # "train/0th_env/mean_quantity/inv_player_1": inventories1,
        # "train/0th_env/mean_quantity/inv_player_2": inventories2,
        ### for vmap plotting
        "vmap_metrics/total_profit_mean_player_1": total_profit1_mean,
        "vmap_metrics/total_profit_mean_player_2": total_profit2_mean,
        "vmap_metrics/total_profit_var_player_1": total_profit1_var,
        "vmap_metrics/total_profit_var_player_2": total_profit2_var,
        ## these are plain not needed
        # "vmap_metrics/action_mean_player_1": actions1_mean,
        # "vmap_metrics/action_mean_player_2": actions2_mean,
        "vmap_metrics/action_var_player_1": actions1_var,
        "vmap_metrics/action_var_player_2": actions2_var,
        ## commented out for perf
        # "vmap_metrics/price_var_player_1": prices1_var,
        # "vmap_metrics/price_var_player_2": prices2_var,
        "vmap_metrics/inv_var_player_1": inventories1_var,
        "vmap_metrics/inv_var_player_2": inventories2_var,
        "vmap_metrics/collusion_index_var_player_1": ag1_ep_profit_gain_var,
        "vmap_metrics/collusion_index_var_player_2": ag2_ep_profit_gain_var,
        "vmap_metrics/greedy_action_mean_player_1": greedy_actions1_mean,
        "vmap_metrics/greedy_action_mean_player_2": greedy_actions2_mean,
        "vmap_metrics/greedy_action_var_player_1": greedy_actions1_var,
        "vmap_metrics/greedy_action_var_player_2": greedy_actions2_var,
    }


def marketenv_eval_stats(
    env_traj: NamedTuple,
    traj1: NamedTuple,
    traj2: NamedTuple,
    info_traj: NamedTuple,
    initial_inventories: Array,
    competitive_profits_onestep: Array,
    collusive_profits_onestep: Array,
) -> dict:
    """Generate statistics for the MarketEnv. Acts on data from one eval episode.
    Data has leading dims [time_horizon, n_opps, n_envs] but n_opps=n_envs=1.
    Args:
        state: EnvState
        traj1: EvalSample
        traj2: EvalSample
    Returns:
        dict: Dictionary containing the evaluation metrics
    """

    # reward per timestep, unnormalized
    rewards1 = traj1.rewards_unnormalized.squeeze()  # [time_horizon]
    rewards2 = traj2.rewards_unnormalized.squeeze()

    rewards1_rescaled = traj1.rewards_rescaled.squeeze()
    rewards2_rescaled = traj2.rewards_rescaled.squeeze()

    # price per timestep
    prices1 = env_traj.env_state.env_state.last_prices[:, :, :, 0].squeeze()
    prices2 = env_traj.env_state.env_state.last_prices[:, :, :, 1].squeeze()

    # action per timestep
    actions1 = traj1.actions.squeeze()  # .astype(jnp.int32)
    actions2 = traj2.actions.squeeze()  # .astype(jnp.int32)

    # demand per timestep
    demands1 = info_traj["demands"][:, :, :, 0].squeeze()
    demands2 = info_traj["demands"][:, :, :, 1].squeeze()

    # quantity sold per timestep
    quantities_sold1 = info_traj["quantity_sold"][:, :, :, 0].squeeze()
    quantities_sold2 = info_traj["quantity_sold"][:, :, :, 1].squeeze()

    # inventory percentage per timestep [T, n_envs, n_opps]
    inventories1 = (
        env_traj.env_state.env_state.inventories[:, :, :, 0].squeeze()
        / initial_inventories[0]
    )
    inventories2 = (
        env_traj.env_state.env_state.inventories[:, :, :, 1].squeeze()
        / initial_inventories[1]
    )

    # collusion index per timestep
    collusion_index1 = (rewards1 - competitive_profits_onestep[0]) / (
        collusive_profits_onestep[0] - competitive_profits_onestep[0]
    )
    collusion_index2 = (rewards2 - competitive_profits_onestep[1]) / (
        collusive_profits_onestep[1] - competitive_profits_onestep[1]
    )
    extras1 = jax.tree.map(
        lambda x: x.squeeze() if isinstance(x, jnp.ndarray) else x, traj1.extras
    )
    extras2 = jax.tree.map(
        lambda x: x.squeeze() if isinstance(x, jnp.ndarray) else x, traj2.extras
    )
    # used to be qvals_1 = traj1.extras["q_vals"].squeeze()  # [T, num_actions]

    # all returns: [time_horizon] except qvals: [time_horizon, num_actions]
    return {
        "rewards_1": rewards1,
        "rewards_2": rewards2,
        "rewards_rescaled_1": rewards1_rescaled,
        "rewards_rescaled_2": rewards2_rescaled,
        "prices_1": prices1,
        "prices_2": prices2,
        "actions_1": actions1,
        "actions_2": actions2,
        "demands_1": demands1,
        "demands_2": demands2,
        "quantities_sold_1": quantities_sold1,
        "quantities_sold_2": quantities_sold2,
        "inventories_1": inventories1,
        "inventories_2": inventories2,
        "collusion_index_1": collusion_index1,
        "collusion_index_2": collusion_index2,
        "extras_1": extras1,
        "extras_2": extras2,
    }


def losses_ppo(agent) -> dict:
    """Extract PPO losses from agent's logger. Returns a dictionary of agent's metrics."""
    pid = agent.player_id
    sgd_steps = agent._logger.metrics["sgd_steps"]  # this works
    scheduler_steps = agent._logger.metrics["scheduler_steps"]
    # these below (from agent.update()) are traced and throw errors
    loss_total = agent._logger.metrics["loss_total"]  # .block_until_ready()
    loss_policy = agent._logger.metrics["loss_policy"]
    loss_value = agent._logger.metrics["loss_value"]
    loss_entropy = agent._logger.metrics["loss_entropy"]
    entropy_cost = agent._logger.metrics["entropy_cost"]
    learning_rate = agent._logger.metrics["learning_rate"]
    norm_grad = agent._logger.metrics["norm_grad"]
    norm_updates = agent._logger.metrics["norm_updates"]
    mean_advantages = agent._logger.metrics["mean_advantages"]
    var_advantages = agent._logger.metrics["var_advantages"]
    agent_metrics = {
        f"train/ppo_{pid}/sgd_steps": sgd_steps,
        f"train/ppo_{pid}/scheduler_steps": scheduler_steps,
        f"train/ppo_{pid}/total": loss_total,
        f"train/ppo_{pid}/policy": loss_policy,
        f"train/ppo_{pid}/value": loss_value,
        f"train/ppo_{pid}/entropy": loss_entropy,
        f"train/ppo_{pid}/entropy_coefficient": entropy_cost,
        f"train/ppo_{pid}/learning_rate": learning_rate,
        f"train/ppo_{pid}/norm_grad": norm_grad,
        f"train/ppo_{pid}/norm_updates": norm_updates,
        f"train/ppo_{pid}/mean_advantages": mean_advantages,
        f"train/ppo_{pid}/var_advantages": var_advantages,
    }
    return agent_metrics


def losses_dqn(agent) -> tuple[dict, bool]:
    """Extract DQN losses from agent's logger. Returns a dictionary of agent's metrics"""
    pid = agent.player_id
    sgd_steps = agent._logger.metrics["sgd_steps"]
    scheduler_steps = agent._logger.metrics["scheduler_steps"]
    trained = agent._logger.metrics["trained"]
    loss_value = agent._logger.metrics["loss_value"]
    max_q_val = agent._logger.metrics["max_q_val"]
    min_q_val = agent._logger.metrics["min_q_val"]
    mean_q_val = agent._logger.metrics["mean_q_val"]
    mean_chosen_q_val = agent._logger.metrics["mean_chosen_q_val"]
    mean_td_target = agent._logger.metrics["mean_td_target"]
    td_error = agent._logger.metrics["td_error"]
    learning_rate = agent._logger.metrics["learning_rate"]
    norm_grad = agent._logger.metrics["norm_grad"]
    norm_updates = agent._logger.metrics["norm_updates"]
    explo_epsilon = agent._logger.metrics["explo_epsilon"]
    agent_metrics = {
        f"train/dqn_{pid}/sgd_steps": sgd_steps,
        f"train/dqn_{pid}/value": loss_value,
        f"train/dqn_{pid}/max_q_val": max_q_val,
        f"train/dqn_{pid}/min_q_val": min_q_val,
        f"train/dqn_{pid}/mean_q_val": mean_q_val,
        f"train/dqn_{pid}/mean_chosen_q_val": mean_chosen_q_val,
        f"train/dqn_{pid}/mean_td_target": mean_td_target,
        f"train/dqn_{pid}/td_error": td_error,
        f"train/dqn_{pid}/learning_rate": learning_rate,
        f"train/dqn_{pid}/norm_grad": norm_grad,
        f"train/dqn_{pid}/norm_updates": norm_updates,
        f"train/dqn_{pid}/explo_epsilon": explo_epsilon,
        f"train/dqn_{pid}/scheduler_steps": scheduler_steps,
    }
    return agent_metrics, trained
