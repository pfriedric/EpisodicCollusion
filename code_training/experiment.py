from datetime import datetime
import os
from runners.twoagent_rl_calvano_runner import CalvanoTwoAgentRLRunner
import wandb
import omegaconf
import jax
import jax.numpy as jnp
from agents.strategies import Random, Deterministic
from environment.market_env import MarketEnv, EnvParams
from environment.market_env_infinite_inventory import MarketEnvInfiniteInventoryEpisodic
from environment.market_env_infinite_inventory_no_resets import (
    MarketEnvInfiniteInventoryInfiniteEpisode,
)
from environment.in_the_matrix import InTheMatrix
from environment.in_the_matrix import EnvParams as InTheMatrixParams
from runners.eval_runner import EvalRunner
from runners.twoagent_rl_runner import TwoAgentRLRunner
from runners.twoagent_rl_calvano_runner import CalvanoTwoAgentRLRunner
from runners.twoagent_gridsearch_runner import TwoAgentGridsearchRunner
from runners.twoagent_gridsearch_runner_no_logging import (
    TwoAgentGridsearchNoLoggingRunner,
)
from agents.ppo.ppo import make_agent
from agents.dqn.dqn import make_DQN_agent

from watchers import losses_ppo, losses_dqn


def global_setup(args):
    """Set up global variables."""
    # check if we're doing a gridsearchupdate_dict = omegaconf.OmegaConf.to_container(args.gridsearch)
    update_dict = omegaconf.OmegaConf.to_container(args.gridsearch)
    doing_gridsearch = (
        any(leaf is not None for leaf in jax.tree_util.tree_leaves(update_dict))
        or args.get("num_seeds") > 1
    )
    if doing_gridsearch:
        # different save_dir
        save_dir = f"./exp/{args.wandb.group}/"
        # create dir if doesn't exist yet
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = f"{args.save_dir}"  # /{str(datetime.now()).replace(' ', '_').replace(':', '.')}"
    if not args.get("runner") == "eval":
        os.makedirs(
            save_dir,
            exist_ok=True,
        )
    if args.wandb.log:
        print("run name", str(args.wandb.name))
        if args.debug:
            args.wandb.group = "debug-" + args.wandb.group
        run = wandb.init(
            reinit=True,
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            mode=str(args.wandb.mode),
            tags=args.wandb.tags,
            config=omegaconf.OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            ),  # type: ignore
            settings=wandb.Settings(code_dir="."),
        )
        print("run id", run.id)
        wandb.run.log_code(".")
    return save_dir


###
def env_setup(args, logger=None):
    """Set up the environment."""
    if (
        args.env_id == "MarketEnv-v1"
        or args.env_id == "MarketEnv-InfiniteInventoryEpisodic"
        or args.env_id == "MarketEnv-InfiniteInventoryInfiniteEpisode"
    ):
        if args.env_id == "MarketEnv-v1":
            env = MarketEnv(
                num_agents=args.num_players,
                num_actions=args.num_prices,
                time_horizon=args.time_horizon,
            )
        elif args.env_id == "MarketEnv-InfiniteInventoryEpisodic":
            env = MarketEnvInfiniteInventoryEpisodic(
                num_agents=args.num_players,
                num_actions=args.num_prices,
                time_horizon=args.time_horizon,
            )
        elif args.env_id == "MarketEnv-InfiniteInventoryInfiniteEpisode":
            env = MarketEnvInfiniteInventoryInfiniteEpisode(
                num_agents=args.num_players,
                num_actions=args.num_prices,
                time_horizon=args.time_horizon,
            )
        env_params = EnvParams(
            time_horizon=args.time_horizon,
            min_price=args.min_price,
            max_price=args.max_price,
            num_prices=args.num_prices,
            possible_prices=jnp.array(
                omegaconf.OmegaConf.to_container(args.possible_prices, resolve=True)
            ),
            qualities=jnp.array(
                omegaconf.OmegaConf.to_container(args.qualities, resolve=True)
            ),
            marginal_costs=jnp.array(
                omegaconf.OmegaConf.to_container(args.marginal_costs, resolve=True)
            ),
            horizontal_diff=args.horizontal_diff,
            demand_scaling_factor=args.demand_scaling_factor,
            initial_inventories=jnp.array(
                omegaconf.OmegaConf.to_container(args.initial_inventories, resolve=True)
            ),
            initial_prices=jnp.array(
                omegaconf.OmegaConf.to_container(args.initial_prices, resolve=True)
            ),
            initial_actions=jnp.array(
                omegaconf.OmegaConf.to_container(args.initial_actions, resolve=True)
            ),
        )
        if logger:
            logger.info(
                f"Environment: {env.name} | Episode length: {args.time_horizon} | Num players: {args.num_players} | Prices between: {args.min_price} and {args.max_price} | Num price steps: {args.num_prices} | Initial inventories: {args.initial_inventories} | Initial prices: {args.initial_prices} | Initial actions: {args.initial_actions} | Qualities: {args.qualities} | Marginal costs: {args.marginal_costs} | Horizontal diff: {args.horizontal_diff} | Demand scaling factor: {args.demand_scaling_factor}"
            )
    elif args.env_id == "InTheMatrix":
        payoff = jnp.array(args.payoff)
        env_params = InTheMatrixParams(payoff_matrix=payoff, freeze_penalty=args.freeze)
        env = InTheMatrix(
            num_inner_steps=args.num_inner_steps,
            num_outer_steps=args.num_outer_steps,
            fixed_coin_location=args.fixed_coins,
        )
        if logger:
            logger.info(
                f"Env Type: InTheMatrix | Inner Episode Length: {args.num_inner_steps}"
            )
    else:
        raise ValueError(f"Unknown env id {args.env_id}")
    return env, env_params


def runner_setup(args, env, agents, save_dir, logger):
    """Set up the runner for the experiment."""
    # TODO: implement runner!
    if args.get("runner") == "eval":
        logger.info("Evaluating with EvalRunner")
        return EvalRunner(agents, env, args)

    elif args.get("runner") == "rl":
        logger.info("Training with two player RL Runner")
        return TwoAgentRLRunner(agents, env, save_dir, args)
    elif args.get("runner") == "rl-calvano":
        logger.info("Training with two player RL Runner, infinite game w/ memory")
        return CalvanoTwoAgentRLRunner(agents, env, save_dir, args)
    elif args.get("runner") == "rl-gridsearch":
        logger.info("Gridsearch with two player RL Runner")
        return TwoAgentGridsearchRunner(agents, env, save_dir, args)
    elif args.get("runner") == "rl-gridsearch-no-logging":
        logger.info("Gridsearch with two player RL Runner, no logging")
        return TwoAgentGridsearchNoLoggingRunner(agents, env, save_dir, args)
    else:
        raise ValueError(f"Unknown runner type {args.get('runner')}")


def agent_setup(args, env, env_params, logger):
    """Set up agents for the experiment."""
    print(env.observation_space(env_params))  # traced vals.
    if (
        args.get("env_id") == "MarketEnv-v1"
        or args.get("env_id") == "MarketEnv-InfiniteInventoryEpisodic"
        or args.get("env_id") == "MarketEnv-InfiniteInventoryInfiniteEpisode"
        or args.get("env_id") == "InTheMatrix"
    ):
        obs_shape = jax.tree_util.tree_map(
            lambda x: x.shape, env.observation_space(env_params)
        )
        obs_dtypes = jax.tree_util.tree_map(
            lambda x: x.dtype, env.observation_space(env_params)
        )
        # print(obs_shape)
    num_actions = env.num_actions

    def get_random_agent(seed, player_id):
        random_agent = Random(num_actions, args.get("num_envs"), obs_shape)
        random_agent.player_id = player_id
        return random_agent

    def get_PPO_agent(seed, player_id):
        default_player_args = args.get("ppo_default")
        player_key = f"ppo{player_id}"
        player_args = args.get(player_key, default_player_args)
        if player_args is None:
            raise ValueError(
                f"No args found for player {player_key} and no default set"
            )

        num_iterations = args.get("num_iters")
        if player_id == 1 and args.get("env_type") == "meta":
            num_iterations = args.get("num_outer_steps")
        return make_agent(
            args,
            player_args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            num_iterations=num_iterations,
            seed=seed,
            player_id=player_id,
        )

    def get_DQN_agent(seed, player_id):
        default_player_args = args.get("dqn_default")
        player_key = f"dqn{player_id}"
        player_args = args.get(player_key, default_player_args)
        if player_args is None:
            raise ValueError(
                f"No args found for player {player_key} and no default set"
            )

        num_iterations = args.get("num_iters")
        if player_id == 1 and args.get("env_type") == "meta":
            num_iterations = args.get("num_outer_steps")
        return make_DQN_agent(
            args,
            player_args,
            obs_spec=obs_shape,
            obs_dtypes=obs_dtypes,
            action_spec=num_actions,
            num_iterations=num_iterations,
            seed=seed,
            player_id=player_id,
        )

    # main calc's the collusive/competitive price and puts it into args
    # then the agent is created with that price
    def get_competitive_agent(seed, player_id):
        competitive_agent = Deterministic(
            num_actions, args.get("num_envs"), obs_shape, args.get("competitive_action")
        )
        competitive_agent.player_id = player_id
        return competitive_agent

    def get_collusive_agent(seed, player_id):
        collusive_agent = Deterministic(
            num_actions, args.get("num_envs"), obs_shape, args.get("collusive_action")
        )
        collusive_agent.player_id = player_id
        return collusive_agent

    strategies = {
        "Random": get_random_agent,
        "PPO": get_PPO_agent,
        "DQN": get_DQN_agent,
        "Competitive": get_competitive_agent,
        "Collusive": get_collusive_agent,
    }

    num_players = args.get("num_players")
    default_agent = args.get("agent_default", None)
    agent_strategies = [
        args.get(f"agent{i}", default_agent) for i in range(1, num_players + 1)
    ]
    # Check that all strategies are valid
    for strategy in agent_strategies:
        assert strategy in strategies

    base_seed = args.get("seed")

    def create_agent(player_idx):
        player_seed = base_seed + player_idx
        player_id = player_idx + 1
        strategy = agent_strategies[player_idx]
        return (strategies[strategy](player_seed, player_id), player_seed)

    agents = [create_agent(i)[0] for i in range(num_players)]
    seeds = [create_agent(i)[1] for i in range(num_players)]
    logger.info(f"Agent Pair: {agents}")
    logger.info(f"Agent seeds: {seeds}")

    return agents


def watcher_setup(args, logger):
    """Set up the watcher variables"""

    def ppo_log(agent):
        losses = losses_ppo(agent)
        if args.wandb.log:
            wandb.log(losses, commit=False)
        return

    def dqn_log(agent):
        losses, trained = losses_dqn(agent)
        if args.wandb.log:
            if trained:
                wandb.log(losses, commit=False)
            else:
                # print("Not logging, since it's a non-training episode")
                pass
        return

    def dumb_log(agent, *args):
        return

    strategies = {
        "Random": dumb_log,
        "PPO": ppo_log,
        "DQN": dqn_log,
        "Competitive": dumb_log,
        "Collusive": dumb_log,
    }

    agent_log = []
    default_agent = omegaconf.OmegaConf.select(args, "agent_default", default=None)
    agent_strategies = [
        omegaconf.OmegaConf.select(args, "agent" + str(i), default=default_agent)
        for i in range(1, args.num_players + 1)
    ]
    for strategy in agent_strategies:
        assert strategy in strategies
        agent_log.append(strategies[strategy])
    return agent_log
