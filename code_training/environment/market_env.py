from typing import Tuple, Dict

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from jaxtyping import Array, Float, Integer, Num
from numpy import integer  # used for all Jax arrays


@chex.dataclass
class EnvState:
    inventories: Integer[Array, "..."]
    last_prices: Float[Array, "..."]
    last_actions: Integer[Array, "..."]
    t: Integer[Array, "1"]


# for the time being, all these are shared among agents. note that these aren't used in the environment class, they're passed through along with State.
@chex.dataclass
class EnvParams:
    time_horizon: int
    # prices are a discretized interval between [min_price, max_price] with num_price_steps many steps
    min_price: float
    max_price: float
    num_prices: int
    possible_prices: Float[Array, "..."]
    qualities: Float[Array, "..."]
    marginal_costs: Float[Array, "..."]
    horizontal_diff: float
    demand_scaling_factor: int
    initial_inventories: Integer[Array, "..."]  # will be initial capacity of each agent
    initial_prices: Float[
        Array, "..."
    ]  # should be a dummy variable that a neural net can identify as the initial price
    initial_actions: Num[
        Array, "..."
    ]  # should be a dummy variable that a neural net can identify as the initial action


# region ENVIRONMENT
class MarketEnv(environment.Environment):
    def __init__(self, num_agents: int, num_actions: int, time_horizon: int):
        super().__init__()
        self.num_players = num_agents
        self._num_actions = num_actions
        self._time_horizon = time_horizon

        def _MNL_demand(
            state: EnvState,
            actions: Float[Array, "num_agents"],
            qualities: Float[Array, "num_agents"],
            horizontal_diff: float,
            demand_scaling_factor: int,
        ) -> Integer[Array, "num_agents"]:
            nonzero_inventory_mask = (
                state.inventories > 0
            )  # mask selects only agents with nonzero inventory
            utilities = jnp.exp(
                (qualities - actions) / horizontal_diff
            )  # all agent utilities via element wise exponential
            conditional_utilities = jnp.where(
                nonzero_inventory_mask, utilities, 0
            )  # zero out utilities for agents with zero inventory
            sum_utilities = jnp.sum(
                conditional_utilities
            )  # Compute sum of utilities to go in denominator, but only for agents that are active
            demands = utilities / (sum_utilities + 1)  # Compute demand for each agent
            scaled_demands = demand_scaling_factor * demands  # scale the demands
            integer_demands = jnp.floor(scaled_demands)  # floor them to get integer demands
            return integer_demands

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Integer[Array, "num_agents"],
            params: EnvParams,
        ) -> Tuple[Tuple[Dict], EnvState, Float[Array, "num_agents"], bool, Dict]:
            # checkify.check(
            #     len(actions) == num_agents,
            #     "actions must have length num_agents, had {foo}",
            #     foo=len(actions),
            # )  # check that we have the right number of actions
            t = state.t  # get the current time step
            done = t + 1 >= params.time_horizon  # check if we're done

            actions = jnp.array(actions).squeeze()  # make sure actions is flat
            actions_pricevals = params.possible_prices[
                actions
            ]  # retrieve the price values from the discretized interval

            # calculate demand for each agent
            demands = _MNL_demand(
                state,
                actions_pricevals,
                params.qualities,
                params.horizontal_diff,
                params.demand_scaling_factor,
            ).astype(jnp.int32)

            ## update state
            # calculate quantity each agent actually sells, which can't exceed inventory
            quantities_sold = jnp.minimum(demands, state.inventories).astype(jnp.int32)
            new_inventories = state.inventories - quantities_sold
            # check that we haven't gone negative
            # checkify.check(
            #     jnp.all(new_inventories >= 0), "inventories must be nonnegative"
            # )

            new_prices = actions_pricevals
            new_t = t + 1
            state = EnvState(
                inventories=new_inventories,
                last_prices=new_prices,
                last_actions=actions,
                t=new_t,
            )

            # observations are state, actions as actions (not prices)
            obs = {
                "inventories": state.inventories,
                "last_prices": state.last_prices,
                "last_actions": state.last_actions,
                "t": state.t,
            }
            # compute rewards, for each agent it's (price - marginal_cost) * quantity_sold
            rewards = (new_prices - params.marginal_costs) * quantities_sold  # shape (num_agents,)

            # info, logging
            info = {
                "demands": demands,
                "quantity_sold": quantities_sold,
            }
            all_obs = tuple([obs for _ in range(num_agents)])
            return all_obs, state, rewards, done, info

        def _reset(key: chex.PRNGKey, params: EnvParams) -> Tuple[Tuple[Dict, ...], EnvState]:
            state = EnvState(
                inventories=params.initial_inventories.astype(jnp.int32),
                last_prices=params.initial_prices.astype(jnp.float32),
                last_actions=params.initial_actions.astype(jnp.int32),
                t=jnp.zeros(1, dtype=jnp.int32),
            )
            obs = {
                "inventories": state.inventories,
                "last_prices": state.last_prices,
                "last_actions": state.last_actions,
                "t": state.t,
            }
            all_obs = tuple([obs for _ in range(num_agents)])
            return all_obs, state

        # if not using checkify, we can just use the jit decorator
        # _step_checked = checkify.checkify(_step)
        # self.step = jax.jit(_step_checked)
        self.step = jax.jit(_step)
        # self.step = _step

        self.reset = jax.jit(_reset)
        self.MNL_demand = _MNL_demand

    @property
    def name(self) -> str:
        """Environment name."""
        return "MarketEnv-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._num_actions

    @property
    def time_horizon(self) -> int:
        """Time horizon of the environment."""
        return self._time_horizon

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment for one agent"""
        # return DiscretizedInterval(
        #     min_val=params.min_price,
        #     max_val=params.max_price,
        #     num_vals=params.num_price_steps,
        # )
        return spaces.Discrete(params.num_prices)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        # return spaces.Dict(
        #     {
        return {
            # inventories: array of shape (self.num_players,) filled with integers from 0 to some really high number
            "inventories": spaces.Box(
                low=0, high=integer.max, shape=self.num_players, dtype=jnp.int32
            ),
            "last_prices": spaces.Box(
                low=params.min_price,
                high=params.max_price,
                shape=self.num_players,
                dtype=jnp.float32,
            ),
            # last_actions: array of shape (self.num_players,) filled with integers from 0 to num_price_steps
            "last_actions": spaces.Box(
                low=0,
                high=params.num_prices,
                shape=self.num_players,
                dtype=jnp.int32,
            ),
            "t": spaces.Discrete(
                params.time_horizon
            ),  # this has default jnp.int64 which throws a warning that it's truncated to jnp.int32, nothing we can do as gymnax/ghm doesn't support setting dtype for discrete spaces
        }
        # )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "inventories": spaces.Box(
                    low=0, high=integer.max, shape=self.num_players, dtype=jnp.int32
                ),
                "last_prices": spaces.Box(
                    low=params.min_price,
                    high=params.max_price,
                    shape=self.num_players,
                    dtype=jnp.float32,
                ),
                "last_actions": spaces.Box(
                    low=0,
                    high=params.num_prices,
                    shape=self.num_players,
                    dtype=jnp.int32,
                ),
                "t": spaces.Discrete(params.time_horizon),
            }
        )

    # @staticmethod
    # def nash_policy(params: EnvParams) -> float:
    #     return 2 * (params.a - params.marginal_cost) / (3 * params.b)

    # @staticmethod
    # def nash_reward(params: EnvParams) -> float:
    #     q = CournotGame.nash_policy(params)
    #     p = params.a - params.b * q
    #     return p * q - params.marginal_cost * q


# endregion
