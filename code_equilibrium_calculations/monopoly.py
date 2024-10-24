# imports
import numpy as np
from numpy.linalg import norm
from typing import Tuple, Dict, List

# install pyomo on the runtime
import sys
import os
import time

from amplpy import modules
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    Integers,
    Binary,
    SolverFactory,
    value,
    maximize,
    exp,
    summation,
    RangeSet,
    ConstraintList,
    floor,
    quicksum,
)
import pyomo.environ as pyo
import argparse


# Define a single agent's optimization problem
def define_monopoly_model(
    N: int,
    time_horizon: int,
    quality_factors: np.ndarray,
    mu: float,
    marginal_costs: np.ndarray,
    capacities: np.ndarray,
    demand_scale_factor: float,
    discount_factors: np.ndarray,
) -> ConcreteModel:
    """
    Define the pyomo model for a single agent's optimization problem

    Note: I call the demand_scaling_factor 'lambda'.

    Returns:
    - Pyomo model of optimization problem with pyomo variables, constraints and objectives.

    """
    model = ConcreteModel()
    lower_bound = 0
    upper_bound = 5  # this must be chosen sensibly!

    # Index sets for agents and time. Pyomo indexes [first_val, ..., last_val] hence the -1
    model.Time = RangeSet(0, time_horizon - 1)
    model.Items = RangeSet(
        0, N - 1
    )  # we call "agents" "items" now to emphasize that there's no decision making

    # Variables
    model.prices = Var(
        model.Items,
        model.Time,
        within=NonNegativeReals,
        bounds=(lower_bound, upper_bound),
    )  # Decision variables (prices) p_{i,t}
    model.demand = Var(
        model.Items, model.Time, within=Integers, bounds=(0, None)
    )  # Integer demands d_{i,t}
    model.actives = Var(model.Items, model.Time, within=Binary)  # Active status a_{i,t}

    # Parameters for qualities, costs, and capacities
    model.quality_factors = quality_factors
    model.marginal_costs = marginal_costs
    model.capacities = capacities

    # Initial active set: all items are active at t=0, so we set all a_i[i,0] to 1
    for i in model.Items:
        model.actives[i, 0].fix(1)  # All items start as active at t=0

    # Demand function for item i at time t. Takes into account active set at time t via the Vars active[j,t] (if 0, then term in summation for that item = 0).
    def demand_function(model, i, t):
        active_agents_sum = sum(
            exp((model.quality_factors[j] - model.prices[j, t]) / mu)
            * model.actives[j, t]
            for j in model.Items
            if j != i
        )
        numerator = exp((model.quality_factors[i] - model.prices[i, t]) / mu)
        denominator = 1 + numerator + active_agents_sum
        return numerator / denominator

    # Build up constraints
    model.activation_demand_constraints = (
        ConstraintList()
    )  # N*T many. These force a[i,t]=1 if item i active (not sold out) at t
    model.integer_demand_constraints = (
        ConstraintList()
    )  # N*T many. These force d[i,t] to be == floor(demand[i,t] * lambda)
    M1 = max(
        1e5, time_horizon * demand_scale_factor
    )  # ensures M is big enough to unbind activation constraints (theoretical max value for cumulative demand on 1 agent is T*lambda)
    M2 = max(
        1e5, max(capacities)
    )  # this ensures M is big enough to bind deactivation constraints

    # Defining active & integer demand constraints. This way each new constraint depends on already defined vars (not sure if it's needed to do it in this order)
    for t in model.Time:
        # add each item's constraints
        for i in model.Items:
            if t == 0:  # at t=0 all items are active, omit constraints for active set.
                model.integer_demand_constraints.add(
                    model.demand[i, t]
                    <= demand_function(model, i, t) * demand_scale_factor
                )
                model.integer_demand_constraints.add(
                    model.demand[i, t]
                    >= demand_function(model, i, t) * demand_scale_factor - 1
                )
            else:
                # add constraints for active set at t (depend on d_j for s < t (!)). Comment out next 2 lines to simulate not having an active set.
                cumulative_demand_i_up_to_t_minus_1 = quicksum(
                    model.demand[i, s] for s in range(t)
                )  # j's cumulative demand for s<t determines if j is active at t
                model.activation_demand_constraints.add(
                    cumulative_demand_i_up_to_t_minus_1
                    <= model.capacities[i] - 1 + M1 * (1 - model.actives[i, t])
                )  # "activation": if i active (a=1), then cumulative demand (D) < capacity (I). vice-versa, deactivates i (a=0) if D >= I.
                model.activation_demand_constraints.add(
                    cumulative_demand_i_up_to_t_minus_1
                    >= model.capacities[i] - M2 * model.actives[i, t]
                )  # "deactivation": if i inactive (a=0), then D >= I. vice-versa, activates i (a=1) if D <= I.

                # add j's constraints at t (depend on a_j at t)
                model.integer_demand_constraints.add(
                    model.demand[i, t]
                    <= demand_function(model, i, t) * demand_scale_factor
                )
                model.integer_demand_constraints.add(
                    model.demand[i, t]
                    >= demand_function(model, i, t) * demand_scale_factor - 1
                )

    # Inventory constraint for agent i
    def inventory_constraint_rule(model, i):
        return sum(model.demand[i, t] for t in model.Time) <= model.capacities[i]

    model.InventoryConstraint = Constraint(model.Items, rule=inventory_constraint_rule)

    def profit_objective(model):
        total_revenue = sum(
            model.prices[i, t] * model.demand[i, t] * (discount_factors[i] ** t)
            for i in model.Items
            for t in model.Time
        )  # sum_product(model.prices, model.demand)
        total_cost = sum(
            model.marginal_costs[i] * model.demand[i, t] * (discount_factors[i] ** t)
            for i in model.Items
            for t in model.Time
        )
        return total_revenue - total_cost

    model.ProfitObjective = Objective(rule=profit_objective, sense=maximize)

    return model


def solve_monopoly(
    N: int,
    time_horizon: int,
    quality_factors: List[float],
    mu: float,
    marginal_costs: List[float],
    capacities: List[int],
    demand_scale_factor: float,
    discount_factors: List[float],
    solver_name="ipopt",
    debug=False,
    initial_prices="zeros",
) -> None:
    """
    Initializes the pyomo model and solves a GNEP.

    Returns:
    - Price vectors for all agents that together constitute a Nash Equilibrium.
    """

    # Initial guess for prices, structured as a dictionary mapping each agent to a numpy array of prices
    if initial_prices == "zeros":
        init_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, 0) for i in range(N)
        }
    elif initial_prices == "random":  # in (0,1)
        init_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, np.random.rand()) for i in range(N)
        }
    elif initial_prices == "marginal_cost":
        init_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, marginal_costs[i]) for i in range(N)
        }
    elif initial_prices == "marginal_cost_plus_random":
        init_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, marginal_costs[i] + np.random.rand())
            for i in range(N)
        }
    elif initial_prices == "quality_factor":
        init_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, quality_factors[i]) for i in range(N)
        }
    else:
        raise ValueError(
            f"invalid initial_prices chosen: {initial_prices}. It must be 'zeros', 'random', 'marginal_cost' or 'marginal_cost_plus_random'!"
        )

    if debug:
        print(f"initial prices: {init_prices}")

    model = define_monopoly_model(
        N,
        time_horizon,
        quality_factors,
        mu,
        marginal_costs,
        capacities,
        demand_scale_factor,
        discount_factors,
    )

    solver = SolverFactory(solver_name)
    start_time = time.time()
    if debug:
        solver.options["print_level"] = 5
        solver.solve(model, tee=True)
    else:
        solver.solve(model)
    end_time = time.time()

    prices_dict = {}
    demands_dict = {}
    total_demands_dict = {}

    total_profit = value(model.ProfitObjective)
    final_profits: Dict[int, np.ndarray] = {i: np.zeros(time_horizon) for i in range(N)}
    print(f"Total Profit: {total_profit:.2f}")

    for i in model.Items:
        prices_dict[i] = np.array([value(model.prices[i, t]) for t in model.Time])
        formatted_prices = (
            "[" + ", ".join([f"{price:.3f}" for price in prices_dict[i]]) + "]"
        )
        demands_dict[i] = np.array([value(model.demand[i, t]) for t in model.Time])
        formatted_demands = (
            "[" + ", ".join([f"{demand:.0f}" for demand in demands_dict[i]]) + "]"
        )
        total_demands_dict[i] = np.sum(demands_dict[i])
        ## the below is redundant
        # item_profit = sum(
        #     (value(model.prices[i, t]) - model.marginal_costs[i])
        #     * value(model.demand[i, t])
        #     * (discount_factors[i] ** t)
        #     for t in model.Time
        # )
        for t in model.Time:
            final_profits[i][t] = (
                (prices_dict[i][t] - marginal_costs[i])
                * demands_dict[i][t]
                * (discount_factors[i] ** t)
            )
        total_profit = sum(final_profits[i])

        formatted_profits = (
            "[" + ", ".join([f"{profit:.2f}" for profit in final_profits[i]]) + "]"
        )
        print(f"Item {i}:")
        print(f"  Prices: {formatted_prices}")
        print(f"  Demands: {formatted_demands}")
        print(f"  Profits: {formatted_profits}")
        print(
            f"  Total Demand: {total_demands_dict[i]} vs capacity {model.capacities[i]}. Profit: {total_profit:.2f}"
        )

    # Print execution time
    print(f"Execution Time: {end_time - start_time} seconds")

    return prices_dict, demands_dict, total_demands_dict


# Argparser for input
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Solve the GNEP using specified parameters."
    )
    parser.add_argument("--N", type=int, default=2, help="Number of agents")
    parser.add_argument(
        "--time_horizon", type=int, default=5, help="Number of time steps"
    )
    parser.add_argument(
        "--mu", type=float, default=0.25, help="Mu parameter for the model"
    )
    parser.add_argument(
        "--discount_factors",
        type=float,
        nargs="+",
        help="Discount factors for future revenues",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-4, help="Convergence criterion threshold"
    )
    parser.add_argument(
        "--demand_scale_factor",
        type=int,
        default=1000,
        help="Scaling factor for demand per timestep",
    )
    parser.add_argument(
        "--quality_factors", type=float, nargs="+", help="Quality factors"
    )
    parser.add_argument(
        "--capacities",
        nargs="*",
        default=None,
        help='List of capacities for each agent, separated by space, or "unconstrained" to give all agents unlimited capacity',
    )
    parser.add_argument(
        "--marginal_costs", type=float, nargs="+", help="Marginal costs"
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        default="bonmin",
        choices=["ipopt", "bonmin", "couenne"],
        help="Solver to use",
    )
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode")
    parser.add_argument(
        "--initial_prices",
        type=str,
        default="quality_factor",
        choices=[
            "zeros",
            "random",
            "marginal_cost",
            "marginal_cost_plus_random",
            "quality_factor",
        ],
        help="Method to initialize prices",
    )

    args = parser.parse_args()

    if args.capacities == ["unconstrained"]:
        args.capacities = [
            int(args.time_horizon * args.demand_scale_factor * 1e5)
        ] * args.N
    elif args.capacities is None:
        args.capacities = [1000] * args.N  # default capacity if nothing is specified
    else:
        args.capacities = list(
            map(int, args.capacities)
        )  # ensure that all entries are integers

    if not args.quality_factors:
        args.quality_factors = [2] * args.N

    if not args.marginal_costs:
        args.marginal_costs = [1] * args.N

    if not args.discount_factors:
        args.discount_factors = [1] * args.N
    return args


def main():
    args = parse_arguments()

    # Ensure capacities is a numpy array
    capacities = np.array(args.capacities)
    quality_factors = np.array(args.quality_factors)
    marginal_costs = np.array(args.marginal_costs)
    discount_factors = np.array(args.discount_factors)

    ###
    # Calvano: N=2, T=1, mu=0.25, quality=2, cost=1, capacity=1, demand_scale_factor=1, truncated=False ==> expect competitive price of ca 1.47. can also be done with truncated=False, demand_scale_factor=1000, capacity=10e6.
    # If truncation: note that demand_scale_factor must be >1, else all demand=0.
    # For non-inventory constrained case: set capacity=lambda * T
    ###

    print(f"Setup:")
    print(
        f"N={args.N}, T={args.time_horizon}, solver: {args.solver_name}, initial prices: {args.initial_prices}"
    )
    print(
        f"Total demand over time: {args.time_horizon * args.demand_scale_factor} vs capacities of {capacities} w/ cumulative capacity of {np.sum(capacities)}"
    )
    print(
        f"Quality factors: {quality_factors}, Marginal costs: {marginal_costs}, Discount factors: {discount_factors}"
    )

    # Solve the Opt. Problem
    prices, demands, total_demands = solve_monopoly(
        args.N,
        args.time_horizon,
        quality_factors,
        args.mu,
        marginal_costs,
        capacities,
        args.demand_scale_factor,
        args.discount_factors,
        args.solver_name,
        args.debug,
        args.initial_prices,
    )


if __name__ == "__main__":
    main()
