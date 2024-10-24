# imports
import numpy as np
from numpy.linalg import norm
from typing import Tuple, Dict, List

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
def define_agent_model(
    N: int,
    agent_index: int,
    time_horizon: int,
    quality_factors: np.ndarray,
    mu: float,
    marginal_costs: np.ndarray,
    capacities: np.ndarray,
    demand_scale_factor: float,
    discount_factors: np.ndarray,
    current_prices: Dict[int, np.ndarray],
    regularization_tau=0,
) -> ConcreteModel:
    """
    Define the pyomo model for a single agent's optimization problem

    Note: I call the demand_scaling_factor 'lambda'.

    Returns:
    - Pyomo model of optimization problem with pyomo variables, constraints and objectives.

    """
    model = ConcreteModel()
    lower_bound = 0
    upper_bound = 3 + max(
        quality_factors
    )  # this must be chosen sensibly! price=5 already means demand=0 for q=2.

    # Index sets for agents and time. Pyomo indexes [first_val, ..., last_val] hence the -1
    model.Time = RangeSet(0, time_horizon - 1)
    model.Agents = RangeSet(0, N - 1)
    model.OtherAgents = model.Agents - [agent_index]

    # Variables
    model.prices = Var(
        model.Time, bounds=(lower_bound, upper_bound), domain=NonNegativeReals
    )  # Decision variables for i's prices.
    model.d_i = Var(
        model.Time, within=Integers, bounds=(0, None)
    )  # Agent i's integer demand
    model.d_j = Var(
        model.OtherAgents, model.Time, within=Integers, bounds=(0, None)
    )  # Other agents' integer demand
    model.a_j = Var(
        model.OtherAgents, model.Time, within=Binary
    )  # Other agents' active status

    # Initial active set: all agents are active at t=0, so we set all a_j[j,0] to 1
    for j in model.OtherAgents:
        model.a_j[j, 0].fix(1)  # All agents start as active at t=0

    # Demand function for agent i. Takes into account active set at time t via the Vars a_j[j,t] (if 0, then term in summation for that agent = 0).
    def demand_i(model, t):
        active_agents_sum = sum(
            exp((quality_factors[j] - current_prices[j][t]) / mu) * model.a_j[j, t]
            for j in model.OtherAgents
        )
        numerator = exp((quality_factors[agent_index] - model.prices[t]) / mu)
        denominator = 1 + numerator + active_agents_sum
        return numerator / denominator

    # Demand function for agents j != i. Takes into account active set at time t via the Vars a_j[j,t] (if 0, then term in summation for that agent = 0).
    def demand_j(model, j, t):
        active_agents_sum = sum(
            exp((quality_factors[k] - current_prices[k][t]) / mu) * model.a_j[k, t]
            for k in model.OtherAgents
            if k != j
        )
        numerator = exp((quality_factors[j] - current_prices[j][t]) / mu)
        denominator = (
            1
            + numerator
            + exp((quality_factors[agent_index] - model.prices[t]) / mu)
            + active_agents_sum
        )
        return numerator / denominator

    # Build up constraints
    model.activation_demand_constraints = (
        ConstraintList()
    )  # N*T many. These force a[j,t]=1 if agent j active (not sold out) at t
    model.integer_demand_constraints_i = (
        ConstraintList()
    )  # T many. These force d[i,t] to be == floor(demand[i,t] * lambda)
    model.integer_demand_constraints_j = (
        ConstraintList()
    )  # (N-1)*T many. These force d[j,t] to be == flor(demand[i,t] * lambda)
    M1 = max(
        1e5, time_horizon * demand_scale_factor
    )  # ensures M is big enough to unbind activation constraints (theoretical max value for cumulative demand on 1 agent is T*lambda)
    M2 = max(
        1e5, max(capacities)
    )  # this ensures M is big enough to bind deactivation constraints

    # t=0:
    for t in model.Time:
        # add i's constraints
        model.integer_demand_constraints_i.add(
            model.d_i[t] <= demand_i(model, t) * demand_scale_factor
        )
        model.integer_demand_constraints_i.add(
            model.d_i[t] >= demand_i(model, t) * demand_scale_factor - 1
        )

        # add j's constraints
        for j in model.OtherAgents:
            if t == 0:  # at t=0 all agents are active, omit constraints for active set.
                model.integer_demand_constraints_j.add(
                    model.d_j[j, t] <= demand_j(model, j, t) * demand_scale_factor
                )
                model.integer_demand_constraints_j.add(
                    model.d_j[j, t] >= demand_j(model, j, t) * demand_scale_factor - 1
                )
            else:
                # add constraints for active set at t (depend on d_j for s < t (!)). Comment out next 2 lines to simulate not having an active set.
                cumulative_demand_j_up_to_t_minus_1 = sum(
                    model.d_j[j, s] for s in range(t)
                )  # j's cumulative demand for s<t determines if j is active at t
                model.activation_demand_constraints.add(
                    cumulative_demand_j_up_to_t_minus_1
                    <= capacities[j] - 1 + M1 * (1 - model.a_j[j, t])
                )  # "activation": if j active (a=1), then cumulative demand (D) < capacity (I). vice-versa, deactivates j (a=0) if D >= I.
                model.activation_demand_constraints.add(
                    cumulative_demand_j_up_to_t_minus_1
                    >= capacities[j] - M2 * model.a_j[j, t]
                )  # "deactivation": if j inactive (a=0), then D >= I. vice-versa, activates j (a=1) if D <= I.

                # add j's constraints at t (depend on a_j at t)
                model.integer_demand_constraints_j.add(
                    model.d_j[j, t] <= demand_j(model, j, t) * demand_scale_factor
                )
                model.integer_demand_constraints_j.add(
                    model.d_j[j, t] >= demand_j(model, j, t) * demand_scale_factor - 1
                )

    # Inventory constraint for agent i
    def inventory_constraint_i(model):
        return summation(model.d_i) <= capacities[agent_index]

    model.InventoryConstraintI = Constraint(rule=inventory_constraint_i)

    # Objective for agent i
    def revenue_objective(model):
        # total_revenue = summation({t: (model.prices[t] - marginal_costs[agent_index]) * model.d_i[t] for t in model.Time})
        total_revenue = sum(
            (model.prices[t] - marginal_costs[agent_index])
            * model.d_i[t]
            * (discount_factors[agent_index] ** t)
            for t in model.Time
        )
        regularization_term = -regularization_tau * sum(
            (model.prices[t] - current_prices[agent_index][t]) ** 2 for t in model.Time
        )  # penalizes large jumps in prices from the last iteration.
        return total_revenue + regularization_term

    model.RevenueObjective = Objective(rule=revenue_objective, sense=maximize)

    return model


def solve_gnep(
    N: int,
    time_horizon: int,
    quality_factors: List[float],
    mu: float,
    marginal_costs: List[float],
    capacities: List[int],
    demand_scale_factor: float,
    discount_factors: List[float],
    epsilon=1e-3,
    solver_name="ipopt",
    method="gauss-seidel",
    regularization_tau=0,
    debug=False,
    initial_prices="zeros",
) -> None:
    """
    Initializes the pyomo model and solves a GNEP.

    Returns:
    - Price vectors for all agents that together constitute a Nash Equilibrium.
    """
    total_start_time = time.time()

    ## Initial guess for prices, structured as a dictionary mapping each agent to a numpy array of prices
    if initial_prices == "zeros":
        current_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, 0) for i in range(N)
        }
    elif initial_prices == "random":  # in (0,1)
        current_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, np.random.rand()) for i in range(N)
        }
    elif initial_prices == "marginal_cost":
        current_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, marginal_costs[i]) for i in range(N)
        }
    elif initial_prices == "marginal_cost_plus_random":
        current_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, marginal_costs[i] + np.random.rand())
            for i in range(N)
        }
    elif initial_prices == "quality_factor":
        current_prices: Dict[int, np.ndarray] = {
            i: np.full(time_horizon, quality_factors[i]) for i in range(N)
        }
    else:
        print(
            "invalid initial_prices chosen. initial_prices must be 'zeros', 'random', 'marginal_cost', 'marginal_cost_plus_random' or 'quality_factor'. exiting."
        )
        sys.exit(1)
    if debug:
        print(f"initial prices: {current_prices}")

    ## Initialize dicts/arrays
    current_demands: Dict[int, np.ndarray] = {
        i: np.zeros(time_horizon) for i in range(N)
    }
    current_demands_full: Dict[int, np.ndarray] = {
        i: np.zeros((N, time_horizon)) for i in range(N)
    }
    current_actives: Dict[int, np.ndarray] = {
        i: np.zeros((N, time_horizon)) for i in range(N)
    }
    previous_prices: Dict[int, np.ndarray] = {
        i: np.zeros(time_horizon) for i in range(N)
    }  # init prev prices for convergence check
    iteration_count = 0  # iteration counter
    max_iterations = 100  # Prevent infinite loops

    ## Start main GNEP solving loop
    while iteration_count < max_iterations:
        # Copy current prices to previous for convergence check
        for i in range(N):
            previous_prices[i] = np.copy(current_prices[i])

        ## Iteratively solve each agent's problem.
        # Current_prices are updated either inside (=Gauss-Seidel) or outside (=Jacobi) the loop.
        start_times = np.zeros(N)
        end_times = np.zeros(N)
        for i in range(N):
            if method == "jacobi":
                ### Jacobi: uses previous_prices, i.e. the estimates from beginning of current iteration
                model = define_agent_model(
                    N,
                    i,
                    time_horizon,
                    quality_factors,
                    mu,
                    marginal_costs,
                    capacities,
                    demand_scale_factor,
                    discount_factors,
                    previous_prices,  # use previous prices for Jacobi
                    regularization_tau,
                )
            elif method == "gauss-seidel":
                ### Gauss-Seidel: uses most up-to-date estimates via current_prices (j<i: updated. i: variable. j>i: not yet updated)
                model = define_agent_model(
                    N,
                    i,
                    time_horizon,
                    quality_factors,
                    mu,
                    marginal_costs,
                    capacities,
                    demand_scale_factor,
                    discount_factors,
                    current_prices,  # use current prices for Gauss-Seidel
                    regularization_tau,
                )
            else:
                raise ValueError(
                    f"invalid method chosen. options are 'jacobi', 'gauss-seidel'. exiting."
                )
            solver = SolverFactory(modules.find(solver_name), solve_io="nl")
            start_times[i] = time.time()
            if debug:
                solver.options["print_level"] = 5
                solver.solve(model, tee=True)
            else:
                solver.solve(model)
            end_times[i] = time.time()
            print(
                f"It {iteration_count}, Agent {i}, solved in {end_times[i] - start_times[i]:.2f}s."
            )

            # Update current prices based on the optimization results
            current_prices[i] = np.array(
                [value(model.prices[t]) for t in range(time_horizon)]
            )
            current_demands[i] = np.array(
                [value(model.d_i[t]) for t in range(time_horizon)]
            )
            current_demands_full[i][i, :] = np.array(
                [value(model.d_i[t]) for t in range(time_horizon)]
            )
            for j in model.OtherAgents:
                for t in model.Time:
                    current_demands_full[i][j, t] = value(model.d_j[j, t])

            current_actives[i][i, :] = np.ones(time_horizon)
            for j in model.OtherAgents:
                for t in model.Time:
                    current_actives[i][j, t] = value(model.a_j[j, t])
            if debug:
                print(
                    f"it {iteration_count}, after solving for i={i}, price_{i}={current_prices[i]}, other prices={current_prices}"
                )
            ### end of agent loop ###

        price_changes = [norm(current_prices[i] - previous_prices[i]) for i in range(N)]

        ## Calc and print current loop prices/demands etc
        all_prices = np.concatenate(list(current_prices.values()))
        price_per_agent = [current_prices[i] for i in range(N)]
        median_price_per_agent = np.median(price_per_agent)
        mean_price_per_agent = [np.round(np.mean(ppa), 2) for ppa in price_per_agent]
        stdev_price_per_agent = [
            np.round(np.std(price_per_agent), 2) for ppa in price_per_agent
        ]
        demands_per_agent = [current_demands[i] for i in range(N)]
        cumulative_demand_per_agent = [
            np.round(np.sum(dpa), 2) for dpa in demands_per_agent
        ]
        it_times = [np.round(end_times[i] - start_times[i], 2) for i in range(N)]
        total_time = end_times[-1] - start_times[0]

        ## printing some statistics about this iteration's result
        print(
            f"it: {iteration_count} in {it_times}s, total {total_time:.2f}s, all prices median/mean/stdev: ({np.median(all_prices):.3f}|{np.mean(all_prices):.3f}|{np.std(all_prices):.3f}), delta median/mean/stdev: ({np.median(price_changes):.3f}|{np.mean(price_changes):.3f}|{np.std(price_changes):.3f}) per agent prices mean/stdev: ({mean_price_per_agent}|{stdev_price_per_agent}), demands: {cumulative_demand_per_agent}"
        )

        print(f"Raw prices per agent (p_i the GNEP sol'n):")
        # printing the raw prices of each agent resulting from their optimization subproblem
        for i in range(N):
            formatted_prices = (
                "[" + ", ".join([f"{price:.4f}" for price in current_prices[i]]) + "]"
            )
            print(f" Agent {i}: {formatted_prices}")

        # printing the raw demands of all agents for each agent's optimization subproblem output
        print(f"Raw demands per agent: (d_j[0],..,d_i,...,d_j[N]):")
        for i in range(N):
            raw_demands = ", ".join(
                "[" + " ".join(f"{num:.0f}" for num in row) + "]"
                for row in current_demands_full[i]
            )
            print(f" Agent {i}: {raw_demands}")

        # printing the active set of all agents for each agent's optimization subproblem output
        print(f"Raw actives per agent (a_j[0],...,a_i=[1..1], a_j[N]):")
        for i in range(N):
            raw_actives = ", ".join(
                "[" + " ".join(f"{num:.0f}" for num in row) + "]"
                for row in current_actives[i]
            )
            print(f" Agent {i}: {raw_actives}")

        ## Convergence check:
        if all(change < epsilon for change in price_changes):
            print(f"Convergence achieved after {iteration_count} iterations.")
            break
        iteration_count += 1
        ### end of iteration loop ###

    if iteration_count == max_iterations:
        print("Maximum iterations reached without convergence.")

    ## Once done optimising, simulate one period using the final price vectors
    final_demands: Dict[int, np.ndarray] = {i: np.zeros(time_horizon) for i in range(N)}
    final_profits: Dict[int, np.ndarray] = {i: np.zeros(time_horizon) for i in range(N)}
    inventory_left = capacities
    active_agents = np.full(N, True)
    social_welfare = 0
    for t in range(time_horizon):
        # at time t, calculate demand and revenue for each agent i
        for i in range(N):
            # calc i's demand. uses active_agents from previous round (initially all are active)
            active_agents_sum = sum(
                exp((quality_factors[j] - current_prices[j][t]) / mu)
                for j in range(N)
                if (j != i and active_agents[j])
            )
            numerator = exp((quality_factors[i] - current_prices[i][t]) / mu)
            denominator = 1 + numerator + active_agents_sum
            final_demands[i][t] = np.floor(
                numerator / denominator * demand_scale_factor
            )

            # calc i's revenue this round
            final_profits[i][t] = (
                (current_prices[i][t] - marginal_costs[i])
                * final_demands[i][t]
                * (discount_factors[i] ** t)
            )

        # once revenues & demands are calculated, update the inventories & active set
        for i in range(N):
            inventory_left[i] -= final_demands[i][t]
            active_agents[i] = inventory_left[i] > 0

    # print results
    print(f"Sim results after convergence:")
    for i in range(N):
        formatted_prices = (
            "[" + ", ".join([f"{price:.3f}" for price in current_prices[i]]) + "]"
        )
        formatted_demands = (
            "[" + ", ".join([f"{demand:.0f}" for demand in final_demands[i]]) + "]"
        )
        total_demand = sum(final_demands[i])
        formatted_profits = (
            "[" + ", ".join([f"{profit:.2f}" for profit in final_profits[i]]) + "]"
        )
        total_profit = sum(final_profits[i])
        social_welfare += total_profit
        print(f"Agent {i}:")
        print(f"  Prices: {formatted_prices}")
        print(f"  Demands: {formatted_demands}. Total demand: {total_demand:.2f}.")
        print(f"  Profits: {formatted_profits}. Total profit: {total_profit:.2f}")
    print(f"Social welfare: {social_welfare:.2f}")
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f}s")
    return current_prices, final_demands, final_profits


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
    parser.add_argument(
        "--method",
        type=str,
        default="gauss-seidel",
        choices=["gauss-seidel", "jacobi"],
        help="GNEP method to use",
    )
    parser.add_argument(
        "--regularization_tau",
        type=float,
        default=0.01,
        help="Regularization parameter",
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

    # Adjust regularization_tau
    regularization_tau_scaled = args.regularization_tau * args.demand_scale_factor

    ####
    # Calvano: N=2, T=1, mu=0.25, quality=2, cost=1, capacity=1, demand_scale_factor=1, truncated=False ==> expect competitive price of ca 1.47. can also be done with truncated=False, demand_scale_factor=1000, capacity=10e6.
    # If truncation: note that demand_scale_factor must be >1, else all demand=0.
    # For non-inventory constrained case: set capacity=lambda * T
    ####

    print(f"Setup:")
    print(
        f"N={args.N}, T={args.time_horizon}, Epsilon: {args.epsilon}, regularisation: {args.regularization_tau}, solver: {args.solver_name}, method: {args.method}, initial prices: {args.initial_prices}"
    )
    print(
        f"Total demand over time: {args.time_horizon * args.demand_scale_factor} vs capacities of {capacities} w/ cumulative capacity of {np.sum(capacities)}"
    )
    print(
        f"Quality factors: {quality_factors}, Marginal costs: {marginal_costs}, Discount factors: {discount_factors}"
    )

    # Solve the GNEP
    prices, demands, profits = solve_gnep(
        args.N,
        args.time_horizon,
        quality_factors,
        args.mu,
        marginal_costs,
        capacities,
        args.demand_scale_factor,
        args.discount_factors,
        args.epsilon,
        args.solver_name,
        args.method,
        regularization_tau_scaled,
        args.debug,
        args.initial_prices,
    )


if __name__ == "__main__":
    main()
