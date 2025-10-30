# GNEP Solver

This script solves a Generalized Nash Equilibrium Problem (GNEP) for a multi-agent pricing scenario. Navigate into this folder before running any scripts.

### Requirements
Create and activate a virtual environment, then install requirements:

```
$ uv venv coll_venv --python 3.10
$ source coll_venv/bin/activate
$ uv pip install --no-cache-dir -r requirements.txt
```

For `amplpy`, you may need a free license, see instructions [on their site](https://amplpy.ampl.com/en/latest/).

Additionally, install numerical solvers via

`$ uv pip install -i https://portal.ampl.com/dl/pypi ampl_module_base ampl_module_coin`

We're using the following numerical solvers:
- IPOPT
- BONMIN
- COUENNE

Details can be found on the [COIN-OR site](https://www.coin-or.org/downloading/). Note: only Bonmin or Couenne correctly model the integer-valued demand.

### Usage

Run the script from the command line with various arguments:

```
$ python gnep_cs.py [arguments]
```

### Arguments

The script accepts the following command-line arguments:

- `--N`: Number of agents (default: 2)
- `--time_horizon`: Number of time steps (default: 5)
- `--mu`: Mu parameter for the model (default: 0.25)
- `--discount_factors`: Discount factors for future revenues (space-separated list)
- `--epsilon`: Convergence criterion threshold (default: 1e-4)
- `--demand_scale_factor`: Scaling factor for demand per timestep (default: 1000)
- `--quality_factors`: Quality factors (space-separated list)
- `--capacities`: List of capacities for each agent (space-separated list), or "unconstrained" for unlimited capacity
- `--marginal_costs`: Marginal costs (space-separated list)
- `--solver_name`: Solver to use (choices: "ipopt", "bonmin", "couenne", default: "bonmin"). IPOPT does not do demand substitution.
- `--method`: GNEP method to use (choices: "gauss-seidel", "jacobi", default: "gauss-seidel")
- `--regularization_tau`: Regularization parameter (default: 0.01)
- `--debug`: Enable debug mode (default: False)
- `--initial_prices`: Method to initialize prices (choices: "zeros", "random", "marginal_cost", "marginal_cost_plus_random", "quality_factor", default: "quality_factor")

### Example

```
$ python gnep_cs.py --N 2 --time_horizon 5 --capacities 440 440 --solver_name bonmin
```