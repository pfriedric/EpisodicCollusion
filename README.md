# Learning Collusion in Episodic, Inventory-Constrained Markets

This is the codebase for the [arXiv paper of the same title](https://arxiv.org/abs/2410.18871). For the training code, we have heavily adapted the [Pax: Scalable Opponent Shaping in JAX](https://github.com/ucl-dark/pax/tree/main) repo.

The project consists of two folders: 
- training and plotting agents (code_training)
- calculating the generalized Nash equilibria and monopolistic prices (code_equilibrium_calculations)

To avoid errors, treat the two folders `code_training` and `code_equilibrium_calculations` as separate projects and run the scripts from within each respective folder.

We recommend using the `uv` package manager ([link to docs](https://docs.astral.sh/uv/getting-started/installation/)). Get `uv` via

Macos/Unix: 
`$ curl -LsSf https://astral.sh/uv/install.sh | sh`

Windows:
`$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

or, platform-agnostic using `pip`:
`$ pip install uv`


## Instructions for training and plotting agents (code_training)
This part of the project uses Python, and JAX for the learning code. Navigate into this folder before running any scripts.

### Requirements
Create and activate a virtual environment with `uv` (alternatively e.g. `conda`), then install requirements with `$ uv pip` (if not using `uv`, just `$ pip`):

```
$ uv venv coll_venv --python 3.10
$ source coll_venv/bin/activate
$ uv pip install --no-cache-dir -r requirements.txt
```

This defaults to CPU. Use `requirements-gpu.txt` for the GPU version, which uses CUDA.

### Running experiments
Ensure the config you want is on the top level of the `conf/` directory, then call the main script with:

```
$ python main.py -cn CONFIG_NAME
```

where CONFIG_NAME is the name of your config, without `.yaml`. E.g. `-cn config_DQN`. To test if your setup is working, you can run the debug scripts (run in 100seconds on a M1 Max CPU)

```
$ python main.py -cn config_PPO_debug
$ python main.py -cn config_DQN_debug
```

The run results are saved to a folder in `exp/`.

### Plotting
To plot, use the plotting scripts. They use VSCode's functionality of using cells in a .py file, so you can run them as a notebook (running them as a script isn't tested).

At the top of the plotting script, directly below the imports, adjust the `save_dir` string to the run that you want to plot. E.g., if the run was saved to `exp/compPPO`, that should be the `save_dir`. 

We have the following plotting scripts:

- `plotting_means.py`: creates the arithmetic/geometric means plot. saves to main directory.
- `plotting_rewards.py`: some utils around visualizing reward surfaces. 
- `plotting_fig2_trainresult.py`: plots the evolution of actions & collusion index for a training run. function `plot_PPO_and_DQN_training_runs()` plots both DQN and PPO in one figure, make sure you've ran both algorithms. saves to the run's directory into `paper_plots/`, while the figure for both DQN and PPO saves to `exp/fig2_combined_runs/`
- `plotting_fig3a_deviation.py`: plots intra-episode behavior and forced deviation. saves to the run's directory into the `paper_plots/` subdirectory.
- `plotting_fig3b_reaction_surface.py`: plots the reaction surfaces of both agents. saves to the run's directory into the `paper_plots/` subdirectory.
- `plotting_fig4_hyperparams.py`: boxplots for hyperparameter gridsearch results. NOTE: to run this: 
  - collect the relevant configs in `conf/hyperparam boxplots/ (choose DQN or PPO)` and drag them into the top level of conf. 
  - start a run for each config. 
  - adjust the `save_dir = "DQN"` line in `plotting_fig4_hyperparams.py` to `DQN` or `PPO` depending on what you want to plot
  - figures saved to `exp/fig4_DQN_plots/` (or PPO)

## Instructions for calculating the equilibrium price levels (code_equilibrium_calculations)

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
