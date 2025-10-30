# Training and plotting pricing agents
This part of the project uses Python and JAX. We have built on the [Pax: Scalable Opponent Shaping in JAX](https://github.com/ucl-dark/pax/tree/main) repo.

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
