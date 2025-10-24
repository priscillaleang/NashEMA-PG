# Nash Policy Gradient: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria

This repository contains the implementation and experiments code for the paper "Nash Policy Gradient: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria".

## Environments

We implement several game environments, each with JAX-based implementations for efficient multi-agent reinforcement learning (MARL):

- **Kuhn Poker** (`envs/kuhn_poker.py`) - A simplified poker game with perfect recall observations [[Wikipedia](https://en.wikipedia.org/wiki/Kuhn_poker)]
- **Leduc Poker** (`envs/leduc_poker.py`) - A more complex poker variant with perfect recall observations [[OpenSpiel](https://github.com/google-deepmind/open_spiel/tree/master)]
- **Dark Hex 3** (`envs/dark_hex3.py`) - 3x3 hexagonal grid game where players cannot see opponent's stones. Players connect opposite edges. Supports both classic and abrupt variants [[OpenSpiel](https://github.com/google-deepmind/open_spiel/tree/master)]
- **Phantom Tic-Tac-Toe** (`envs/phantom_tictactoe.py`) - Tic-tac-toe where players cannot see opponent's moves and must deduce positions from blocked moves. Supports both classic and abrupt variants [[OpenSpiel](https://github.com/google-deepmind/open_spiel/tree/master)]
- **Liar's Dice** (`envs/liar_dice.py`) - Single hand version with 5 dice and 6 sides (compared to smaller variants in other frameworks) [[Wikipedia](https://en.wikipedia.org/wiki/Liar%27s_dice)]
- **Battleship** (`envs/battleship`) - Classic naval strategy game following standard rules [[Wikipedia](https://en.wikipedia.org/wiki/Battleship_(game))]
- **No-Limit Head-Up Texas Hold'em** (`envs/head_up_poker.py`) - Two-player no-limit Texas Hold'em following standard rules [[Wikipedia](https://en.wikipedia.org/wiki/Texas_hold_%27em)]

Each environment has corresponding agent implementations in the `agents/` directory.

## Installation

We use [uv](https://github.com/astral-sh/uv) as our package manager. Install dependencies by:
```bash
uv sync
```

## Training Scripts

We implement four algorithms and use [Hydra](https://hydra.cc/docs/intro/) to manage configuration. Each algorithm uses different run scripts and config files. You can look into `conf/algorithm/` to understand the hyper-parameters for each algorithm. Below are sample running commands.

To run training algorithms, we specify `env_name`. You can look into `conf/env/` to see supported environments. Generally, `agent_name` is the same as `env_name` in our implementation.

### Nash Policy Gradient (Nash PG)
```bash
uv run train/nash_pg.py \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/nash_pg"
```

### Magnetic Mirror Descent [[MMD](https://openreview.net/pdf?id=DpE5UYUQzZH#page=1.40)]
```bash
uv run train/mmd.py \
    algorithm.num_update=10000 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/mmd"
```

### Policy Space Response Oracles [[PSRO](https://arxiv.org/pdf/1711.00832#page=6.41)]
```bash
uv run train/psro.py \
    --config-name psro \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/psro"
```

### Neural Fictitious Self-Play [[NFSP](https://arxiv.org/pdf/1603.01121)]
```bash
uv run train/psro.py \
    --config-name fsp \
    algorithm.num_inner_update=1000 \
    algorithm.num_outer_update=10 \
    agent={env_name} \
    env={env_name} \
    run_name="{env_name}/fsp"
```

## Reproduce Our Experiment Results

The above commands allow you to run each algorithm individually. However, in our paper, we need to run all algorithms for multiple runs and calculate exploitability and Elo ratings. We created a script that runs all experiments and generates JSON log files and checkpoints.

To train agents for all four algorithms with multiple runs
```bash
uv run scripts/run_training.py --env $env_name --agent $env_name --num-runs $num_runs
```

However, these log files don't have exploitability data yet. We need to train best-response agents for each checkpoint using `scripts/compute_exploits.py`. This script will load the corresponding JSON file and checkpoints, train best-response agents to calculate exploitability, and store the results back to the JSON file.

After training, calculate exploitability of trained policies:

```bash
uv run scripts/compute_exploits.py --env $env_name --num-runs $num_runs
```

Finally, we have another script that takes all checkpoints, runs a Swiss tournament, and saves the Elo ratings in `data/`.

Calculate Elo ratings using Swiss tournament:

```bash
uv run scripts/compute_elo.py --env $env_name --num-runs $num_runs
```


## Repository Structure

```
nash_policy_gradient/
├── envs/                    # Game environment implementations
├── agents/                  # Neural network agent architectures
├── train/                   # Training algorithm implementations
├── scripts/                 # Evaluation and utility scripts
├── conf/                    # Configuration files
└── run_experiments.sh       # Automated experiment runner
```
## Citation

If you find this repository useful in your research or work, please consider citing our paper:

```bibtex
@misc{yu2025nashpolicygradientpolicy,
      title={Nash Policy Gradient: A Policy Gradient Method with Iteratively Refined Regularization for Finding Nash Equilibria}, 
      author={Eason Yu and Tzu Hao Liu and Yunke Wang and Clément L. Canonne and Nguyen H. Tran and Chang Xu},
      year={2025},
      eprint={2510.18183},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.18183}, 
}
```
