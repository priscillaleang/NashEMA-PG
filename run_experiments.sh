#!/bin/bash
# This file run trainin agent -> compute exploitability -> compute Elo ratings
env_name=kuhn_poker
num_runs=4

uv run scripts/run_training.py --env $env_name --agent $env_name --num-runs $num_runs

uv run scripts/compute_exploits.py --env $env_name --num-runs $num_runs

uv run scripts/compute_elo.py --env $env_name --num-runs $num_runs