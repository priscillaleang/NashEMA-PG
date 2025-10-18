"""ELO computation module for trained agents using Swiss tournament."""

import os
import warnings
import logging
import json
from pathlib import Path

# Environment setup and imports suppression
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('orbax').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*Sharding info not provided when restoring.*')


"""Configuration constants for ELO computation."""

# Default configuration constants
DEFAULT_MAG_COEFFICIENTS = [0.2]
DEFAULT_MAG_DIVERGENCE_TYPE = ["kl"]
DEFAULT_MMD_COEFFICIENTS = [0.05]
DEFAULT_NUM_RUNS = 4
DEFAULT_ALGORITHMS = ["nash_pg", "mmd", "fsp", "psro"]

# ELO system constants
ELO_K_FACTOR = 32
INITIAL_ELO = 1500.0
NUM_TOURNAMENT_ROUNDS = 100  # Swiss tournament rounds
GAMES_PER_PAIRING = 100  # Number of games to play per pairing to reduce variance
STEP_PER_GAME = 128 # number of step for a game (we just JAX hence need fixed step)

# Default environment  
DEFAULT_ENV = "kuhn_poker"