"""Command line interface for ELO computation."""

import argparse
from scripts.compute_elo.config import (
    DEFAULT_ENV, DEFAULT_ALGORITHMS, DEFAULT_MAG_COEFFICIENTS, DEFAULT_MAG_DIVERGENCE_TYPE,
    DEFAULT_MMD_COEFFICIENTS, DEFAULT_NUM_RUNS, GAMES_PER_PAIRING, NUM_TOURNAMENT_ROUNDS, STEP_PER_GAME
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compute ELO ratings for trained agents using Swiss tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/compute_elo --env kuhn_poker
  python scripts/compute_elo --env liar_dice --games-per-pairing 50
  python scripts/compute_elo --env tictactoe --tournament-rounds 50 --output results/my_elo.json
  python scripts/compute_elo --env kuhn_poker --algorithms nash_pg fsp --mag-coefs 0.0 0.5 1.0
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV,
        help=f"Environment name (default: {DEFAULT_ENV})"
    )
    
    parser.add_argument(
        "--games-per-pairing",
        type=int,
        default=GAMES_PER_PAIRING,
        help=f"Number of games per pairing to reduce variance (default: {GAMES_PER_PAIRING})"
    )
    
    parser.add_argument(
        "--tournament-rounds",
        type=int,
        default=NUM_TOURNAMENT_ROUNDS,
        help=f"Number of Swiss tournament rounds (default: {NUM_TOURNAMENT_ROUNDS})"
    )
    
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=["nash_pg", "mmd", "fsp", "psro"],
        help=f"Algorithms to include (default: {DEFAULT_ALGORITHMS})"
    )
    
    parser.add_argument(
        "--mag-coefs",
        type=float,
        nargs="+",
        default=DEFAULT_MAG_COEFFICIENTS,
        help=f"Magnitude coefficients for Nash PG (default: {DEFAULT_MAG_COEFFICIENTS})"
    )
    
    parser.add_argument(
        "--mag-divergence-types",
        type=str,
        nargs="+",
        default=DEFAULT_MAG_DIVERGENCE_TYPE,
        choices=["kl", "l2"],
        help=f"Divergence types for Nash PG (default: {DEFAULT_MAG_DIVERGENCE_TYPE})"
    )
    
    parser.add_argument(
        "--mmd-coefs",
        type=float,
        nargs="+",
        default=DEFAULT_MMD_COEFFICIENTS,
        help=f"MMD coefficients (default: {DEFAULT_MMD_COEFFICIENTS})"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of runs to include per algorithm (default: {DEFAULT_NUM_RUNS})"
    )
    
    parser.add_argument(
        "--step-per-game",
        type=int,
        default=STEP_PER_GAME,
        help=f"Number of steps per game (default: {STEP_PER_GAME})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ELO results JSON file (default: data/{env}_elo_results.json)"
    )
    
    return parser.parse_args()