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

from envs import create_env
from flax import nnx

from scripts.compute_elo.cli import parse_arguments
from scripts.compute_elo.loaders import load_agents
from scripts.compute_elo.tournament import run_swiss_tournament
from scripts.compute_elo.elo import compute_elo_by_step


def main():
    """Main function to run ELO computation"""
    try:
        args = parse_arguments()
        
        # Create environment and RNG based on argument
        env = create_env(args.env)
        rngs = nnx.Rngs(0)
        
        # Update global constants with command line arguments
        import scripts.compute_elo.config as config
        config.GAMES_PER_PAIRING = args.games_per_pairing
        config.NUM_TOURNAMENT_ROUNDS = args.tournament_rounds
        
        print(f"Environment: {args.env}")
        print(f"Algorithms: {args.algorithms}")
        print(f"Magnitude coefficients (Nash PG): {args.mag_coefs}")
        print(f"Divergence types (Nash PG): {args.mag_divergence_types}")
        print(f"MMD coefficients: {args.mmd_coefs}")
        print(f"Number of runs: {args.num_runs}")
        print(f"Games per pairing: {args.games_per_pairing}")
        print(f"Tournament rounds: {args.tournament_rounds}")
        print("Loading agents from checkpoints...")
        
        players = load_agents(args.env, args.algorithms, args.mag_coefs, args.mag_divergence_types, args.mmd_coefs, args.num_runs)
        
        if not players:
            print("No players loaded! Check checkpoint paths and log files.")
            return
        
        print(f"Loaded {len(players)} players")
        print(f"Each match will consist of {args.games_per_pairing} games to reduce variance")
        
        # Run Swiss tournament
        run_swiss_tournament(players, env, rngs, args.tournament_rounds)
        
        # Compute ELO by step for plotting
        elo_results = compute_elo_by_step(players)
        
        print("\n--- Final Results ---")
        for family, step_elos in elo_results.items():
            print(f"\n{family}:")
            for step, avg_elo in step_elos:
                print(f"  Step {step}: {avg_elo:.1f} ELO")
        
        # Save results for plotting
        if args.output is None:
            output_file = f"data/{args.env}_elo_results.json"
        else:
            output_file = args.output
            
        # Create output directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(elo_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()