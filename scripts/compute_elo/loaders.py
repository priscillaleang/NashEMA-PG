"""Agent loading and discovery logic."""

from typing import List
import json
from pathlib import Path
import jax.random as jrandom
from tqdm import tqdm

from agents import create_agent
from scripts.compute_elo.elo_agents import Player, MixtureAgent


def discover_runs(env_name: str, algorithms: list, mag_coefficients: list, mag_divergence_types: list, mmd_coefficients: list, num_runs: int) -> dict:
    """Discover all available runs for each algorithm"""
    runs = {"nash_pg": [], "mmd": [], "fsp": [], "psro": []}
    
    # Check nash_pg runs
    if "nash_pg" in algorithms:
        nash_pg_dir = Path(f"logs/{env_name}/nash_pg")
        if nash_pg_dir.exists():
            for mag_coef in mag_coefficients:
                for divergence_type in mag_divergence_types:
                    for run_num in range(num_runs):
                        log_file = nash_pg_dir / f"mag{mag_coef}_{divergence_type}_run{run_num}.json"
                        if log_file.exists():
                            runs["nash_pg"].append((mag_coef, divergence_type, run_num))
    
    # Check mmd runs
    if "mmd" in algorithms:
        mmd_dir = Path(f"logs/{env_name}/mmd")
        if mmd_dir.exists():
            for mmd_coef in mmd_coefficients:
                for run_num in range(num_runs):
                    log_file = mmd_dir / f"coef{mmd_coef}_run{run_num}.json"
                    if log_file.exists():
                        runs["mmd"].append((mmd_coef, run_num))
    
    # Check fsp runs
    if "fsp" in algorithms:
        fsp_dir = Path(f"logs/{env_name}/fsp")
        if fsp_dir.exists():
            for run_num in range(num_runs):
                log_file = fsp_dir / f"default_run{run_num}.json"
                if log_file.exists():
                    runs["fsp"].append(run_num)
    
    # Check psro runs
    if "psro" in algorithms:
        psro_dir = Path(f"logs/{env_name}/psro")
        if psro_dir.exists():
            for run_num in range(num_runs):
                log_file = psro_dir / f"default_run{run_num}.json"
                if log_file.exists():
                    runs["psro"].append(run_num)
    
    return runs


def load_agents(env_name: str, algorithms: list, mag_coefficients: list, mag_divergence_types: list, mmd_coefficients: list, num_runs: int) -> List[Player]:
    """Load checkpoints to get list of players"""
    players = []
    
    # Discover all available runs
    available_runs = discover_runs(env_name, algorithms, mag_coefficients, mag_divergence_types, mmd_coefficients, num_runs)
    
    # Log total discovered runs
    total_runs = sum(len(runs) for runs in available_runs.values())
    print(f"Discovered {total_runs} total runs across all algorithms")

    # Load Nash PG agents for each magnetism coefficient, divergence type and run
    if "nash_pg" in algorithms:
        nash_pg_runs = available_runs["nash_pg"]
        for mag_coef, divergence_type, run_num in tqdm(nash_pg_runs, desc="Loading Nash PG runs"):
            log_file = f"logs/{env_name}/nash_pg/mag{mag_coef}_{divergence_type}_run{run_num}.json"
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            config = log_data["config"]
            checkpoint_dir = config["logging"]["checkpoint_dir"]
            num_inner_update = config["algorithm"]["num_inner_update"]
            
            # Construct the run directory path
            run_dir = Path(checkpoint_dir) / env_name / "nash_pg" / f"mag{mag_coef}_{divergence_type}_run{run_num}"
            
            # Get available checkpoints
            checkpoint_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
            checkpoint_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
            
            key = jrandom.PRNGKey(0)
            
            for checkpoint_dir in tqdm(checkpoint_dirs, desc=f"Nash PG checkpoints (mag{mag_coef}_{divergence_type}_run{run_num})", leave=False):
                step = int(checkpoint_dir.name.split("_")[1])
                # Use actual training step, not outer loop step
                
                try:
                    # Create agent template and load checkpoint
                    agent = create_agent(config["agent"]["agent_name"], key)
                    loaded_agent = agent.__class__.load_checkpoint(
                        str(run_dir), step, key
                    )
                    
                    # Include run number and divergence type in family name to make each run its own family
                    family = f"nash_pg_mag{mag_coef}_{divergence_type}_run{run_num}"
                    player = Player(loaded_agent, family, step)  # Use training step
                    players.append(player)
                    
                except Exception as e:
                    print(f"Failed to load Nash PG checkpoint {checkpoint_dir}: {e}")
    
    # Load MMD agents for each coefficient and run
    if "mmd" in algorithms:
        for mmd_coef, run_num in tqdm(available_runs["mmd"], desc="Loading MMD runs"):
            log_file = f"logs/{env_name}/mmd/coef{mmd_coef}_run{run_num}.json"
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            config = log_data["config"]
            checkpoint_dir = config["logging"]["checkpoint_dir"]
            
            # Construct the run directory path
            run_dir = Path(checkpoint_dir) / env_name / "mmd" / f"coef{mmd_coef}_run{run_num}"
            
            # Get available checkpoints
            checkpoint_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
            checkpoint_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
            
            key = jrandom.PRNGKey(0)
            
            for checkpoint_dir in checkpoint_dirs:
                step = int(checkpoint_dir.name.split("_")[1])
                # Use actual training step, not outer loop step
                
                try:
                    # Create agent template and load checkpoint
                    agent = create_agent(config["agent"]["agent_name"], key)
                    loaded_agent = agent.__class__.load_checkpoint(
                        str(run_dir), step, key
                    )
                    
                    # Include run number in family name to make each run its own family
                    family = f"mmd_coef{mmd_coef}_run{run_num}"
                    player = Player(loaded_agent, family, step)  # Use training step
                    players.append(player)
                    
                except Exception as e:
                    print(f"Failed to load MMD checkpoint {checkpoint_dir}: {e}")
    
    # Load FSP agents for each run
    if "fsp" in algorithms:
        for run_num in tqdm(available_runs["fsp"], desc="Loading FSP runs"):
            log_file = f"logs/{env_name}/fsp/default_run{run_num}.json"
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            config = log_data["config"]
            checkpoint_dir = config["logging"]["checkpoint_dir"]
            
            # Construct the run directory path
            run_dir = Path(checkpoint_dir) / env_name / "fsp" / f"default_run{run_num}"
            
            # Load mixture probabilities
            mixture_file = run_dir / "mixture_probs.json"
            with open(mixture_file, 'r') as f:
                mixture_probs = json.load(f)[:-1]
            
            key = jrandom.PRNGKey(0)
            
            # Pre-load all FSP individual agents once (optimization)
            max_agents_needed = max(len(probs) for probs in mixture_probs) if mixture_probs else 0
            fsp_agents_cache = {}
            for i in range(max_agents_needed):
                try:
                    agent = create_agent(config["agent"]["agent_name"], key)
                    loaded_agent = agent.__class__.load_checkpoint(str(run_dir), i, key)
                    fsp_agents_cache[i] = loaded_agent
                except Exception as e:
                    print(f"Failed to load FSP checkpoint {i}: {e}")
            
            # Get num_inner_update from config to convert to training steps
            num_inner_update = config.get("algorithm", {}).get("num_inner_update", 1)
            
            # Create mixture agents for each outer step using cached agents
            for outer_step, probs in enumerate(mixture_probs):
                
                agents = []
                for i in range(len(probs)):
                    agents.append(fsp_agents_cache[i])
                
                if agents:
                    # Create mixture agent
                    mixture_agent = MixtureAgent(agents, probs)
                    # Include run number in family name to make each run its own family
                    family = f"fsp_run{run_num}"
                    # Convert outer step to training step
                    training_step = outer_step * num_inner_update
                    player = Player(mixture_agent, family, training_step)
                    players.append(player)
    
    # Load PSRO agents for each run
    if "psro" in algorithms:
        for run_num in tqdm(available_runs["psro"], desc="Loading PSRO runs"):
            log_file = f"logs/{env_name}/psro/default_run{run_num}.json"
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)

            config = log_data["config"]
            checkpoint_dir = config["logging"]["checkpoint_dir"]

            # Construct the run directory path
            run_dir = Path(checkpoint_dir) / env_name / "psro" / f"default_run{run_num}"

            # Load mixture probabilities
            mixture_file = run_dir / "mixture_probs.json"
            if mixture_file.exists():
                with open(mixture_file, 'r') as f:
                    mixture_probs = json.load(f)[:-1]
                
                key = jrandom.PRNGKey(0)
                
                # Pre-load all PSRO individual agents once (optimization)
                max_agents_needed = max(len(probs) for probs in mixture_probs) if mixture_probs else 0
                psro_agents_cache = {}
                for i in range(max_agents_needed):
                    try:
                        agent = create_agent(config["agent"]["agent_name"], key)
                        loaded_agent = agent.__class__.load_checkpoint(str(run_dir), i, key)
                        psro_agents_cache[i] = loaded_agent
                    except Exception as e:
                        print(f"Failed to load PSRO checkpoint {i}: {e}")
                
                # Get num_inner_update from config to convert to training steps
                num_inner_update = config.get("algorithm", {}).get("num_inner_update", 1)
                
                # Create mixture agents for each outer step using cached agents
                for outer_step, probs in enumerate(mixture_probs):
                    
                    agents = []
                    for i in range(len(probs)):
                        agents.append(psro_agents_cache[i])
                    
                    if agents:
                        # Create mixture agent
                        mixture_agent = MixtureAgent(agents, probs)
                        # Include run number in family name to make each run its own family
                        family = f"psro_run{run_num}"
                        # Convert outer step to training step
                        training_step = outer_step * num_inner_update
                        player = Player(mixture_agent, family, training_step)
                        players.append(player)
    
    return players