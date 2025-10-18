#!/usr/bin/env python3
"""
Parallel Multi-Algorithm Training Script

This script runs Nash PG, FSP, and PSRO training algorithms in parallel
across multiple GPUs with load balancing. It trains models with different
mag_coef values (for Nash PG) and run IDs for statistical robustness.

Features:
- Distributes work across multiple GPUs for optimal resource utilization
- Limits concurrent processes per GPU (default: 3) to prevent overload
- Provides real-time progress tracking and error reporting
- Supports Nash PG, FSP, and PSRO algorithms
- Configurable magnetic coefficient values and number of runs
- Graceful error handling with detailed status reporting

Usage:
  uv run scripts/run_all.py --num-gpus 4
  uv run scripts/run_all.py --num-gpus 4 --max-concurrent-per-gpu 4
  uv run scripts/run_all.py --num-gpus 2 --mag-coefs 0.0 0.1 0.2 --num-runs 3
  uv run scripts/run_all.py --num-gpus 4 --algorithms nash_pg fsp
"""

import asyncio
import argparse
import logging
from typing import List, Tuple
from datetime import datetime

# Configuration constants
DEFAULT_MAG_COEFFICIENTS = [0.2]
DEFAULT_MMD_COEFFICIENTS = [0.05]
DEFAULT_DIVERGENCE_TYPES = ["kl"]
DEFAULT_NUM_RUNS = 4
DEFAULT_ALGORITHMS = ["nash_pg", "fsp", "psro", "mmd"]

# Default training parameters
DEFAULT_NUM_INNER_UPDATE = 1000
DEFAULT_NUM_OUTER_UPDATE = 50
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_LOG_INTERVAL = 10

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('scripts/npg_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_training_configs(algorithms: List[str], mag_coefs: List[float], mmd_coefs: List[float], divergence_types: List[str], num_runs: int) -> List[Tuple[str, float, str, int]]:
    """Generate all training configurations to run.
    
    Args:
        algorithms: List of algorithms to run (nash_pg, fsp, psro, mmd)
        mag_coefs: List of magnetic coefficient values (only used for nash_pg)
        mmd_coefs: List of MMD coefficient values (only used for mmd)
        divergence_types: List of divergence types (only used for nash_pg)
        num_runs: Number of runs per configuration
        
    Returns:
        List of (algorithm, coef_value, divergence_type, run_id) tuples
    """
    configs = []
    for algorithm in algorithms:
        if algorithm == "nash_pg":
            # Nash PG uses different magnetic coefficients and divergence types
            for mag_coef in mag_coefs:
                for divergence_type in divergence_types:
                    for run_id in range(num_runs):
                        configs.append((algorithm, mag_coef, divergence_type, run_id))
        elif algorithm == "mmd":
            # MMD uses different MMD coefficients, always uses KL divergence
            for mmd_coef in mmd_coefs:
                for run_id in range(num_runs):
                    configs.append((algorithm, mmd_coef, "kl", run_id))
        else:
            # FSP and PSRO don't use magnetic coefficients or divergence types
            for run_id in range(num_runs):
                configs.append((algorithm, 0.0, "kl", run_id))  # coef_value and divergence_type not used for FSP/PSRO
    
    return configs


class GPUTaskManager:
    """Manages parallel task execution across multiple GPUs with concurrency limits."""
    
    def __init__(self, num_gpus: int, max_concurrent_per_gpu: int = 3):
        """Initialize GPU task manager.
        
        Args:
            num_gpus: Number of GPUs to use
            max_concurrent_per_gpu: Maximum concurrent processes per GPU
        """
        self.num_gpus = num_gpus
        self.max_concurrent_per_gpu = max_concurrent_per_gpu
        self.gpu_semaphores = [asyncio.Semaphore(max_concurrent_per_gpu) for _ in range(num_gpus)]
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_tasks = 0
        self.start_time = None
        
    async def run_algorithm_training(self, algorithm: str, coef_value: float, divergence_type: str, run_id: int, gpu_id: int, agent: str, env: str, 
                                   num_inner_update: int, num_outer_update: int, save_interval: int, log_interval: int) -> Tuple[str, bool, str]:
        """Run algorithm training for a single configuration on specified GPU.
        
        Args:
            algorithm: Algorithm type (nash_pg, fsp, psro, mmd)
            coef_value: Coefficient value (mag_coef for nash_pg, mmd_coef for mmd)
            divergence_type: Divergence type (only used for nash_pg)
            run_id: Run identifier
            gpu_id: GPU device ID to use
            agent: Agent type to use
            env: Environment type to use
            num_inner_update: Number of inner update steps
            num_outer_update: Number of outer update steps
            save_interval: Checkpoint save interval
            log_interval: Logging interval
            
        Returns:
            Tuple of (config_name, success, error_message)
        """
        async with self.gpu_semaphores[gpu_id]:
            # Use "both" logging for run_id 0, "json" for others
            logging_type = "both" if run_id == 0 else "json"
            
            if algorithm == "nash_pg":
                config_name = f"nash_pg_mag{coef_value}_{divergence_type}_run{run_id}"
                run_name = f"{env}/nash_pg/mag{coef_value}_{divergence_type}_run{run_id}"
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu_id} uv run train/nash_pg.py "
                    f"algorithm.num_inner_update={num_inner_update} "
                    f"algorithm.num_outer_update={num_outer_update} "
                    f"agent={agent} "
                    f"env={env} "
                    f"algorithm.mag_coef={coef_value} "
                    f"algorithm.mag_divergence_type={divergence_type} "
                    f"logging={logging_type} "
                    f"logging.save_interval={save_interval} "
                    f"logging.log_interval={log_interval} "
                    f"seed={run_id+100} "
                    f'run_name="{run_name}"'
                )
            elif algorithm == "mmd":
                config_name = f"mmd_coef{coef_value}_run{run_id}"
                run_name = f"{env}/mmd/coef{coef_value}_run{run_id}"
                total_inner_updates = num_inner_update * num_outer_update
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu_id} uv run train/nash_pg.py "
                    f"algorithm.num_inner_update={total_inner_updates} "
                    f"algorithm.num_outer_update=1 "
                    f"agent={agent} "
                    f"env={env} "
                    f"algorithm.mag_coef={coef_value} "
                    f"algorithm.mag_divergence_type=kl "
                    f"logging={logging_type} "
                    f"logging.save_interval={save_interval} "
                    f"logging.log_interval={log_interval} "
                    f"seed={run_id+100} "
                    f'run_name="{run_name}"'
                )
            elif algorithm == "fsp":
                config_name = f"fsp_run{run_id}"
                run_name = f"{env}/fsp/default_run{run_id}"
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu_id} uv run train/psro.py "
                    f"--config-name fsp "
                    f"agent={agent} "
                    f"env={env} "
                    f"algorithm.num_inner_update={num_inner_update} "
                    f"algorithm.num_outer_update={num_outer_update+1} "
                    f"logging={logging_type} "
                    f"logging.save_interval={save_interval} "
                    f"logging.log_interval={log_interval} "
                    f"seed={run_id+100} "
                    f'run_name="{run_name}"'
                )
            elif algorithm == "psro":
                config_name = f"psro_run{run_id}"
                run_name = f"{env}/psro/default_run{run_id}"
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={gpu_id} uv run train/psro.py "
                    f"--config-name psro "
                    f"agent={agent} "
                    f"env={env} "
                    f"algorithm.num_inner_update={num_inner_update} "
                    f"algorithm.num_outer_update={num_outer_update+1} "
                    f"logging={logging_type} "
                    f"logging.save_interval={save_interval} "
                    f"logging.log_interval={log_interval} "
                    f"seed={run_id+100} "
                    f'run_name="{run_name}"'
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            try:
                logger.info(f"Starting GPU {gpu_id}: {config_name}")
                
                # Run the command
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                _, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.completed_tasks += 1
                    logger.info(f"✓ GPU {gpu_id} completed: {config_name}")
                    return config_name, True, ""
                else:
                    self.failed_tasks += 1
                    error_msg = stderr.decode() if stderr else f"Process failed with return code {process.returncode}"
                    logger.error(f"✗ GPU {gpu_id} failed: {config_name} - {error_msg}")
                    return config_name, False, error_msg
                    
            except Exception as e:
                self.failed_tasks += 1
                error_msg = str(e)
                logger.error(f"✗ GPU {gpu_id} exception: {config_name} - {error_msg}")
                return config_name, False, error_msg
    
    async def process_all_configs(self, configs: List[Tuple[str, float, str, int]], agent: str, env: str,
                                 num_inner_update: int, num_outer_update: int, save_interval: int, log_interval: int) -> List[Tuple[str, bool, str]]:
        """Process all training configurations in parallel across GPUs.
        
        Args:
            configs: List of (algorithm, coef_value, divergence_type, run_id) tuples to process
            agent: Agent type to use
            env: Environment type to use
            num_inner_update: Number of inner update steps
            num_outer_update: Number of outer update steps
            save_interval: Checkpoint save interval
            log_interval: Logging interval
            
        Returns:
            List of results (config_name, success, error_message)
        """
        self.total_tasks = len(configs)
        self.start_time = datetime.now()
        
        logger.info(f"Starting parallel training of {self.total_tasks} configurations across {self.num_gpus} GPUs")
        logger.info(f"Max concurrent processes per GPU: {self.max_concurrent_per_gpu}")
        
        # Create tasks with GPU assignment (round-robin distribution)
        tasks = []
        for i, (algorithm, coef_value, divergence_type, run_id) in enumerate(configs):
            gpu_id = i % self.num_gpus
            task = self.run_algorithm_training(algorithm, coef_value, divergence_type, run_id, gpu_id, agent, env,
                                             num_inner_update, num_outer_update, save_interval, log_interval)
            tasks.append(task)
        
        # Start progress monitoring
        progress_task = asyncio.create_task(self._monitor_progress())
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cancel progress monitoring
        progress_task.cancel()
        
        # Process results and handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                self.failed_tasks += 1
                final_results.append(("unknown", False, str(result)))
            else:
                final_results.append(result)
        
        self._print_final_summary()
        return final_results
    
    async def _monitor_progress(self):
        """Monitor and report progress every 30 seconds."""
        try:
            while True:
                await asyncio.sleep(30)
                completed = self.completed_tasks + self.failed_tasks
                if completed > 0:
                    elapsed = datetime.now() - self.start_time
                    rate = completed / elapsed.total_seconds() * 60  # configs per minute
                    remaining = self.total_tasks - completed
                    eta_minutes = remaining / rate if rate > 0 else 0
                    
                    logger.info(f"Progress: {completed}/{self.total_tasks} "
                               f"({completed/self.total_tasks*100:.1f}%) - "
                               f"Rate: {rate:.1f} configs/min - "
                               f"ETA: {eta_minutes:.1f} min")
        except asyncio.CancelledError:
            pass
    
    def _print_final_summary(self):
        """Print final execution summary."""
        elapsed = datetime.now() - self.start_time
        logger.info("="*60)
        logger.info("EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total configurations processed: {self.total_tasks}")
        logger.info(f"Successful: {self.completed_tasks}")
        logger.info(f"Failed: {self.failed_tasks}")
        logger.info(f"Success rate: {self.completed_tasks/self.total_tasks*100:.1f}%")
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Average time per config: {elapsed.total_seconds()/self.total_tasks:.1f}s")


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Nash PG, FSP, and PSRO training in parallel across GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/run_all.py --num-gpus 4 --agent kuhn_poker --env kuhn_poker
  python scripts/run_all.py --num-gpus 2 --max-concurrent-per-gpu 2 --agent liar_dice --env liar_dice
  python scripts/run_all.py --num-gpus 1 --mag-coefs 0.0 0.1 0.2 --num-runs 3
  python scripts/run_all.py --num-gpus 4 --algorithms nash_pg fsp
  python scripts/run_all.py --num-gpus 4 --dry-run
        """
    )
    
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for parallel processing (default: 1)"
    )
    
    parser.add_argument(
        "--max-concurrent-per-gpu",
        type=int,
        default=2,
        help="Maximum concurrent processes per GPU (default: 3)"
    )
    
    parser.add_argument(
        "--mag-coefs",
        type=float,
        nargs="+",
        default=DEFAULT_MAG_COEFFICIENTS,
        help=f"Magnetic coefficient values to use (default: {DEFAULT_MAG_COEFFICIENTS})"
    )
    
    parser.add_argument(
        "--divergence-types",
        nargs="+",
        choices=DEFAULT_DIVERGENCE_TYPES,
        default=DEFAULT_DIVERGENCE_TYPES,
        help=f"Divergence types for Nash PG (default: {DEFAULT_DIVERGENCE_TYPES})"
    )
    
    parser.add_argument(
        "--mmd-coefs",
        type=float,
        nargs="+",
        default=DEFAULT_MMD_COEFFICIENTS,
        help=f"MMD coefficient values to use (default: {DEFAULT_MMD_COEFFICIENTS})"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of runs per configuration (default: {DEFAULT_NUM_RUNS})"
    )
    
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=DEFAULT_ALGORITHMS,
        default=DEFAULT_ALGORITHMS,
        help=f"Algorithms to run (default: nash_pg, fsp, psro, mmd)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be trained without actually running commands"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        default="liar_dice",
        help="Agent type to use (default: liar_dice)"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        default="liar_dice",
        help="Environment type to use (default: liar_dice)"
    )
    
    parser.add_argument(
        "--num-inner-update",
        type=int,
        default=DEFAULT_NUM_INNER_UPDATE,
        help=f"Number of inner update steps (default: {DEFAULT_NUM_INNER_UPDATE})"
    )
    
    parser.add_argument(
        "--num-outer-update",
        type=int,
        default=DEFAULT_NUM_OUTER_UPDATE,
        help=f"Number of outer update steps (default: {DEFAULT_NUM_OUTER_UPDATE})"
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_SAVE_INTERVAL,
        help=f"Checkpoint save interval (default: {DEFAULT_SAVE_INTERVAL})"
    )
    
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help=f"Logging interval (default: {DEFAULT_LOG_INTERVAL})"
    )
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate all training configurations
    logger.info("Generating training configurations...")
    configs = generate_training_configs(args.algorithms, args.mag_coefs, args.mmd_coefs, args.divergence_types, args.num_runs)
    logger.info(f"Generated {len(configs)} training configurations")
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Magnetic coefficients (for Nash PG): {args.mag_coefs}")
    logger.info(f"MMD coefficients (for MMD): {args.mmd_coefs}")
    logger.info(f"Divergence types (for Nash PG): {args.divergence_types}")
    logger.info(f"Runs per configuration: {args.num_runs}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Would train the following configurations:")
        for i, (algorithm, coef_value, divergence_type, run_id) in enumerate(configs):
            gpu_id = i % args.num_gpus
            if algorithm == "nash_pg":
                config_name = f"{algorithm}_mag{coef_value}_{divergence_type}_run{run_id}"
            elif algorithm == "mmd":
                config_name = f"{algorithm}_coef{coef_value}_run{run_id}"
            else:
                config_name = f"{algorithm}_run{run_id}"
            logger.info(f"  GPU {gpu_id}: {config_name}")
        logger.info(f"Total: {len(configs)} configurations across {args.num_gpus} GPUs")
        return 0
    
    # Process configurations
    task_manager = GPUTaskManager(args.num_gpus, args.max_concurrent_per_gpu)
    results = await task_manager.process_all_configs(configs, args.agent, args.env,
                                                    args.num_inner_update, args.num_outer_update,
                                                    args.save_interval, args.log_interval)
    
    # Report failed configurations
    failed_configs = [result for result in results if not result[1]]
    if failed_configs:
        logger.error(f"\nFailed to train {len(failed_configs)} configurations:")
        for config_name, _, error in failed_configs:
            logger.error(f"  - {config_name}: {error}")
    
    return 0 if task_manager.failed_tasks == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)