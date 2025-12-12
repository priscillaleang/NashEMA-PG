#!/usr/bin/env python3
"""
Script to analyze training data from checkpoints and create visualizations.

Usage:
    python scripts/analyze_checkpoints.py --log-file logs/kuhn_poker/nash_pg/nash_pg_cv.json
    python scripts/analyze_checkpoints.py --log-dir logs/  # Analyze all logs in directory
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np


def load_log_file(log_path: Path) -> Dict[str, Any]:
    """Load a JSON log file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def smooth_data(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    """Apply moving average smoothing to data."""
    if window_size <= 1:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def extract_metric(entries: List[Dict], metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract steps and values for a given metric from log entries."""
    steps = []
    values = []
    for entry in entries:
        if metric_name in entry:
            steps.append(entry['step'])
            values.append(entry[metric_name])
    return np.array(steps), np.array(values)


def plot_training_metrics(
    log_data: Dict[str, Any],
    output_dir: Path,
    run_name: str,
    smoothing: int = 50
) -> None:
    """Create plots for training metrics."""
    train_data = log_data.get('train', [])
    if not train_data:
        print(f"No training data found for {run_name}")
        return

    # Define metrics to plot
    train_metrics = [
        ('actor_loss', 'Actor Loss'),
        ('ppo_loss', 'PPO Loss'),
        ('entropy', 'Entropy'),
        ('critic_loss', 'Critic Loss'),
        ('approx_kl', 'Approx KL'),
        ('mag_kl', 'MAG KL'),
        ('clip_frac', 'Clip Fraction'),
        ('explained_var', 'Explained Variance'),
    ]

    # Filter to metrics that exist in data
    available_metrics = []
    for metric_key, metric_label in train_metrics:
        steps, values = extract_metric(train_data, metric_key)
        if len(steps) > 0:
            available_metrics.append((metric_key, metric_label, steps, values))

    if not available_metrics:
        print(f"No training metrics found for {run_name}")
        return

    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, (metric_key, metric_label, steps, values) in enumerate(available_metrics):
        ax = axes[idx]

        # Plot raw data with low alpha
        ax.plot(steps, values, alpha=0.3, color='blue', linewidth=0.5)

        # Plot smoothed data
        if smoothing > 1 and len(values) > smoothing:
            smoothed = smooth_data(values, smoothing)
            smoothed_steps = steps[smoothing-1:]
            ax.plot(smoothed_steps, smoothed, color='blue', linewidth=1.5,
                   label=f'Smoothed (window={smoothing})')

        ax.set_xlabel('Step')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} over Training')
        ax.grid(True, alpha=0.3)
        if smoothing > 1:
            ax.legend(loc='best')

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Training Metrics - {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{run_name.replace("/", "_")}_training_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training metrics plot to {output_path}")


def plot_exploitability(
    log_data: Dict[str, Any],
    output_dir: Path,
    run_name: str,
) -> None:
    """Create plot for exploitability metric."""
    eval_data = log_data.get('eval', [])
    if not eval_data:
        print(f"No eval data found for {run_name}")
        return

    steps, values = extract_metric(eval_data, 'exploitability')
    if len(steps) == 0:
        print(f"No exploitability data found for {run_name}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(steps, values, color='red', linewidth=1.5, marker='o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Exploitability')
    ax.set_title(f'Exploitability over Training - {run_name}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f'{run_name.replace("/", "_")}_exploitability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved exploitability plot to {output_path}")


def plot_rollout_metrics(
    log_data: Dict[str, Any],
    output_dir: Path,
    run_name: str,
    smoothing: int = 50
) -> None:
    """Create plots for rollout metrics."""
    rollout_data = log_data.get('rollout', [])
    if not rollout_data:
        print(f"No rollout data found for {run_name}")
        return

    # Define metrics to plot
    rollout_metrics = [
        ('return', 'Return'),
        ('eps_len', 'Episode Length'),
    ]

    # Filter to metrics that exist in data
    available_metrics = []
    for metric_key, metric_label in rollout_metrics:
        steps, values = extract_metric(rollout_data, metric_key)
        if len(steps) > 0:
            available_metrics.append((metric_key, metric_label, steps, values))

    if not available_metrics:
        print(f"No rollout metrics found for {run_name}")
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for idx, (metric_key, metric_label, steps, values) in enumerate(available_metrics):
        ax = axes[idx]

        # Plot raw data with low alpha
        ax.plot(steps, values, alpha=0.3, color='green', linewidth=0.5)

        # Plot smoothed data
        if smoothing > 1 and len(values) > smoothing:
            smoothed = smooth_data(values, smoothing)
            smoothed_steps = steps[smoothing-1:]
            ax.plot(smoothed_steps, smoothed, color='green', linewidth=1.5,
                   label=f'Smoothed (window={smoothing})')

        ax.set_xlabel('Step')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} over Training')
        ax.grid(True, alpha=0.3)
        if smoothing > 1:
            ax.legend(loc='best')

    plt.suptitle(f'Rollout Metrics - {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{run_name.replace("/", "_")}_rollout_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rollout metrics plot to {output_path}")


def plot_combined_summary(
    log_data: Dict[str, Any],
    output_dir: Path,
    run_name: str,
    smoothing: int = 50
) -> None:
    """Create a combined summary plot with key metrics."""
    train_data = log_data.get('train', [])
    rollout_data = log_data.get('rollout', [])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Key metrics to show in summary
    summary_metrics = [
        ('train', 'actor_loss', 'Actor Loss', 'blue'),
        ('train', 'critic_loss', 'Critic Loss', 'red'),
        ('train', 'entropy', 'Entropy', 'purple'),
        ('train', 'approx_kl', 'Approx KL', 'orange'),
        ('rollout', 'return', 'Return', 'green'),
        ('rollout', 'eps_len', 'Episode Length', 'brown'),
    ]

    for idx, (data_type, metric_key, metric_label, color) in enumerate(summary_metrics):
        ax = axes[idx // 3, idx % 3]

        data = train_data if data_type == 'train' else rollout_data
        steps, values = extract_metric(data, metric_key)

        if len(steps) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_label)
            continue

        # Plot raw data
        ax.plot(steps, values, alpha=0.3, color=color, linewidth=0.5)

        # Plot smoothed data
        if smoothing > 1 and len(values) > smoothing:
            smoothed = smooth_data(values, smoothing)
            smoothed_steps = steps[smoothing-1:]
            ax.plot(smoothed_steps, smoothed, color=color, linewidth=1.5)

        ax.set_xlabel('Step')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Training Summary - {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{run_name.replace("/", "_")}_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to {output_path}")


def print_statistics(log_data: Dict[str, Any], run_name: str) -> None:
    """Print summary statistics for the training run."""
    config = log_data.get('config', {})
    train_data = log_data.get('train', [])
    rollout_data = log_data.get('rollout', [])

    print(f"\n{'='*60}")
    print(f"Statistics for: {run_name}")
    print(f"{'='*60}")

    # Config info
    if config:
        algo_config = config.get('algorithm', {})
        print(f"\nConfiguration:")
        print(f"  Learning Rate: {algo_config.get('lr', 'N/A')}")
        print(f"  Entropy Coef: {algo_config.get('ent_coef', 'N/A')}")
        print(f"  Num Envs: {algo_config.get('num_envs', 'N/A')}")
        print(f"  CV Enabled: {algo_config.get('cv_enabled', 'N/A')}")

    # Training stats
    if train_data:
        print(f"\nTraining Data: {len(train_data)} entries")
        steps = [e['step'] for e in train_data]
        print(f"  Step range: {min(steps)} - {max(steps)}")

        # Get final values (average of last 100)
        for metric in ['actor_loss', 'critic_loss', 'entropy', 'approx_kl']:
            _, values = extract_metric(train_data, metric)
            if len(values) > 0:
                final_avg = np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)
                print(f"  Final {metric}: {final_avg:.6f}")

    # Rollout stats
    if rollout_data:
        print(f"\nRollout Data: {len(rollout_data)} entries")
        _, returns = extract_metric(rollout_data, 'return')
        _, eps_lens = extract_metric(rollout_data, 'eps_len')

        if len(returns) > 0:
            final_return = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
            print(f"  Final avg return: {final_return:.6f}")
            print(f"  Max return: {np.max(returns):.6f}")
            print(f"  Min return: {np.min(returns):.6f}")

        if len(eps_lens) > 0:
            final_eps_len = np.mean(eps_lens[-100:]) if len(eps_lens) >= 100 else np.mean(eps_lens)
            print(f"  Final avg episode length: {final_eps_len:.4f}")

    # Exploitability stats
    eval_data = log_data.get('eval', [])
    if eval_data:
        print(f"\nExploitability Data: {len(eval_data)} entries")
        _, exploitability = extract_metric(eval_data, 'exploitability')
        if len(exploitability) > 0:
            print(f"  Final exploitability: {exploitability[-1]:.6f}")
            print(f"  Min exploitability: {np.min(exploitability):.6f}")
            print(f"  Max exploitability: {np.max(exploitability):.6f}")


def find_log_files(log_dir: Path) -> List[Path]:
    """Find all JSON log files in a directory."""
    return list(log_dir.rglob('*.json'))


def analyze_log_file(
    log_path: Path,
    output_dir: Path,
    smoothing: int = 50,
    show_stats: bool = True
) -> None:
    """Analyze a single log file and create plots."""
    print(f"\nAnalyzing: {log_path}")

    log_data = load_log_file(log_path)

    # Determine run name from path or config
    config = log_data.get('config', {})
    run_name = config.get('run_name', log_path.stem)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_training_metrics(log_data, output_dir, run_name, smoothing)
    plot_rollout_metrics(log_data, output_dir, run_name, smoothing)
    plot_combined_summary(log_data, output_dir, run_name, smoothing)
    plot_exploitability(log_data, output_dir, run_name)

    # Print statistics
    if show_stats:
        print_statistics(log_data, run_name)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training checkpoints and create visualizations'
    )
    parser.add_argument(
        '--log-file', '-f',
        type=Path,
        help='Path to a specific JSON log file'
    )
    parser.add_argument(
        '--log-dir', '-d',
        type=Path,
        help='Directory containing JSON log files (will process all)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('plots'),
        help='Output directory for plots (default: plots/)'
    )
    parser.add_argument(
        '--smoothing', '-s',
        type=int,
        default=50,
        help='Window size for moving average smoothing (default: 50)'
    )
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable printing statistics'
    )

    args = parser.parse_args()

    if args.log_file is None and args.log_dir is None:
        # Default: look for logs in the logs/ directory
        default_log_dir = Path('logs')
        if default_log_dir.exists():
            args.log_dir = default_log_dir
        else:
            parser.error('Please specify --log-file or --log-dir')

    log_files = []
    if args.log_file:
        log_files.append(args.log_file)
    if args.log_dir:
        found_files = find_log_files(args.log_dir)
        log_files.extend(found_files)
        print(f"Found {len(found_files)} log files in {args.log_dir}")

    if not log_files:
        print("No log files found to analyze")
        return

    for log_file in log_files:
        try:
            analyze_log_file(
                log_file,
                args.output_dir,
                args.smoothing,
                not args.no_stats
            )
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")

    print(f"\nPlots saved to: {args.output_dir.absolute()}")


if __name__ == '__main__':
    main()
