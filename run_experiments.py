#!/usr/bin/env python3
"""
Run multiple EarthScape experiments sequentially.

Automatically skips experiments that already have outputs in the outputs folder.
Logs progress and handles failures gracefully.

Usage:
    # Run all experiments
    python run_experiments.py

    # Run specific experiments
    python run_experiments.py --experiments experiments/swin_fusion.yaml experiments/resnet50_rgb.yaml

    # Dry run (show what would be executed)
    python run_experiments.py --dry-run
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_experiment_name(experiment_path: str) -> str:
    """Extract experiment name from YAML file."""
    import yaml
    
    try:
        with open(experiment_path) as f:
            exp = yaml.safe_load(f)
            return exp.get("name", Path(experiment_path).stem)
    except Exception:
        return Path(experiment_path).stem


def check_experiment_completed(experiment_name: str, outputs_dir: str = "outputs") -> bool:
    """
    Check if experiment has already been completed.
    
    An experiment is considered complete if it has:
    - A model directory with final_model.pth
    - Test metrics file
    """
    exp_output_dir = os.path.join(outputs_dir, experiment_name)
    
    if not os.path.exists(exp_output_dir):
        return False
    
    # Check for key output files
    final_model = os.path.join(exp_output_dir, "final_model.pth")
    test_metrics = os.path.join(exp_output_dir, "test_metrics.npy")
    
    return os.path.exists(final_model) and os.path.exists(test_metrics)


def run_experiment(experiment_path: str, dry_run: bool = False) -> tuple[bool, str]:
    """
    Run a single experiment.
    
    Returns:
        (success: bool, message: str)
    """
    experiment_name = get_experiment_name(experiment_path)
    
    # Check if already completed
    if check_experiment_completed(experiment_name):
        return True, f"SKIPPED (already completed)"
    
    cmd = [sys.executable, "train.py", "--experiment", experiment_path]
    
    if dry_run:
        return True, f"DRY RUN: would execute: {' '.join(cmd)}"
    
    print(f"\n{'='*80}")
    print(f"Starting: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        return True, "SUCCESS"
    
    except subprocess.CalledProcessError as e:
        return False, f"FAILED (exit code {e.returncode})"
    
    except KeyboardInterrupt:
        return False, "INTERRUPTED by user"
    
    except Exception as e:
        return False, f"ERROR: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Run EarthScape experiments sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiment files to run (default: all in experiments/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory to check for completed experiments (default: outputs)"
    )
    args = parser.parse_args()
    
    # Get list of experiments (only top-level experiments/*.yaml)
    if args.experiments:
        experiment_files = args.experiments
    else:
        experiment_files = sorted(glob.glob("experiments/*.yaml"))
    
    if not experiment_files:
        print("ERROR: No experiment files found")
        return 1
    
    print(f"\n{'='*80}")
    print(f"EarthScape Experiment Runner")
    print(f"{'='*80}")
    print(f"Found {len(experiment_files)} experiments")
    print(f"Outputs directory: {args.outputs_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"{'='*80}\n")
    
    # Track results
    results = []
    
    for i, exp_file in enumerate(experiment_files, 1):
        exp_name = get_experiment_name(exp_file)
        
        print(f"\n[{i}/{len(experiment_files)}] {exp_name}")
        print(f"File: {exp_file}")
        
        success, message = run_experiment(exp_file, dry_run=args.dry_run)
        results.append((exp_name, exp_file, success, message))
        
        print(f"Result: {message}")
        
        # Stop on keyboard interrupt
        if "INTERRUPTED" in message:
            print("\n\nExecution interrupted by user")
            break
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, _, success, _ in results if success)
    skipped = sum(1 for _, _, _, msg in results if "SKIPPED" in msg)
    failed = sum(1 for _, _, success, msg in results if not success and "SKIPPED" not in msg)
    
    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  ✗ Failed: {failed}")
    
    print(f"\nDetailed Results:")
    for exp_name, exp_file, success, message in results:
        status = "✓" if success else "✗"
        print(f"  {status} {exp_name:30s} - {message}")
    
    print(f"\n{'='*80}\n")
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
