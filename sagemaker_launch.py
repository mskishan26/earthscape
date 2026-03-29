"""
SageMaker Launch Script  (SDK v2 — PyTorch Estimator API)

Launches one training job per experiment YAML in experiments/.
Each job runs on its own spot instance in parallel.

Usage:
    # Launch all experiments
    python sagemaker_launch.py

    # Launch a single experiment
    python sagemaker_launch.py --experiment experiments/midfusion_all.yaml

    # Override instance type
    python sagemaker_launch.py --instance-type ml.g5.xlarge

    # Use on-demand instead of spot
    python sagemaker_launch.py --no-spot
"""

import argparse
import glob
import os
import shutil
import tempfile
from pathlib import Path

import boto3
import yaml
import sagemaker
from sagemaker.pytorch import PyTorch


BUCKET = "earthscape-dataset"
OUTPUT_PREFIX = "sagemaker-outputs"
CHECKPOINT_PREFIX = "sagemaker-checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Single experiment YAML to run (default: all in experiments/)",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g4dn.xlarge",
        help="SageMaker instance type (default: ml.g4dn.xlarge)",
    )
    parser.add_argument(
        "--max-run-hours",
        type=int,
        default=24,
        help="Max runtime per job in hours (default: 24)",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Disable spot instances (use on-demand)",
    )
    parser.add_argument(
        "--role",
        default=os.environ.get("SM_EXECUTION_ROLE", ""),
        help="SageMaker execution role ARN (or set SM_EXECUTION_ROLE env var).",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        help="AWS region for SageMaker jobs (default: us-east-2)",
    )
    parser.add_argument(
        "--wandb-api-key",
        default=os.environ.get("WANDB_API_KEY", ""),
        help="W&B API key (reads WANDB_API_KEY env var by default)",
    )
    return parser.parse_args()


def get_experiment_name(path: str) -> str:
    """Read experiment name from YAML, fall back to filename stem."""
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("name", Path(path).stem)
    except Exception:
        return Path(path).stem


# Directories / patterns to EXCLUDE from source upload.
# Everything else under the project root is included.
IGNORE_PATTERNS = {
    ".venv",
    ".git",
    ".ruff_cache",
    "__pycache__",
    "data_cache",
    "outputs",
    "outputs1",
    "wandb",
    "standalone_scripts",
    "uv.lock",
    ".python-version",
}
IGNORE_EXTENSIONS = {
    ".html",
    ".pyc",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".gz"
}


def stage_source_dir(project_root: str) -> str:
    """Copy only the needed source files into a temp directory.

    SageMaker v2 tars the entire source_dir — this avoids uploading
    data_cache, .venv, wandb, etc.
    """
    tmp = tempfile.mkdtemp(prefix="sm_source_")
    root = Path(project_root)

    for item in root.rglob("*"):
        # Skip anything whose path components match an ignore pattern
        if any(part in IGNORE_PATTERNS for part in item.parts):
            continue
        if item.suffix in IGNORE_EXTENSIONS:
            continue
        if item.is_file():
            rel = item.relative_to(root)
            dest = Path(tmp) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    return tmp


def launch():
    args = parse_args()

    if args.experiment:
        experiment_files = [args.experiment]
    else:
        experiment_files = sorted(glob.glob("experiments/*.yaml"))
    if not experiment_files:
        print("No experiment YAMLs found in experiments/")
        return

    boto_session = boto3.Session(region_name=args.region)
    session = sagemaker.Session(boto_session=boto_session)
    print(f"Region: {args.region}")

    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            print(
                "ERROR: Could not auto-detect SageMaker execution role.\n"
                "  When launching locally, pass --role <ARN> or set SM_EXECUTION_ROLE.\n"
                "  The role must have a trust policy allowing sagemaker.amazonaws.com."
            )
            return
    print(f"Role: {role}")

    use_spot = not args.no_spot
    max_run_seconds = args.max_run_hours * 3600
    max_wait_seconds = max_run_seconds + 3600 if use_spot else None
    spot_label = "SPOT" if use_spot else "ON-DEMAND"

    # Stage a clean copy of the source tree (avoids uploading data_cache, .venv, etc.)
    project_root = os.path.dirname(os.path.abspath(__file__))
    staged_dir = stage_source_dir(project_root)
    print(f"Staged source to {staged_dir}")

    print(f"Found {len(experiment_files)} experiments")
    print(f"Instance: {args.instance_type} ({spot_label})\n")

    jobs = []
    for exp_file in experiment_files:
        exp_name = get_experiment_name(exp_file)

        try:
            estimator = PyTorch(
                entry_point="train.py",
                source_dir=staged_dir,
                role=role,
                sagemaker_session=session,
                framework_version="2.5.1",
                py_version="py311",
                instance_type=args.instance_type,
                instance_count=1,
                base_job_name=f"es-{exp_name}".replace("_", "-"),
                hyperparameters={"experiment": exp_file},
                environment={"WANDB_API_KEY": args.wandb_api_key},
                output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/",
                checkpoint_s3_uri=f"s3://{BUCKET}/{CHECKPOINT_PREFIX}/{exp_name}/",
                checkpoint_local_path="/opt/ml/checkpoints",
                use_spot_instances=use_spot,
                max_run=max_run_seconds,
                max_wait=max_wait_seconds,
                disable_profiler=True,
                keep_alive_period_in_seconds=0,
            )

            estimator.fit(wait=False)
            job_name = estimator.latest_training_job.name
            jobs.append((exp_name, job_name))
            print(f"  Launched: {exp_name:30s} -> {job_name}")

        except Exception as e:
            print(f"  FAILED:   {exp_name:30s} -> {e}")

    print(f"\n{'='*60}")
    print(f"Launched {len(jobs)} jobs")
    print(f"Output:      s3://{BUCKET}/{OUTPUT_PREFIX}/")
    print(f"Checkpoints: s3://{BUCKET}/{CHECKPOINT_PREFIX}/")
    print(f"{'='*60}")

    # Clean up staged source
    shutil.rmtree(staged_dir, ignore_errors=True)
    print(f"Cleaned up staged source: {staged_dir}")


if __name__ == "__main__":
    launch()
