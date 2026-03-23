"""
SageMaker Launch Script  (SDK v3 — ModelTrainer API)

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
from pathlib import Path

import yaml
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.shapes.shapes import (
    CheckpointConfig,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train.model_trainer import ModelTrainer


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


def launch():
    args = parse_args()

    if args.experiment:
        experiment_files = [args.experiment]
    else:
        experiment_files = sorted(glob.glob("experiments/*.yaml"))
    if not experiment_files:
        print("No experiment YAMLs found in experiments/")
        return

    session = Session()
    role = get_execution_role()
    region = session.boto_region_name

    # Resolve the PyTorch training image for the target instance
    training_image = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.5",
        py_version="py311",
        instance_type=args.instance_type,
        image_scope="training",
    )

    use_spot = not args.no_spot
    max_run_seconds = args.max_run_hours * 3600
    max_wait_seconds = max_run_seconds + 3600 if use_spot else None
    spot_label = "SPOT" if use_spot else "ON-DEMAND"

    print(f"Found {len(experiment_files)} experiments")
    print(f"Instance: {args.instance_type} ({spot_label})")
    print(f"Image:    {training_image}\n")

    # Shared source code config (uploaded once per job)
    source_code = SourceCode(
        source_dir=".",
        entry_script="train.py",
        requirements="requirements.txt",
        ignore_patterns=[
            ".venv",
            ".git",
            ".ruff_cache",
            "__pycache__",
            "data_cache",
            "outputs",
            "wandb",
            "standalone_scripts",
            "*.html",
            "uv.lock",
        ],
    )

    # Stopping condition — only set max_wait when using spot
    stopping = StoppingCondition(max_runtime_in_seconds=max_run_seconds)
    if use_spot:
        stopping.max_wait_time_in_seconds = max_wait_seconds

    jobs = []
    for exp_file in experiment_files:
        exp_name = get_experiment_name(exp_file)

        try:
            trainer = ModelTrainer(
                sagemaker_session=session,
                role=role,
                training_image=training_image,
                base_job_name=f"es-{exp_name}",
                source_code=source_code,
                compute=Compute(
                    instance_type=args.instance_type,
                    instance_count=1,
                    enable_managed_spot_training=use_spot,
                ),
                stopping_condition=stopping,
                output_data_config=OutputDataConfig(
                    s3_output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/",
                ),
                checkpoint_config=CheckpointConfig(
                    s3_uri=f"s3://{BUCKET}/{CHECKPOINT_PREFIX}/{exp_name}/",
                    local_path="/opt/ml/checkpoints",
                ),
                hyperparameters={"experiment": exp_file},
                environment={"WANDB_API_KEY": args.wandb_api_key},
            )

            trainer.train(wait=False)
            job_name = trainer._latest_training_job.training_job_name
            jobs.append((exp_name, job_name))
            print(f"  Launched: {exp_name:30s} -> {job_name}")

        except Exception as e:
            print(f"  FAILED:   {exp_name:30s} -> {e}")

    print(f"\n{'='*60}")
    print(f"Launched {len(jobs)} jobs")
    print(f"Output:      s3://{BUCKET}/{OUTPUT_PREFIX}/")
    print(f"Checkpoints: s3://{BUCKET}/{CHECKPOINT_PREFIX}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    launch()
