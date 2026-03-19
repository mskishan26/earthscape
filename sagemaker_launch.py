"""
SageMaker Launch Script

This is a placeholder — we'll flesh this out together once the training
code is validated locally on your RTX 4060.

Key things to configure:
  - output_path: where model artifacts go in S3
  - checkpoint_s3_uri: continuous checkpoint sync
  - hyperparameters: flow through to config overrides
  - instance_type: ml.g4dn.xlarge (T4, 16GB) or ml.g5.xlarge (A10G, 24GB)
"""

import sagemaker
from sagemaker.pytorch import PyTorch


def launch():
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    bucket = "deeplearning-midterm-data"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="./resnet_midfusion",
        role=role,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",          # Updated from 1.9!
        py_version="py310",               # Updated from py38!
        base_job_name="earthscape-midfusion",
        
        # --- Critical: S3 output path ---
        output_path=f"s3://{bucket}/model-artifacts/",
        
        # --- Continuous checkpoint sync ---
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/",
        checkpoint_local_path="/opt/ml/checkpoints",
        
        # --- Hyperparameters (override config.yaml values) ---
        hyperparameters={
            "training.num_epochs": 10,
            "training.batch_size": 32,
            "training.lr": "1e-4",
            "amp.enabled": "true",
            "logging.use_wandb": "true",
            "data.max_sets": "null",  # Use all sets
        },

        # --- Environment variables ---
        environment={
            "WANDB_API_KEY": "YOUR_KEY_HERE",  # Or use SM secrets
        },

        # --- Metric definitions for SM console ---
        metric_definitions=[
            {"Name": "train:loss", "Regex": r"Train — Loss: ([0-9.]+)"},
            {"Name": "val:loss", "Regex": r"Val   — Loss: ([0-9.]+)"},
            {"Name": "val:macro_f1", "Regex": r"Val   — .* Macro-F1: ([0-9.]+)"},
            {"Name": "test:macro_f1", "Regex": r"Test Macro-F1: ([0-9.]+)"},
        ],

        # Max runtime (seconds) - 24 hours
        max_run=86400,
    )

    print("Launching training job...")
    estimator.fit(wait=False)  # Don't block - monitor in SM console
    print(f"Job name: {estimator.latest_training_job.name}")
    print(f"Output: s3://{bucket}/model-artifacts/")


if __name__ == "__main__":
    launch()
