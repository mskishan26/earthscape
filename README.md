# EarthScape

Multi-modal deep learning for geological mapping using satellite imagery and topographic data.

## Overview

EarthScape is a PyTorch-based framework for multi-label geological classification that combines:
- **RGB satellite imagery** (Sentinel-2)
- **Spectral bands** (SWIR, NIR, coastal aerosol, water vapor)
- **Topographic features** (elevation, slope, aspect, curvature)

The framework supports multiple fusion architectures and is designed for training on AWS SageMaker with S3-based data storage.

## Features

- **Multi-modal fusion architectures**: Early, mid, and late fusion strategies
- **Flexible backbone models**: ResNet50, EfficientNet-B0, and custom architectures
- **S3-native data pipeline**: Efficient streaming from S3 with local caching
- **Spatial-aware splitting**: Geographic-based train/val/test splits to prevent data leakage
- **Multi-label classification**: Handles 7 geological classes with class imbalance handling
- **Production-ready**: Supports both local development and SageMaker training

## Project Structure

```
earthscape/
├── config.yaml                 # Base configuration
├── experiments/                # Experiment-specific configs
│   ├── midfusion_all.yaml
│   ├── midfusion_topo_only.yaml
│   ├── resnet50_rgb.yaml
│   └── efficientnet_rgb.yaml
├── models/                     # Model architectures
│   ├── midfusion.py           # Mid-fusion architecture
│   └── rgb_backbone.py        # RGB-only backbones
├── standalone_scripts/         # Utility scripts
│   ├── create_splits.py       # Create train/val/test splits
│   ├── map_patches.py         # Visualize patches on map
│   └── label_eda.py           # Label distribution analysis
├── dataset.py                 # S3-based dataset loader
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── metrics.py                 # Evaluation metrics
└── utils.py                   # Utility functions
```

## Dataset

The dataset consists of geological patches from multiple regions, stored in S3:
- **Total patches**: 7,714 (across multiple datasets)
- **Image size**: 256×256 pixels
- **Classes**: 7 geological formations (af1, Qal, Qaf, Qat, Qc, Qca, Qr)
- **Split**: 70% train / 15% val / 15% test (spatially aware)

### Label Distribution

```
Class    Train     Val      Test     Overall
af1      28.5%    28.7%    28.5%    28.5%
Qal      65.9%    56.1%    66.4%    64.2%
Qaf       0.5%     0.5%     1.8%     0.8%
Qat       4.2%     2.3%     0.3%     3.3%
Qc       51.3%    68.6%    46.9%    52.8%
Qca      42.4%    25.3%    18.6%    35.9%
Qr       93.2%    96.6%    99.7%    95.3%
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd earthscape

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### 1. Create Train/Val/Test Splits

Run this once to create spatial splits for all datasets:

```bash
python standalone_scripts/create_splits.py --config config.yaml
```

This will:
- Check each dataset for existing `split.csv`
- Skip datasets that already have splits
- Create spatially-aware splits for new datasets
- Upload splits to S3

Options:
- `--force`: Overwrite existing splits
- `--dry_run`: Preview splits without uploading
- `--max_sets N`: Only process first N datasets

### 2. Visualize Patches

Create an interactive map showing all patches with split colors:

```bash
python standalone_scripts/map_patches.py --config config.yaml
```

This generates `interactive_patch_map.html` combining all datasets.

### 3. Analyze Label Distribution

Run exploratory data analysis on the combined dataset:

```bash
python standalone_scripts/label_eda.py --config config.yaml
```

This shows:
- Label frequencies and co-occurrence
- Class imbalance metrics
- Multi-label statistics
- Per-split distributions

### 4. Train a Model

```bash
# Train with default config (mid-fusion, all modalities)
python train.py --config config.yaml

# Train RGB-only ResNet50
python train.py --config config.yaml --experiment experiments/resnet50_rgb.yaml

# Train with custom learning rate
python train.py --config config.yaml --training.lr 3e-4
```

### 5. Evaluate

```bash
python evaluate.py --config config.yaml --checkpoint outputs/checkpoints/best.pt
```

## Configuration

The framework uses hierarchical YAML configs:

- **`config.yaml`**: Base infrastructure config (data paths, training params, etc.)
- **`experiments/*.yaml`**: Model-specific overrides (architecture, modalities, etc.)

Key config sections:

```yaml
data:
  bucket: "earthscape-dataset"
  base_prefixes:
    - "earthscape_data/"
    - "earthscape_data_v2/"
  label_cols: ["af1", "Qal", "Qaf", "Qat", "Qc", "Qca", "Qr"]
  img_size: 256

training:
  batch_size: 32
  num_epochs: 10
  lr: 1.0e-4
  num_workers: 8

model:
  architecture: "midfusion"  # or "resnet50", "efficientnet"
  num_classes: 7
  dropout: 0.3
```

## Standalone Scripts

All scripts in `standalone_scripts/` support both S3 and local file modes:

### Create Splits
```bash
# From S3 (default)
python standalone_scripts/create_splits.py --config config.yaml

# Custom split ratios
python standalone_scripts/create_splits.py --config config.yaml \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### Map Patches
```bash
# From S3 (combines all datasets)
python standalone_scripts/map_patches.py --config config.yaml

# From local files
python standalone_scripts/map_patches.py --local \
    --geojson_file locations.geojson --split_csv_file split.csv
```

### Label EDA
```bash
# From S3 (combines all datasets)
python standalone_scripts/label_eda.py --config config.yaml

# From local file
python standalone_scripts/label_eda.py --local --split_csv split.csv
```

## AWS SageMaker

Launch training on SageMaker:

```bash
python sagemaker_launch.py --experiment experiments/midfusion_all.yaml
```

The framework automatically:
- Detects SageMaker environment
- Uses `/opt/ml/input/data_cache` for caching
- Saves checkpoints to `/opt/ml/model`
- Streams data efficiently from S3

## Model Architectures

### Mid-Fusion (Default)
Fuses RGB, spectral, and topographic features at intermediate layers.

### RGB-Only Backbones
- **ResNet50**: Standard ImageNet-pretrained backbone
- **EfficientNet-B0**: Efficient architecture with compound scaling

## Metrics

The framework tracks:
- **Per-class metrics**: Precision, recall, F1-score
- **Multi-label metrics**: Hamming loss, subset accuracy
- **Macro/micro averages**: Overall performance
- **Confusion matrices**: Per-class analysis

## License

[Add your license here]

## Citation

[Add citation information if applicable]
