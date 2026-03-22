#!/bin/bash
# Simple bash script to run all experiments sequentially
# For more control (skip completed, dry-run, etc.), use run_experiments.py

set -e  # Exit on error

echo "=========================================="
echo "Running EarthScape Experiments"
echo "=========================================="

for exp in experiments/*.yaml; do
    echo ""
    echo "=========================================="
    echo "Experiment: $(basename $exp)"
    echo "=========================================="
    python train.py --experiment "$exp"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
