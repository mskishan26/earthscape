#!/usr/bin/env python3
"""
Label EDA Script for EarthScape Dataset
Analyzes label distributions across the entire dataset and each split (train/val/test)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import _get_s3_client, list_sets, read_csv_from_s3
from utils import load_config


def analyze_labels(df, split_name="Overall"):
    """Analyze label distributions for a given dataframe"""

    label_cols = ["af1", "Qal", "Qaf", "Qat", "Qc", "Qca", "Qr"]

    print(f"\n{'=' * 80}")
    print(f"{split_name.upper()} DATASET ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Total patches: {len(df)}")

    # Individual label statistics
    print(f"\n{'-' * 80}")
    print("INDIVIDUAL LABEL STATISTICS")
    print(f"{'-' * 80}")
    print(
        f"{'Label':<10} {'Count':<10} {'Percentage':<12} {'Positive':<10} {'Negative':<10}"
    )
    print(f"{'-' * 80}")

    for label in label_cols:
        count_positive = (df[label] == 1.0).sum()
        count_negative = (df[label] == 0.0).sum()
        percentage = (count_positive / len(df)) * 100
        print(
            f"{label:<10} {len(df):<10} {percentage:>6.2f}%      {count_positive:<10} {count_negative:<10}"
        )

    # Multi-label statistics
    print(f"\n{'-' * 80}")
    print("MULTI-LABEL STATISTICS")
    print(f"{'-' * 80}")

    # Number of labels per patch
    labels_per_patch = df[label_cols].sum(axis=1)
    print(f"Labels per patch (mean): {labels_per_patch.mean():.2f}")
    print(f"Labels per patch (median): {labels_per_patch.median():.2f}")
    print(f"Labels per patch (min): {int(labels_per_patch.min())}")
    print(f"Labels per patch (max): {int(labels_per_patch.max())}")

    # Distribution of number of labels
    print(f"\n{'-' * 80}")
    print("DISTRIBUTION OF NUMBER OF LABELS PER PATCH")
    print(f"{'-' * 80}")
    print(f"{'# Labels':<12} {'Count':<10} {'Percentage':<12}")
    print(f"{'-' * 80}")

    label_count_dist = labels_per_patch.value_counts().sort_index()
    for num_labels, count in label_count_dist.items():
        percentage = (count / len(df)) * 100
        print(f"{int(num_labels):<12} {count:<10} {percentage:>6.2f}%")

    # Label co-occurrence analysis
    print(f"\n{'-' * 80}")
    print("LABEL CO-OCCURRENCE MATRIX (Count)")
    print(f"{'-' * 80}")

    # Create co-occurrence matrix
    cooccurrence = pd.DataFrame(0, index=label_cols, columns=label_cols)
    for i, label1 in enumerate(label_cols):
        for j, label2 in enumerate(label_cols):
            if i <= j:
                cooccurrence.loc[label1, label2] = (
                    (df[label1] == 1.0) & (df[label2] == 1.0)
                ).sum()
                if i != j:
                    cooccurrence.loc[label2, label1] = cooccurrence.loc[label1, label2]

    print(cooccurrence.to_string())

    # Most common label combinations
    print(f"\n{'-' * 80}")
    print("TOP 10 MOST COMMON LABEL COMBINATIONS")
    print(f"{'-' * 80}")

    label_combinations = df[label_cols].apply(lambda row: tuple(row), axis=1)
    combination_counts = label_combinations.value_counts().head(10)

    print(f"{'Rank':<6} {'Count':<10} {'Percentage':<12} {'Label Combination'}")
    print(f"{'-' * 80}")

    for rank, (combination, count) in enumerate(combination_counts.items(), 1):
        percentage = (count / len(df)) * 100
        labels_present = [
            label_cols[i] for i, val in enumerate(combination) if val == 1.0
        ]
        label_str = ", ".join(labels_present) if labels_present else "None"
        print(f"{rank:<6} {count:<10} {percentage:>6.2f}%      {label_str}")

    # Class imbalance metrics
    print(f"\n{'-' * 80}")
    print("CLASS IMBALANCE METRICS")
    print(f"{'-' * 80}")

    for label in label_cols:
        positive = (df[label] == 1.0).sum()
        negative = (df[label] == 0.0).sum()
        if positive > 0:
            imbalance_ratio = negative / positive
            print(f"{label:<10} Imbalance Ratio (neg/pos): {imbalance_ratio:.2f}")
        else:
            print(f"{label:<10} Imbalance Ratio (neg/pos): inf (no positive samples)")


def load_splits_from_s3(bucket, set_prefixes):
    """
    Load and combine split.csv files from multiple S3 locations.
    
    Args:
        bucket: S3 bucket name
        set_prefixes: List of dataset prefixes to load from S3
        
    Returns:
        Combined DataFrame with all splits
    """
    all_dfs = []
    
    for set_prefix in set_prefixes:
        split_key = f"{set_prefix}split.csv"
        try:
            df = read_csv_from_s3(bucket, split_key)
            df["source"] = set_prefix
            all_dfs.append(df)
            print(f"Loaded {len(df)} patches from {set_prefix}")
        except Exception as e:
            print(f"! Could not load {split_key}: {e}")
    
    if not all_dfs:
        print("\nNo split files loaded. Exiting.")
        return None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal patches loaded: {len(combined_df)}")
    return combined_df


def save_plots(df, output_dir="eda_plots"):
    """Generate and save presentation-ready plots to output_dir."""
    label_cols = ["af1", "Qal", "Qaf", "Qat", "Qc", "Qca", "Qr"]
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 1. Label frequency bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = {l: int((df[l] == 1.0).sum()) for l in label_cols}
    bars = ax.bar(freqs.keys(), freqs.values(), color="steelblue", edgecolor="black")
    ax.bar_label(bars, fmt="%d", fontsize=10)
    ax.set_title("Label Frequency (Positive Count)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    fig.tight_layout()
    fig.savefig(out / "label_frequency.png", dpi=200)
    plt.close(fig)

    # 2. Labels-per-patch distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    lpp = df[label_cols].sum(axis=1)
    counts = lpp.value_counts().sort_index()
    bars = ax.bar(counts.index.astype(int).astype(str), counts.values, color="coral", edgecolor="black")
    ax.bar_label(bars, fmt="%d", fontsize=10)
    ax.set_title("Distribution of Labels per Patch")
    ax.set_xlabel("Number of Labels")
    ax.set_ylabel("Patch Count")
    fig.tight_layout()
    fig.savefig(out / "labels_per_patch.png", dpi=200)
    plt.close(fig)

    # 3. Co-occurrence heatmap
    cooc = pd.DataFrame(0, index=label_cols, columns=label_cols)
    for i, l1 in enumerate(label_cols):
        for j, l2 in enumerate(label_cols):
            if i <= j:
                val = int(((df[l1] == 1.0) & (df[l2] == 1.0)).sum())
                cooc.loc[l1, l2] = val
                cooc.loc[l2, l1] = val
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("Label Co-occurrence Matrix")
    fig.tight_layout()
    fig.savefig(out / "cooccurrence_heatmap.png", dpi=200)
    plt.close(fig)

    # 4. Top 10 label combinations (horizontal bar)
    combos = df[label_cols].apply(tuple, axis=1)
    top10 = combos.value_counts().head(10)
    labels_str = [
        ", ".join(label_cols[i] for i, v in enumerate(c) if v == 1.0) or "None"
        for c in top10.index
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels_str[::-1], top10.values[::-1], color="mediumseagreen", edgecolor="black")
    ax.set_title("Top 10 Label Combinations")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(out / "top_combinations.png", dpi=200)
    plt.close(fig)

    # 5. Class imbalance ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ratios = {}
    for l in label_cols:
        pos = (df[l] == 1.0).sum()
        neg = (df[l] == 0.0).sum()
        ratios[l] = neg / pos if pos > 0 else float("inf")
    bars = ax.bar(ratios.keys(), ratios.values(), color="salmon", edgecolor="black")
    ax.bar_label(bars, fmt="%.1f", fontsize=10)
    ax.set_title("Class Imbalance Ratio (Negative / Positive)")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Label")
    fig.tight_layout()
    fig.savefig(out / "imbalance_ratio.png", dpi=200)
    plt.close(fig)

    # 6. Split distribution pie chart
    if "split" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        split_counts = df["split"].value_counts().sort_index()
        ax.pie(
            split_counts.values,
            labels=split_counts.index,
            autopct="%1.1f%%",
            colors=["#66b3ff", "#99ff99", "#ff9999", "#ffcc99"],
            startangle=90,
            textprops={"fontsize": 12},
        )
        ax.set_title("Split Distribution", fontsize=14)
        fig.tight_layout()
        fig.savefig(out / "split_distribution.png", dpi=200)
        plt.close(fig)

    saved = list(out.glob("*.png"))
    print(f"\n>> Saved {len(saved)} plots to {out.resolve()}/")
    for f in sorted(saved):
        print(f"   - {f.name}")


def main():
    """Main function to run the EDA"""
    parser = argparse.ArgumentParser(
        description="Analyze label distributions for EarthScape datasets"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max_sets", type=int, default=None)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local split.csv file instead of S3",
    )
    parser.add_argument("--split_csv", type=str, default="split.csv")
    args = parser.parse_args()

    if args.local:
        # Load the data from local file
        csv_path = Path(__file__).parent / args.split_csv

        if not csv_path.exists():
            print(f"Error: {csv_path} not found!")
            return

        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        # Load from S3
        cfg = load_config(args.config, cli_overrides=[])
        bucket = cfg["data"]["bucket"]
        base_prefixes = cfg["data"].get(
            "base_prefixes", [cfg["data"].get("base_prefix", "")]
        )
        if isinstance(base_prefixes, str):
            base_prefixes = [base_prefixes]

        all_sets = []
        for base_prefix in base_prefixes:
            all_sets.extend(list_sets(bucket, base_prefix))
        if args.max_sets:
            all_sets = all_sets[: args.max_sets]

        print(f"Loading splits from {len(all_sets)} dataset(s)...\n")
        df = load_splits_from_s3(bucket, all_sets)
        
        if df is None:
            return

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Overall dataset analysis
    analyze_labels(df, "Overall")

    # Generate and save plots
    plot_dir = str(Path(__file__).parent / "eda_plots")
    save_plots(df, output_dir=plot_dir)

    # Per-split analysis
    splits = df["split"].unique()
    print(f"\n\nFound splits: {sorted(splits)}")

    for split in sorted(splits):
        split_df = df[df["split"] == split]
        analyze_labels(split_df, split)

    # Split distribution summary
    print(f"\n{'=' * 80}")
    print("SPLIT DISTRIBUTION SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Split':<10} {'Count':<10} {'Percentage':<12}")
    print(f"{'-' * 80}")

    for split in sorted(splits):
        count = (df["split"] == split).sum()
        percentage = (count / len(df)) * 100
        print(f"{split:<10} {count:<10} {percentage:>6.2f}%")

    print(f"\n{'=' * 80}")
    print("EDA COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
