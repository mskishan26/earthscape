"""
Create train/val/test splits for EarthScape datasets.

Run this ONCE before training. Writes split.csv to each set's S3 prefix.
All subsequent training/eval runs read from these split files.

Usage:
    # Split all sets with default ratios (70/15/15)
    python create_splits.py --config config.yaml

    # Custom ratios
    python create_splits.py --config config.yaml --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

    # Only split specific sets (debug)
    python create_splits.py --config config.yaml --max_sets 1

    # Dry run (don't upload to S3)
    python create_splits.py --config config.yaml --dry_run

    # Force re-split even if split.csv already exists
    python create_splits.py --config config.yaml --force
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import _get_s3_client, list_sets, read_csv_from_s3
from utils import load_config


def _read_text_from_s3(bucket: str, key: str) -> str:
    s3 = _get_s3_client()
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8")


def _flatten_coords(coords):
    if not isinstance(coords, list):
        return []
    if coords and isinstance(coords[0], (int, float)):
        return [coords]
    points = []
    for item in coords:
        points.extend(_flatten_coords(item))
    return points


def _feature_centroid(feature: dict):
    geometry = feature.get("geometry") or {}
    points = _flatten_coords(geometry.get("coordinates", []))
    if not points:
        return None
    xs = [p[0] for p in points if len(p) >= 2]
    ys = [p[1] for p in points if len(p) >= 2]
    if not xs or not ys:
        return None
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _load_locations_df(bucket: str, set_prefix: str) -> pd.DataFrame | None:
    locations_key = f"{set_prefix}locations.geojson"
    try:
        geojson = json.loads(_read_text_from_s3(bucket, locations_key))
    except Exception as e:
        print(f"  ! Could not load {locations_key}: {e}")
        return None

    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        patch_id = props.get("patch_id")
        centroid = _feature_centroid(feature)
        if patch_id is None or centroid is None:
            continue
        rows.append({"patch_id": str(patch_id), "cx": centroid[0], "cy": centroid[1]})

    if not rows:
        print("  ! No usable patch geometries found in locations.geojson")
        return None

    return pd.DataFrame(rows)


def _assign_spatial_splits(
    labels_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
):
    merged = labels_df.merge(locations_df, on="patch_id", how="left")
    geo_mask = merged[["cx", "cy"]].notna().all(axis=1)

    if geo_mask.sum() == 0:
        return None

    geo_df = merged.loc[geo_mask].copy()
    x_span = geo_df["cx"].max() - geo_df["cx"].min()
    y_span = geo_df["cy"].max() - geo_df["cy"].min()
    sort_cols = ["cx", "cy"] if x_span >= y_span else ["cy", "cx"]
    geo_df = geo_df.sort_values(sort_cols + ["patch_id"]).reset_index(drop=True)

    n_geo = len(geo_df)
    n_train = int(n_geo * train_ratio)
    n_val = int(n_geo * val_ratio)

    geo_df["split"] = ""
    geo_df.iloc[:n_train, geo_df.columns.get_loc("split")] = "train"
    geo_df.iloc[n_train : n_train + n_val, geo_df.columns.get_loc("split")] = "val"
    geo_df.iloc[n_train + n_val :, geo_df.columns.get_loc("split")] = "test"

    result = merged.drop(columns=["cx", "cy"], errors="ignore")
    result = result.merge(
        geo_df[["patch_id", "split"]], on="patch_id", how="left", suffixes=("", "_geo")
    )
    if "split_geo" in result.columns:
        result["split"] = result["split_geo"]
        result = result.drop(columns=["split_geo"])
    return result


def _assign_fallback_splits(
    labels_df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(labels_df))
    n_train = int(len(labels_df) * train_ratio)
    n_val = int(len(labels_df) * val_ratio)
    labels_df = labels_df.copy()
    labels_df["split"] = ""
    labels_df.iloc[indices[:n_train], labels_df.columns.get_loc("split")] = "train"
    labels_df.iloc[
        indices[n_train : n_train + n_val], labels_df.columns.get_loc("split")
    ] = "val"
    labels_df.iloc[indices[n_train + n_val :], labels_df.columns.get_loc("split")] = (
        "test"
    )
    return labels_df


def check_split_exists(bucket: str, set_prefix: str) -> bool:
    """
    Check if split.csv already exists for a given dataset.
    
    Args:
        bucket: S3 bucket
        set_prefix: dataset prefix
        
    Returns:
        True if split.csv exists, False otherwise
    """
    split_key = f"{set_prefix}split.csv"
    try:
        existing = read_csv_from_s3(bucket, split_key)
        n_train = (existing["split"] == "train").sum()
        n_val = (existing["split"] == "val").sum()
        n_test = (existing["split"] == "test").sum()
        print(
            f"  ✔ split.csv already exists ({n_train}/{n_val}/{n_test}). Skipping."
        )
        return True
    except Exception:
        return False


def create_split(
    bucket: str,
    set_prefix: str,
    label_cols: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    dry_run: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Create a spatial split for one dataset.

    Splits at the patch level to avoid data leakage from
    overlapping patches across splits.

    Args:
        bucket: S3 bucket
        set_prefix: dataset prefix (e.g., "raw-data/warren_county/")
        label_cols: label column names for distribution reporting
        train_ratio: proportion for training
        val_ratio: proportion for validation
        test_ratio: proportion for test
        seed: random seed
        dry_run: if True, compute split but don't upload
        force: if True, overwrite existing split.csv

    Returns:
        DataFrame with split assignments
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    s3 = _get_s3_client()
    split_key = f"{set_prefix}split.csv"

    # Check for existing split
    if not force:
        if check_split_exists(bucket, set_prefix):
            try:
                return read_csv_from_s3(bucket, split_key)
            except Exception:
                pass  # Continue to create new split

    # Load labels
    labels_key = f"{set_prefix}labels.csv"
    try:
        labels_df = read_csv_from_s3(bucket, labels_key)
    except Exception as e:
        print(f"  ✗ Could not load {labels_key}: {e}")
        return None

    labels_df["patch_id"] = labels_df["patch_id"].astype(str)
    n_total = len(labels_df)
    print(f"  Total patches: {n_total}")

    if n_total == 0:
        print("  ✗ No patches found, skipping")
        return None

    locations_df = _load_locations_df(bucket, set_prefix)
    if locations_df is not None:
        split_df = _assign_spatial_splits(
            labels_df, locations_df, train_ratio, val_ratio
        )
        if split_df is not None and split_df["split"].notna().all():
            labels_df = split_df
            print(f"  Spatial split source: {set_prefix}locations.geojson")
        else:
            print(
                "  ! Falling back to random split because locations could not be joined"
            )
            labels_df = _assign_fallback_splits(labels_df, train_ratio, val_ratio, seed)
    else:
        print(
            "  ! Falling back to random split because locations.geojson is unavailable"
        )
        labels_df = _assign_fallback_splits(labels_df, train_ratio, val_ratio, seed)

    train_df = labels_df[labels_df["split"] == "train"]
    val_df = labels_df[labels_df["split"] == "val"]
    test_df = labels_df[labels_df["split"] == "test"]

    print(f"  Train: {len(train_df)} ({len(train_df) / n_total:.1%})")
    print(f"  Val:   {len(val_df)} ({len(val_df) / n_total:.1%})")
    print(f"  Test:  {len(test_df)} ({len(test_df) / n_total:.1%})")

    # Class distribution per split
    available_cols = [c for c in label_cols if c in labels_df.columns]
    if available_cols:
        print(f"\n  {'Class':<8} {'Train':>8} {'Val':>8} {'Test':>8}")
        print(f"  {'-' * 34}")
        for col in available_cols:
            t_pct = (train_df[col] > 0).mean() * 100
            v_pct = (val_df[col] > 0).mean() * 100
            te_pct = (test_df[col] > 0).mean() * 100
            print(f"  {col:<8} {t_pct:>7.1f}% {v_pct:>7.1f}% {te_pct:>7.1f}%")

    if not dry_run:
        csv_buffer = io.StringIO()
        labels_df.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=split_key,
            Body=csv_buffer.getvalue(),
        )
        print(f"\n  ✔ Uploaded to s3://{bucket}/{split_key}")
    else:
        print(f"\n  [DRY RUN] Would upload to s3://{bucket}/{split_key}")

    return labels_df


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for EarthScape"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_sets", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Don't upload to S3")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing splits"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, cli_overrides=[])

    bucket = cfg["data"]["bucket"]
    base_prefixes = cfg["data"].get(
        "base_prefixes", [cfg["data"].get("base_prefix", "")]
    )
    if isinstance(base_prefixes, str):
        base_prefixes = [base_prefixes]
    label_cols = cfg["data"]["label_cols"]
    seed = cfg["seed"]

    all_sets = []
    for base_prefix in base_prefixes:
        all_sets.extend(list_sets(bucket, base_prefix))
    if args.max_sets:
        all_sets = all_sets[: args.max_sets]

    print(f"{'=' * 60}")
    print(f"Creating splits for {len(all_sets)} dataset(s)")
    print(f"Ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"Seed: {seed}")
    print(f"{'=' * 60}")

    for set_prefix in all_sets:
        print(f"\n{set_prefix}")
        
        # Check if split already exists (unless force is True)
        if not args.force and check_split_exists(bucket, set_prefix):
            continue
            
        create_split(
            bucket=bucket,
            set_prefix=set_prefix,
            label_cols=label_cols,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=seed,
            dry_run=args.dry_run,
            force=args.force,
        )

    print(f"\n{'=' * 60}")
    print("Done. You can now run training:")
    print("  python train.py --config config.yaml")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
