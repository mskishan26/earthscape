"""
EarthScape Dataset with simple local disk caching.

Flow:
  1. Index phase: list S3 keys, build sample list (patch_id -> modality -> s3_key)
  2. __getitem__: reads from local cache (fast) or downloads from S3 and caches (first access)
  3. Subsequent epochs: all reads from cache (no S3 access)

PyTorch DataLoader with num_workers handles parallel loading and prefetching.
"""

import hashlib
import io
import threading
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# S3 Helpers
# ============================================================================

_thread_local = threading.local()


def _get_s3_client():
    """Thread-local S3 client — safe for DataLoader workers + prefetch threads."""
    if not hasattr(_thread_local, "s3_client"):
        _thread_local.s3_client = boto3.client("s3")
    return _thread_local.s3_client


def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = _get_s3_client()
    resp = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(resp["Body"].read()))


def list_patch_images(bucket: str, set_prefix: str):
    """Yield all .tif keys under {set_prefix}patches/."""
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{set_prefix}patches/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".tif"):
                yield key


def list_sets(bucket: str, root_prefix: str) -> list:
    """List dataset folders under root_prefix."""
    s3 = _get_s3_client()
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=root_prefix, MaxKeys=50)
        keys = {obj["Key"] for obj in resp.get("Contents", [])}
        has_root_dataset = (
            f"{root_prefix}labels.csv" in keys
            or f"{root_prefix}locations.geojson" in keys
        )
    except Exception:
        has_root_dataset = False

    paginator = s3.get_paginator("list_objects_v2")
    sets = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            candidate = prefix["Prefix"]
            if candidate == f"{root_prefix}patches/":
                continue
            sets.add(candidate)

    if has_root_dataset:
        sets.add(root_prefix)

    return sorted(sets)


def extract_patch_id(filename: str) -> str:
    """Extract patch_id from filename, handling both naming formats."""
    base = filename.split("/")[-1].replace(".tif", "")
    parts = base.split("_")
    try:
        int(parts[0])
        return "_".join(parts[:3])
    except ValueError:
        return "_".join(parts[:4])


# ============================================================================
# Simple Persistent Cache
# ============================================================================


class SimpleCache:
    """
    Simple persistent disk cache for S3 objects.

    Files are downloaded once and kept forever (no eviction).
    Thread-safe for use with PyTorch DataLoader workers.

    Files stored as: {cache_dir}/{s3_key_hash}_{filename}
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._download_in_progress: dict[str, threading.Event] = {}  # Track ongoing downloads

        # Report existing cache
        existing = list(self.cache_dir.rglob("*.tif"))
        if existing:
            total_bytes = sum(f.stat().st_size for f in existing)
            print(
                f"[Cache] Found {len(existing)} cached files ({total_bytes / 1e9:.1f} GB)"
            )

    def _cache_path(self, s3_key: str) -> Path:
        """Deterministic local path for an S3 key."""
        key_hash = hashlib.md5(s3_key.encode()).hexdigest()[:8]
        filename = s3_key.split("/")[-1]
        return self.cache_dir / f"{key_hash}_{filename}"

    def get_or_download(self, s3_key: str, bucket: str) -> Path:
        """Get file from cache, or download if not cached. Thread-safe."""
        local_path = self._cache_path(s3_key)

        # Fast path: already cached
        if local_path.exists():
            return local_path

        # Slow path: need to download
        with self._lock:
            if local_path.exists():
                return local_path

            # Another thread is already downloading this key — wait for it
            if s3_key in self._download_in_progress:
                event = self._download_in_progress[s3_key]
                self._lock.release()
                event.wait()
                self._lock.acquire()
                if local_path.exists():
                    return local_path

            # Mark as in-progress
            self._download_in_progress[s3_key] = threading.Event()

        # Download outside the lock (allow other threads to proceed)
        tmp_path = local_path.with_suffix(".tmp")
        try:
            s3 = _get_s3_client()
            s3.download_file(bucket, s3_key, str(tmp_path))
            tmp_path.rename(local_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        finally:
            with self._lock:
                event = self._download_in_progress.pop(s3_key, None)
                if event:
                    event.set()

        return local_path

    @property
    def size_gb(self) -> float:
        """Current cache size in GB."""
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.rglob("*.tif"))
        return total_bytes / 1e9


# ============================================================================
# Core Dataset
# ============================================================================


class CachedEarthScapeDataset(Dataset):
    """
    EarthScape dataset with simple local caching.

    On __getitem__:
      1. Try to read from local cache (fast, ~1ms per file)
      2. If not cached -> download from S3 and cache (slow, first access only)
      3. Subsequent accesses read from cache

    PyTorch DataLoader with num_workers handles parallel loading.
    """

    def __init__(
        self,
        bucket: str,
        set_prefixes: list[str],
        modalities: list[str],
        split: str,
        label_cols: list[str],
        cache: SimpleCache,
    ):
        self.bucket = bucket
        self.modalities = modalities
        self.split = split
        self.label_cols = label_cols
        self.cache = cache

        # Build sample index
        self.samples: list[
            tuple[str, str, np.ndarray]
        ] = []  # (set_prefix, patch_id, labels)
        self.patch_to_keys: dict[tuple[str, str], dict[str, str]] = {}

        for sp in set_prefixes:
            self._index_set(sp)

        print(
            f"[Dataset] {split}: {len(self.samples)} samples across {len(set_prefixes)} set(s)"
        )

    def _index_set(self, set_prefix: str):
        """Build index for one set: which patches exist, which modalities, labels."""
        print(f"  Indexing {set_prefix} ({self.split})...")

        # Load split info
        try:
            split_df = read_csv_from_s3(self.bucket, f"{set_prefix}split.csv")
        except Exception:
            print("  ⚠ No split.csv, falling back to labels.csv (all → train)")
            split_df = read_csv_from_s3(self.bucket, f"{set_prefix}labels.csv")
            split_df["split"] = "train"

        split_df["patch_id"] = split_df["patch_id"].astype(str)
        split_df = split_df[split_df["split"] == self.split]
        if len(split_df) == 0:
            print(f"  ⚠ No patches for {self.split} in {set_prefix}")
            return

        valid_ids = set(split_df["patch_id"].values)
        patch_mods: dict[str, dict[str, str]] = {}

        for key in list_patch_images(self.bucket, set_prefix):
            pid = extract_patch_id(key)
            if pid not in valid_ids:
                continue
            if pid not in patch_mods:
                patch_mods[pid] = {}
            for mod in self.modalities:
                if f"_{mod}.tif" in key:
                    patch_mods[pid][mod] = key
                    break

        # Keep only patches with ALL modalities
        count = 0
        for pid, mods in patch_mods.items():
            if all(m in mods for m in self.modalities):
                self.patch_to_keys[(set_prefix, pid)] = mods
                row = split_df[split_df["patch_id"] == pid]
                labels = row[self.label_cols].iloc[0].values.astype(np.float32)
                self.samples.append((set_prefix, pid, labels))
                count += 1

        print(f"  ✔ {count} valid patches in {set_prefix}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        set_prefix, patch_id, labels = self.samples[idx]
        mod_keys = self.patch_to_keys[(set_prefix, patch_id)]

        modalities_dict = {}
        for mod in self.modalities:
            s3_key = mod_keys[mod]

            # Get from cache or download
            local_path = self.cache.get_or_download(s3_key, self.bucket)

            # Read from local disk
            image = Image.open(local_path)
            arr = np.array(image, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]  # [1, H, W]
            else:
                arr = np.transpose(arr, (2, 0, 1))  # [C, H, W]

            modalities_dict[mod] = torch.from_numpy(arr)

        labels_tensor = torch.from_numpy(labels)
        return modalities_dict, labels_tensor, f"{set_prefix}{patch_id}"


# ============================================================================
# Adapter: groups modalities into spectral + topo for the model
# ============================================================================


class EarthscapePatchAdapter(Dataset):
    """
    Wraps CachedEarthScapeDataset and groups modalities into either:
    - 'full' mode: spectral (4ch) and topo (7-8ch) tensors
    - 'rgb' mode: single RGB tensor (3ch)

    Both modes support normalization with appropriate stats.
    """

    def __init__(
        self,
        base_dataset: CachedEarthScapeDataset,
        spectral_modalities: list[str] | None = None,
        topo_modalities: list[str] | None = None,
        channel_stats: dict | None = None,
        mode: str = "full",  # "full" or "rgb"
    ):
        self.base = base_dataset
        self.mode = mode
        self.spectral_mods = spectral_modalities or []
        self.topo_mods = topo_modalities or []

        # Pre-compute normalization tensors ONCE (not per __getitem__)
        self.spec_mean = None
        self.spec_std = None
        self.topo_mean = None
        self.topo_std = None
        self.rgb_mean = None
        self.rgb_std = None

        if channel_stats is not None:
            if mode == "full":
                self.spec_mean = torch.tensor(
                    channel_stats["spectral_mean"], dtype=torch.float32
                ).view(-1, 1, 1)
                self.spec_std = torch.tensor(
                    channel_stats["spectral_std"], dtype=torch.float32
                ).view(-1, 1, 1)
                self.topo_mean = torch.tensor(
                    channel_stats["topo_mean"], dtype=torch.float32
                ).view(-1, 1, 1)
                self.topo_std = torch.tensor(
                    channel_stats["topo_std"], dtype=torch.float32
                ).view(-1, 1, 1)
            elif mode == "rgb":
                # Accept either rgb_mean/rgb_std or spectral_mean/spectral_std
                mean_key = (
                    "rgb_mean" if "rgb_mean" in channel_stats else "spectral_mean"
                )
                std_key = "rgb_std" if "rgb_std" in channel_stats else "spectral_std"
                self.rgb_mean = torch.tensor(
                    channel_stats[mean_key], dtype=torch.float32
                ).view(-1, 1, 1)
                self.rgb_std = torch.tensor(
                    channel_stats[std_key], dtype=torch.float32
                ).view(-1, 1, 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        modalities_dict, labels, identifier = self.base[idx]

        if self.mode == "rgb":
            return self._getitem_rgb(modalities_dict, labels)
        else:
            return self._getitem_full(modalities_dict, labels)

    def _getitem_rgb(self, modalities_dict, labels):
        """RGB mode: return single [C, H, W] tensor."""
        if "RGB" in modalities_dict:
            rgb = modalities_dict["RGB"]  # [3, H, W]
        else:
            # Stack individual spectral channels into image tensor
            channels = [modalities_dict[mod].squeeze(0) for mod in self.spectral_mods]
            rgb = torch.stack(channels, dim=0)  # [C, H, W]

        # Normalize if stats provided
        if self.rgb_mean is not None:
            rgb = (rgb - self.rgb_mean) / (self.rgb_std + 1e-9)

        return rgb, labels

    def _getitem_full(self, modalities_dict, labels):
        """Full mode: return spectral + topo dict."""
        # Stack spectral channels
        spec_channels = []
        for mod in self.spectral_mods:
            if mod in modalities_dict:
                spec_channels.append(modalities_dict[mod].squeeze(0))
            else:
                h, w = next(iter(modalities_dict.values())).shape[-2:]
                spec_channels.append(torch.zeros(h, w))

        # Stack topo channels
        topo_channels = []
        for mod in self.topo_mods:
            if mod in modalities_dict:
                topo_channels.append(modalities_dict[mod].squeeze(0))
            else:
                h, w = next(iter(modalities_dict.values())).shape[-2:]
                topo_channels.append(torch.zeros(h, w))

        spec = torch.stack(spec_channels, dim=0)  # [4, H, W]
        topo = torch.stack(topo_channels, dim=0)  # [7-8, H, W]

        # Normalize (tensors pre-computed in __init__, no allocation here)
        if self.spec_mean is not None:
            spec = (spec - self.spec_mean) / (self.spec_std + 1e-9)
            topo = (topo - self.topo_mean) / (self.topo_std + 1e-9)

        return {"spectral": spec, "topo": topo}, labels


# ============================================================================
# Stats Computation (from base dataset, memory-efficient)
# ============================================================================


def compute_channel_stats(
    base_dataset: CachedEarthScapeDataset,
    spectral_mods: list[str],
    topo_mods: list[str],
    n_batches: int = 500,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Compute per-channel mean/std using Welford's online algorithm.
    Runs on CPU, memory-efficient.
    """
    loader = DataLoader(
        base_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    n_spec = len(spectral_mods)
    n_topo = len(topo_mods)

    spec_mean, spec_m2 = torch.zeros(n_spec), torch.zeros(n_spec)
    topo_mean, topo_m2 = torch.zeros(n_topo), torch.zeros(n_topo)
    total_spec_px, total_topo_px = 0, 0

    total_batches = min(n_batches, len(loader))
    print(f"[Stats] Computing channel stats from {total_batches} batches...")

    for i, (mods_batch, _, _) in enumerate(
        tqdm(loader, total=total_batches, desc="Computing stats")
    ):
        if i >= n_batches:
            break

        # Spectral
        spec_list = [mods_batch[m].squeeze(1) for m in spectral_mods if m in mods_batch]
        if spec_list:
            spec_batch = torch.stack(spec_list, dim=1)  # [B, C, H, W]
            px = spec_batch.shape[0] * spec_batch.shape[2] * spec_batch.shape[3]
            bm = spec_batch.mean(dim=[0, 2, 3])
            bv = spec_batch.var(dim=[0, 2, 3], unbiased=False)
            delta = bm - spec_mean
            spec_mean += delta * px / (total_spec_px + px)
            spec_m2 += bv * px + delta**2 * total_spec_px * px / (total_spec_px + px)
            total_spec_px += px
            del spec_batch

        # Topo
        topo_list = [mods_batch[m].squeeze(1) for m in topo_mods if m in mods_batch]
        if topo_list:
            topo_batch = torch.stack(topo_list, dim=1)
            px = topo_batch.shape[0] * topo_batch.shape[2] * topo_batch.shape[3]
            bm = topo_batch.mean(dim=[0, 2, 3])
            bv = topo_batch.var(dim=[0, 2, 3], unbiased=False)
            delta = bm - topo_mean
            topo_mean += delta * px / (total_topo_px + px)
            topo_m2 += bv * px + delta**2 * total_topo_px * px / (total_topo_px + px)
            total_topo_px += px
            del topo_batch

        del mods_batch

    stats = {
        "spectral_mean": spec_mean.numpy(),
        "spectral_std": torch.sqrt(spec_m2 / max(total_spec_px, 1)).numpy(),
        "topo_mean": topo_mean.numpy(),
        "topo_std": torch.sqrt(topo_m2 / max(total_topo_px, 1)).numpy(),
    }

    print(
        f"[Stats] Spectral mean={stats['spectral_mean']}, std={stats['spectral_std']}"
    )
    if n_topo > 0:
        print(f"[Stats] Topo mean={stats['topo_mean']}, std={stats['topo_std']}")
    else:
        print("[Stats] No topo channels (RGB mode)")
    return stats


def compute_pos_weights(
    adapted_dataset: EarthscapePatchAdapter,
    max_batches: int = 500,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from label distribution."""
    loader = DataLoader(
        adapted_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    num_classes = None
    pos_counts = None
    total = 0

    total_batches = min(max_batches, len(loader))
    print(f"[PosWeight] Sampling {total_batches} batches...")

    for i, (_, labels) in enumerate(
        tqdm(loader, total=total_batches, desc="Computing pos_weights")
    ):
        if i >= max_batches:
            break
        batch_labels = labels.numpy()
        if pos_counts is None:
            num_classes = batch_labels.shape[1]
            pos_counts = np.zeros(num_classes, dtype=np.float64)
        pos_counts += batch_labels.sum(axis=0)
        total += batch_labels.shape[0]

    neg_counts = total - pos_counts
    weights = (neg_counts / (pos_counts + 1e-9)).astype(np.float32)

    print(f"[PosWeight] Samples: {total}, pos_weight: {weights}")
    return torch.from_numpy(weights).to(device)
