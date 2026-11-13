"""Chactun tiles: indexing, extraction, splits, normalization and the tf.data pipeline.

Dataset facts this module relies on (Kokalj et al. 2023, Scientific Data 10:558):

* Each record is a 240 m x 240 m tile at 0.5 m/px: tile_<id>_lidar.tif is 480x480x3
  uint8 (sky-view factor, positive openness, slope, each scaled to 0-255), and
  tile_<id>_mask_<class>.tif is 480x480 uint8 with 0 = feature, 255 = background.
* Tiles 0-1764 were released for the ECML PKDD 2021 challenge; tiles 1765-2093 were
  the challenge test set. Tile ids are random and geolocation was removed on purpose,
  so a spatial split is not possible.

Everything up to `make_dataset` is numpy-only so it can be tested without TensorFlow.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

CLASSES = ("building", "platform", "aguada")
TILE_RE = re.compile(r"tile_(\d+)_(lidar|mask_building|mask_platform|mask_aguada)\.tiff?$", re.IGNORECASE)
OFFICIAL_TEST_IDS = range(1765, 2094)
N_TILES = 2094


@dataclass
class Tile:
    tile_id: int
    lidar: Path | None = None
    masks: dict[str, Path] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.lidar is not None and all(c in self.masks for c in CLASSES)


# ---------------------------------------------------------------- extraction

def _is_tile_member(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return not base.startswith("._") and TILE_RE.match(base) is not None


def _extract_tiles(z: zipfile.ZipFile, dest: Path) -> int:
    """Copy lidar and mask tiles out of an open zip into dest/lidar and dest/masks."""
    n = 0
    for info in z.infolist():
        if info.is_dir() or not _is_tile_member(info.filename):
            continue
        base = info.filename.rsplit("/", 1)[-1]
        sub = "lidar" if "_lidar." in base.lower() else "masks"
        out = dest / sub / base
        if out.exists() and out.stat().st_size == info.file_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        n += 1
    return n


def extract_chactun(zip_path: str | Path, dest: str | Path) -> None:
    """Extract lidar tiles and masks from either the Figshare bundle (22202395.zip,
    which holds Chactun_ML_ready_lidar.zip and Chactun_ML_ready_masks.zip) or any
    zip that contains the tile files directly. Files already extracted are skipped."""
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as outer:
        inner = [n for n in outer.namelist()
                 if n.endswith(("_lidar.zip", "_masks.zip")) and not n.rsplit("/", 1)[-1].startswith("._")]
        n = 0
        for name in inner:
            with outer.open(name) as f, zipfile.ZipFile(f) as z:
                n += _extract_tiles(z, dest)
        n += _extract_tiles(outer, dest)
    print(f"extracted {n} new files into {dest}")


def ensure_extracted(zip_path: str | Path | None, dest: str | Path) -> None:
    """Extract only if dest does not already hold every tile."""
    dest = Path(dest)
    if dest.exists() and sum(t.complete for t in index_tiles(dest, strict=False)) >= N_TILES:
        return
    if zip_path is None or not Path(zip_path).exists():
        raise FileNotFoundError(f"No complete tile set in {dest} and no zip at {zip_path}")
    extract_chactun(zip_path, dest)


# ---------------------------------------------------------------- indexing and reading

def index_tiles(root: str | Path, strict: bool = True) -> list[Tile]:
    """Group the tif files under root into tiles by id. Skips macOS AppleDouble files."""
    tiles: dict[int, Tile] = {}
    for p in sorted(Path(root).rglob("*.tif*")):
        if p.name.startswith("._"):
            continue
        m = TILE_RE.match(p.name)
        if not m:
            continue
        tid, role = int(m.group(1)), m.group(2).lower()
        t = tiles.setdefault(tid, Tile(tid))
        if role == "lidar":
            t.lidar = p
        else:
            t.masks[role.removeprefix("mask_")] = p
    if strict and not tiles:
        raise FileNotFoundError(f"No tile_<id>_*.tif files under {root}")
    return [tiles[k] for k in sorted(tiles)]


def read_tif(path: Path) -> np.ndarray:
    """Read a tif as (H, W, C) float32; C = 1 for single-band files."""
    a = tifffile.imread(path)
    if a.ndim == 2:
        a = a[..., None]
    elif a.ndim == 3 and a.shape[0] <= 4 and a.shape[0] < a.shape[-1]:
        a = np.moveaxis(a, 0, -1)
    return a.astype(np.float32)


def detect_mask_present_value(tiles: list[Tile], n: int = 100, seed: int = 0) -> int:
    """Masks hold two values, 0 and 255. Features cover a small fraction of the
    pixels, so whichever value is the minority over many tiles marks a feature."""
    rng = np.random.default_rng(seed)
    pick = [tiles[i] for i in rng.choice(len(tiles), min(n, len(tiles)), replace=False)]
    zeros = total = 0
    for t in pick:
        for p in t.masks.values():
            m = read_tif(p)
            zeros += int((m == 0).sum())
            total += m.size
    return 0 if zeros / max(total, 1) < 0.5 else 255


def load_mask(tile: Tile, present_value: int) -> np.ndarray:
    """(H, W, 3) float32 in {0, 1}, channels in CLASSES order."""
    return np.stack([read_tif(tile.masks[c])[..., 0] == present_value for c in CLASSES], -1).astype(np.float32)


# ---------------------------------------------------------------- splits

def official_split(tiles: list[Tile], val_frac: float = 0.15, seed: int = 42):
    """Challenge split: tiles 1765-2093 are the test set. The remaining tiles are
    split at random into train and validation."""
    test_ids = set(OFFICIAL_TEST_IDS)
    pool = [t for t in tiles if t.tile_id not in test_ids]
    test = [t for t in tiles if t.tile_id in test_ids]
    idx = np.random.default_rng(seed).permutation(len(pool))
    nv = int(round(val_frac * len(pool)))
    return [pool[i] for i in idx[nv:]], [pool[i] for i in idx[:nv]], test


# ---------------------------------------------------------------- normalization and counts

def channel_stats(tiles: list[Tile], n: int | None = None, seed: int = 0) -> dict:
    """Per-channel mean and standard deviation of the lidar tiles (all of them, or
    a random subset of n). Compute this on the training split only."""
    if n is not None and n < len(tiles):
        rng = np.random.default_rng(seed)
        tiles = [tiles[i] for i in rng.choice(len(tiles), n, replace=False)]
    s = s2 = cnt = 0
    for t in tiles:
        a = read_tif(t.lidar)
        x = a.reshape(-1, a.shape[-1]).astype(np.float64)
        s, s2, cnt = s + x.sum(0), s2 + (x ** 2).sum(0), cnt + len(x)
    mean = s / cnt
    return {"mean": mean.tolist(), "std": np.sqrt(s2 / cnt - mean ** 2).tolist(), "n_tiles": len(tiles)}


def class_counts(tiles: list[Tile], present_value: int) -> dict:
    """Per class: fraction of all pixels labeled, and number of tiles containing it."""
    pix, has, n = np.zeros(len(CLASSES)), np.zeros(len(CLASSES), int), 0
    for t in tiles:
        y = load_mask(t, present_value)
        pix += y.sum((0, 1))
        has += y.any((0, 1))
        n += y.shape[0] * y.shape[1]
    return {"pixel_fraction": dict(zip(CLASSES, (pix / n).tolist())),
            "tiles_with_class": dict(zip(CLASSES, has.tolist())),
            "n_tiles": len(tiles)}


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- tf.data

def make_dataset(tiles: list[Tile], stats: dict, present_value: int, batch_size: int = 8,
                 augment: bool = False, shuffle: bool = False, cache: bool = False, seed: int | None = None):
    """Batches of (standardized lidar, binary masks). Augmentation is limited to
    90-degree rotations and flips, which permute pixels without interpolating them.

    Tiles are read and, with cache=True, held in memory as stored uint8 (about
    1.4 MB per tile with its masks); standardization to float32 happens after the
    cache, so the cached and uncached pipelines give identical batches."""
    import tensorflow as tf

    mean = tf.constant(stats["mean"], tf.float32)
    std = tf.constant(stats["std"], tf.float32)
    by_path = {str(t.lidar): t for t in tiles}

    def _load(p):
        t = by_path[p.decode()]
        x = tifffile.imread(t.lidar)
        y = np.stack([tifffile.imread(t.masks[c]) == present_value for c in CLASSES], -1)
        return x.astype(np.uint8), y.astype(np.uint8)

    def _tf_load(p):
        x, y = tf.numpy_function(_load, [p], [tf.uint8, tf.uint8])
        x.set_shape([None, None, 3])
        y.set_shape([None, None, len(CLASSES)])
        return x, y

    def _standardize(x, y):
        return (tf.cast(x, tf.float32) - mean) / std, tf.cast(y, tf.float32)

    def _augment(x, y):
        k = tf.random.uniform([], 0, 4, tf.int32)
        x, y = tf.image.rot90(x, k), tf.image.rot90(y, k)
        flip = tf.random.uniform([]) < 0.5
        x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
        y = tf.cond(flip, lambda: tf.image.flip_left_right(y), lambda: y)
        return x, y

    ds = tf.data.Dataset.from_tensor_slices(list(by_path))
    if cache:
        ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE).cache()
        if shuffle:  # full-size buffer: a complete shuffle each epoch, at the cost of a second copy in RAM
            ds = ds.shuffle(len(by_path), seed=seed, reshuffle_each_iteration=True)
    else:
        if shuffle:
            ds = ds.shuffle(len(by_path), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_standardize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
