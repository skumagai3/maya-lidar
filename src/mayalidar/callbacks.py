"""Training callbacks: the per-epoch log and TensorBoard images. Requires TensorFlow."""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from . import data as D
from . import viz as V


def read_log(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return [{k: (int(float(v)) if k == "epoch" else float(v)) for k, v in row.items() if v != ""}
                for row in csv.DictReader(f)]


class RunLog(keras.callbacks.Callback):
    """Writes the whole CSV log again after every epoch (so the copy on Google
    Drive is always complete) and copies the TensorBoard logs from local disk to
    the run folder. Rows already in the file are kept; a repeated epoch number
    after a resume replaces the earlier row."""

    def __init__(self, path: Path, tb_local: Path | None = None, tb_run: Path | None = None):
        super().__init__()
        self.path, self.tb_local, self.tb_run = Path(path), tb_local, tb_run
        self.rows = {r["epoch"]: r for r in read_log(self.path)}

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch, **{k: float(v) for k, v in (logs or {}).items()}}
        row["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.rows[epoch] = row
        rows = [self.rows[k] for k in sorted(self.rows)]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        tmp = self.path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)
        if self.tb_local is not None and self.tb_local.exists():
            shutil.copytree(self.tb_local, self.tb_run, dirs_exist_ok=True)


def pick_image_tiles(tiles: list[D.Tile], present_value: int) -> list[tuple[D.Tile, str]]:
    """For each class, the tile with the most labeled pixels of it (no repeats).
    The split is seeded, so the same tiles are chosen in every run."""
    masks = np.stack([D.load_mask(t, present_value).sum((0, 1)) for t in tiles])
    out, used = [], set()
    for i, c in enumerate(D.CLASSES):
        for k in np.argsort(-masks[:, i]):
            if k not in used:
                used.add(k)
                out.append((tiles[k], c))
                break
    return out


class TileImages(keras.callbacks.Callback):
    """Writes one image per chosen validation tile to TensorBoard after every
    epoch: sky-view factor with the labels on the left and the prediction
    (sigmoid output > threshold) on the right, in the class colors."""

    def __init__(self, picks, stats: dict, present_value: int, log_dir: Path, threshold: float = 0.5):
        super().__init__()
        mean = np.asarray(stats["mean"], np.float32)
        std = np.asarray(stats["std"], np.float32)
        self.ids = [t.tile_id for t, _ in picks]
        self.classes = [c for _, c in picks]
        raw = [D.read_tif(t.lidar) for t, _ in picks]
        self.x = np.stack([(r - mean) / std for r in raw]).astype(np.float32)
        self.base = [V.stretch(r[..., 0]) for r in raw]
        self.y = [D.load_mask(t, present_value) for t, _ in picks]
        self.threshold = threshold
        self.writer = tf.summary.create_file_writer(str(log_dir / "validation_tiles"))

    def on_epoch_end(self, epoch, logs=None):
        s = np.asarray(self.model.predict_on_batch(self.x), np.float32)
        with self.writer.as_default():
            for k, (tid, cls) in enumerate(zip(self.ids, self.classes)):
                left = V.overlay(self.base[k], self.y[k], alpha=0.5)
                right = V.overlay(self.base[k], (s[k] > self.threshold).astype(np.float32), alpha=0.5)
                gap = np.ones((left.shape[0], 6, 3))
                img = (255 * np.clip(np.concatenate([left, gap, right], 1), 0, 1)).astype(np.uint8)
                tf.summary.image(f"tile {tid} (most {cls}): labels left, sigmoid > {self.threshold} right",
                                 img[None], step=epoch)
        self.writer.flush()
