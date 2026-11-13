"""Scoring a trained model: pixel IoU per class, threshold sweeps, per-tile scores.

IoU here is accumulated over every pixel of the evaluated set (TP, FP and FN summed
over all tiles before dividing), matching the Chactun challenge metric. A pixel is
predicted positive when its sigmoid output exceeds the class threshold.
"""
from __future__ import annotations

import numpy as np

from .data import CLASSES


def predict(model, ds):
    """Run the model over a dataset. Returns labels Y (uint8) and sigmoid outputs
    S (float16, to halve memory), both shaped (N, H, W, n_classes)."""
    ys, ss = [], []
    for xb, yb in ds:
        ys.append(yb.numpy().astype(np.uint8))
        ss.append(model.predict(xb, verbose=0).astype(np.float16))
    return np.concatenate(ys), np.concatenate(ss)


def save_outputs(path, tile_ids, S) -> None:
    """Save sigmoid outputs (float16) with their tile ids, so a run can be inspected
    later without rerunning the model. About 120 MB for the validation split."""
    np.savez(path, tile_ids=np.asarray(tile_ids), sigmoid=S.astype(np.float16))


def load_outputs(path):
    """Returns (tile_ids, sigmoid outputs) saved by save_outputs."""
    with np.load(path) as f:
        return f["tile_ids"], f["sigmoid"]


def iou(y: np.ndarray, s: np.ndarray, threshold: float) -> float:
    t, p = y > 0, s > threshold
    union = np.logical_or(t, p).sum()
    return float(np.logical_and(t, p).sum() / union) if union else float("nan")


def threshold_sweep(Y, S, thresholds) -> dict:
    """{class: {threshold: IoU}} over the given thresholds."""
    return {c: {round(float(th), 3): iou(Y[..., i], S[..., i], th) for th in thresholds}
            for i, c in enumerate(CLASSES)}


def best_thresholds(sweep: dict) -> dict:
    return {c: max(v, key=v.get) for c, v in sweep.items()}


def scores(Y, S, thresholds: dict | float = 0.5) -> dict:
    """Per-class IoU and their unweighted mean, at fixed thresholds."""
    th = thresholds if isinstance(thresholds, dict) else {c: thresholds for c in CLASSES}
    per = {c: iou(Y[..., i], S[..., i], th[c]) for i, c in enumerate(CLASSES)}
    return {"iou": per, "mean_iou": float(np.nanmean(list(per.values()))), "thresholds": th}


def per_tile_errors(Y, S, thresholds: dict, cls: str) -> dict:
    """Per tile, for one class: false-positive and false-negative pixel counts and
    labeled pixel count. Used to rank tiles for galleries."""
    i = CLASSES.index(cls)
    t, p = Y[..., i] > 0, S[..., i] > thresholds[cls]
    return {"fp": (p & ~t).sum((1, 2)), "fn": (t & ~p).sum((1, 2)), "labeled": t.sum((1, 2))}


def pr_curves(Y, S, n_bins: int = 1000) -> dict:
    """Pixel-level precision and recall per class as the threshold on the sigmoid
    output sweeps from 0 to 1, from histograms of the outputs (memory stays flat).
    Also returns average precision (area under the stepwise PR curve)."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = {}
    for i, c in enumerate(CLASSES):
        s = S[..., i].astype(np.float32).ravel()
        t = Y[..., i].ravel() > 0
        pos = np.histogram(s[t], edges)[0][::-1].cumsum()   # predicted positive and labeled, threshold high to low
        neg = np.histogram(s[~t], edges)[0][::-1].cumsum()
        tp, fp, n_pos = pos.astype(float), neg.astype(float), t.sum()
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
        if n_pos == 0:  # class absent from these tiles: recall and AP undefined
            recall = np.full_like(tp, np.nan)
            ap = float("nan")
        else:
            recall = tp / n_pos
            ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
        out[c] = {"threshold": edges[::-1][1:].tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "average_precision": ap}
    return out


def object_size_stats(Y, S, thresholds: dict, bins_m2=(0, 25, 50, 100, 200, 400, 800, 1600, np.inf),
                      m2_per_px: float = 0.25, detect_frac: float = 0.5) -> dict:
    """Object-level view, per class, binned by object area in square meters.

    Objects are 8-connected components of the labels (or of the prediction).
    A labeled object counts as detected if at least detect_frac of its pixels are
    predicted positive. A predicted object counts as a false positive if none of
    its pixels are labeled. Objects cut by the tile edge are included, so sizes of
    large objects are underestimated."""
    from scipy import ndimage

    conn = np.ones((3, 3), bool)
    bins = np.asarray(bins_m2, float)
    out = {"bins_m2": [float(b) for b in bins], "detect_frac": detect_frac}
    for i, c in enumerate(CLASSES):
        lab_area, lab_hit, pred_area, pred_false = [], [], [], []
        for k in range(len(Y)):
            t, p = Y[k, ..., i] > 0, S[k, ..., i] > thresholds[c]
            L, n = ndimage.label(t, conn)
            if n:
                idx = np.arange(1, n + 1)
                area = ndimage.sum(np.ones_like(t), L, idx)
                cover = ndimage.sum(p, L, idx)
                lab_area += list(area * m2_per_px)
                lab_hit += list(cover / area >= detect_frac)
            P, m = ndimage.label(p, conn)
            if m:
                idx = np.arange(1, m + 1)
                area = ndimage.sum(np.ones_like(p), P, idx)
                overlap = ndimage.sum(t, P, idx)
                pred_area += list(area * m2_per_px)
                pred_false += list(overlap == 0)

        def binned(area, flag):
            area, flag = np.asarray(area), np.asarray(flag, float)
            which = np.digitize(area, bins) - 1
            n = np.array([(which == b).sum() for b in range(len(bins) - 1)])
            frac = np.array([flag[which == b].mean() if n[b] else np.nan for b in range(len(bins) - 1)])
            return n.tolist(), frac.tolist()

        n_lab, det = binned(lab_area, lab_hit)
        n_pred, false = binned(pred_area, pred_false)
        out[c] = {"n_labeled": n_lab, "fraction_detected": det, "n_predicted": n_pred, "fraction_false": false}
    return out


def per_tile_mean_iou(Y, S, thresholds: dict) -> np.ndarray:
    """For each tile, the mean IoU over the classes present in its labels."""
    out = np.empty(len(Y))
    for k in range(len(Y)):
        vals = [iou(Y[k, ..., i], S[k, ..., i], thresholds[c]) for i, c in enumerate(CLASSES) if Y[k, ..., i].any()]
        out[k] = np.mean(vals) if vals else np.nan
    return out
