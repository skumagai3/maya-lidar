"""Figures for tiles, labels, model outputs, training curves and threshold sweeps."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES

BAND_NAMES = ("sky-view factor", "positive openness", "slope")
# Okabe-Ito colorblind-safe palette
CLASS_COLORS = {"building": "#D55E00", "platform": "#CC79A7", "aguada": "#0072B2"}
ERROR_COLORS = {"true positive": "#009E73", "false positive": "#E69F00", "false negative": "#56B4E9"}
M_PER_PX = 0.5


def save(fig, run_dir: Path, name: str) -> Path:
    path = Path(run_dir) / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def stretch(a, lo=2, hi=98):
    p1, p2 = np.nanpercentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def overlay(base, mask, alpha=0.45):
    """Color class masks over a grayscale image. base: (H, W) in [0, 1]; mask: (H, W, 3) in {0, 1}."""
    from matplotlib.colors import to_rgb

    rgb = np.repeat(base[..., None], 3, -1)
    for i, c in enumerate(CLASSES):
        m = mask[..., i] > 0.5
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(to_rgb(CLASS_COLORS[c]))
    return rgb


# ---------------------------------------------------------------- tile figure building blocks

def _background(ax, x, band=0):
    """Sky-view factor (or another band), contrast-stretched and lightened slightly
    so colored outlines and fills stand out."""
    ax.imshow(0.25 + 0.75 * stretch(x[..., band]), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])


def _outlines(ax, mask, color, dashed=False, lw=1.3):
    """Contour of a binary mask at the pixel boundary."""
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw,
                   linestyles="--" if dashed else "-")


def _error_fill(ax, t, p, alpha=0.65):
    """Fill true positives, false positives and false negatives for one class."""
    from matplotlib.colors import to_rgba

    rgba = np.zeros(t.shape + (4,))
    for m, name in [(t & p, "true positive"), (p & ~t, "false positive"), (t & ~p, "false negative")]:
        r, g, b, _ = to_rgba(ERROR_COLORS[name])
        rgba[m] = (r, g, b, alpha)
    ax.imshow(rgba)


def _scale_bar(ax, meters=50, m_per_px=M_PER_PX):
    n = meters / m_per_px
    h, w = ax.get_images()[0].get_array().shape[:2]
    x0, y0 = 0.04 * w, 0.94 * h
    ax.plot([x0, x0 + n], [y0, y0], color="black", lw=4, solid_capstyle="butt")
    ax.plot([x0, x0 + n], [y0, y0], color="white", lw=2, solid_capstyle="butt")
    ax.text(x0 + n / 2, y0 - 0.03 * h, f"{meters} m", ha="center", va="bottom", fontsize=7,
            color="black", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))


def _class_legend(ax_or_fig, **kw):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=CLASS_COLORS[c], lw=2, label=c) for c in CLASSES]
    h += [Line2D([], [], color="0.2", lw=1.3, label="label"), Line2D([], [], color="0.2", lw=1.3, ls="--", label="prediction")]
    return ax_or_fig.legend(handles=h, **{"fontsize": 8, "frameon": False, **kw})


def _error_legend(ax_or_fig, **kw):
    h = [plt.Rectangle((0, 0), 1, 1, color=v, label=k) for k, v in ERROR_COLORS.items()]
    return ax_or_fig.legend(handles=h, **{"fontsize": 8, "frameon": False, **kw})


def _tile_iou(y, s, th):
    out = {}
    for i, c in enumerate(CLASSES):
        t, p = y[..., i] > 0.5, s[..., i] > th[c]
        u = (t | p).sum()
        out[c] = (t & p).sum() / u if u else float("nan")
    return out


# ---------------------------------------------------------------- tile figures

def show_tile(x, y, title, s=None, thresholds=None):
    """Detailed figure for one tile.

    x: (H, W, 3) lidar tile as stored (0-255); y: (H, W, 3) labels in {0, 1};
    s: optional (H, W, 3) sigmoid outputs; thresholds: per-class threshold on s.

    Without s: the three bands and the label outlines. With s, two more rows:
    the sigmoid output per class (label outline on top) and the per-class error
    map (true positive, false positive, false negative at the threshold).
    """
    th = thresholds or {c: 0.5 for c in CLASSES}
    rows = 1 if s is None else 3
    fig, ax = plt.subplots(rows, 5, figsize=(17, 3.6 * rows), squeeze=False)
    for i in range(3):
        ax[0, i].imshow(stretch(x[..., i]), cmap="gray")
        ax[0, i].set_title(BAND_NAMES[i], fontsize=9)
        ax[0, i].set_xticks([]), ax[0, i].set_yticks([])
    _scale_bar(ax[0, 0])
    _background(ax[0, 3], x)
    for i, c in enumerate(CLASSES):
        _outlines(ax[0, 3], y[..., i] > 0.5, CLASS_COLORS[c])
    ax[0, 3].set_title("labels over sky-view factor", fontsize=9)
    if s is None:
        ax[0, 4].axis("off")
        _class_legend(ax[0, 4], loc="center")
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    _background(ax[0, 4], x)
    for i, c in enumerate(CLASSES):
        _outlines(ax[0, 4], y[..., i] > 0.5, CLASS_COLORS[c])
        _outlines(ax[0, 4], s[..., i] > th[c], CLASS_COLORS[c], dashed=True)
    ax[0, 4].set_title("labels (solid) and predictions (dashed)", fontsize=9)

    for i, c in enumerate(CLASSES):
        a = ax[1, i]
        im = a.imshow(s[..., i], cmap="viridis", vmin=0, vmax=1)
        _outlines(a, y[..., i] > 0.5, "white", lw=1.0)
        a.set_title(f"{c}: sigmoid output (white: label)", fontsize=9)
        a.set_xticks([]), a.set_yticks([])

        b = ax[2, i]
        _background(b, x)
        _error_fill(b, y[..., i] > 0.5, s[..., i] > th[c])
        b.set_title(f"{c}: errors, threshold {th[c]:.2f}", fontsize=9)
    ax[1, 3].axis("off")
    cax = ax[1, 3].inset_axes([0.05, 0.1, 0.07, 0.8])
    fig.colorbar(im, cax=cax, label="sigmoid output")
    ax[1, 4].axis("off")
    _class_legend(ax[1, 4], loc="center")
    ax[2, 3].axis("off")
    _error_legend(ax[2, 3], loc="center")
    ax[2, 4].axis("off")

    iou = _tile_iou(y, s, th)
    sub = ", ".join(f"{c} {v:.2f}" if np.isfinite(v) else f"{c} n/a" for c, v in iou.items())
    fig.suptitle(f"{title}\nIoU on this tile: {sub}")
    fig.tight_layout(h_pad=1.5)
    return fig


def gallery(xs, ys, ss, thresholds, panel_titles, title, cls=None, ncols=4):
    """Grid of tiles. With cls=None each panel shows label (solid) and prediction
    (dashed) outlines for all classes; with cls set, the error map for that class.
    xs: list of (H, W, 3) lidar tiles; ys, ss: matching labels and sigmoid outputs."""
    n = len(xs)
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.5 * nrows + 0.6), squeeze=False)
    for k, a in enumerate(ax.flat):
        if k >= n:
            a.axis("off")
            continue
        _background(a, xs[k])
        if cls is None:
            for i, c in enumerate(CLASSES):
                _outlines(a, ys[k][..., i] > 0.5, CLASS_COLORS[c], lw=1.0)
                _outlines(a, ss[k][..., i] > thresholds[c], CLASS_COLORS[c], dashed=True, lw=1.0)
        else:
            i = CLASSES.index(cls)
            _error_fill(a, ys[k][..., i] > 0.5, ss[k][..., i] > thresholds[cls])
        a.set_title(panel_titles[k], fontsize=8)
    _scale_bar(ax[0, 0])
    if cls is None:
        _class_legend(fig, loc="lower center", ncol=5)
    else:
        _error_legend(fig, loc="lower center", ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def show_tile_simple(x, y, title, s=None, thresholds=None):
    """Earlier, simpler tile figure: filled class overlays on one row.
    x: (H, W, 3) lidar tile as stored (0-255); y: (H, W, 3) labels;
    s: optional (H, W, 3) sigmoid outputs, shown as s > threshold per class."""
    n = 5 if s is None else 6
    fig, ax = plt.subplots(1, n, figsize=(3.3 * n, 3.7))
    for i in range(3):
        ax[i].imshow(stretch(x[..., i]), cmap="gray")
        ax[i].set_title(BAND_NAMES[i], fontsize=9)
    ax[3].imshow(np.dstack([stretch(x[..., i]) for i in range(3)]))
    ax[3].set_title("bands as RGB", fontsize=9)
    ax[4].imshow(overlay(stretch(x[..., 0]), y))
    ax[4].set_title("labels over sky-view factor", fontsize=9)
    if s is not None:
        th = thresholds or {c: 0.5 for c in CLASSES}
        pred = np.stack([s[..., i] > th[c] for i, c in enumerate(CLASSES)], -1).astype(float)
        ax[5].imshow(overlay(stretch(x[..., 0]), pred))
        ax[5].set_title("sigmoid output > class threshold", fontsize=9)
    for a in ax:
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASSES]
    fig.legend(handles, CLASSES, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig



def plot_band_histograms(samples: np.ndarray, title: str):
    """samples: (N, 3) pixel values from the lidar tiles."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    for i, name in enumerate(BAND_NAMES):
        ax[i].hist(samples[:, i], bins=np.arange(257) - 0.5, color="0.3")
        ax[i].set_title(name, fontsize=10)
        ax[i].set_xlabel("stored value (0-255)")
    ax[0].set_ylabel("pixels")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_counts(counts: dict, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    colors = [CLASS_COLORS[c] for c in CLASSES]
    ax[0].bar(CLASSES, [100 * counts["pixel_fraction"][c] for c in CLASSES], color=colors)
    ax[0].set_ylabel("% of pixels labeled")
    ax[1].bar(CLASSES, [counts["tiles_with_class"][c] for c in CLASSES], color=colors)
    ax[1].axhline(counts["n_tiles"], color="0.5", ls="--", lw=1)
    ax[1].set_ylabel(f"tiles containing class (of {counts['n_tiles']})")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_history(log_csvs: list[Path], title: str):
    """Loss and per-class validation IoU per epoch. Several logs (e.g. frozen then
    fine-tuned encoder) are concatenated, with a dashed line at the switch."""
    import pandas as pd

    logs = [pd.read_csv(p) for p in log_csvs if Path(p).exists()]
    log = pd.concat(logs, ignore_index=True)
    epoch = np.arange(1, len(log) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(epoch, log["loss"], label="train")
    ax[0].plot(epoch, log["val_loss"], label="validation")
    ax[0].set_ylabel("BCE + Dice loss")
    for c in CLASSES:
        ax[1].plot(epoch, log[f"val_iou_{c}"], label=c, color=CLASS_COLORS[c])
    ax[1].set_ylabel("validation IoU (sigmoid > 0.5)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend(fontsize=8)
        if len(logs) > 1:
            a.axvline(len(logs[0]) + 0.5, color="0.5", ls="--", lw=1)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(sweep: dict, chosen: dict, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for c in CLASSES:
        th, v = zip(*sorted(sweep[c].items()))
        ax.plot(th, v, marker="o", ms=3, color=CLASS_COLORS[c], label=c)
        ax.plot(chosen[c], sweep[c][chosen[c]], "k*", ms=9)
    ax.set_xlabel("threshold on sigmoid output")
    ax.set_ylabel("IoU")
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scores(scores_by_split: dict, title: str):
    """Grouped bars of per-class IoU, e.g. {'validation': scores, 'test': scores}."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = list(scores_by_split)
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        vals = [scores_by_split[n]["iou"][c] for c in CLASSES] + [scores_by_split[n]["mean_iou"]]
        xs = np.arange(len(CLASSES) + 1) + (j - (len(names) - 1) / 2) * w
        bars = ax.bar(xs, vals, w, label=n)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(np.arange(len(CLASSES) + 1), list(CLASSES) + ["mean"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("IoU")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pr_curves(pr: dict, chosen: dict, title: str):
    """Pixel-level precision vs. recall per class as the threshold on the sigmoid
    output varies. Dots mark the thresholds chosen on validation."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for c in CLASSES:
        d = pr[c]
        if not np.isfinite(d["average_precision"]):
            ax.plot([], [], color=CLASS_COLORS[c], label=f"{c} (no labeled pixels)")
            continue
        r, p, t = np.array(d["recall"]), np.array(d["precision"]), np.array(d["threshold"])
        ax.plot(r, p, color=CLASS_COLORS[c], label=f"{c} (AP {d['average_precision']:.3f})")
        k = np.argmin(np.abs(t - chosen[c]))
        ax.plot(r[k], p[k], "o", color=CLASS_COLORS[c], mec="black", ms=7)
    ax.set_xlabel("recall (pixels)")
    ax.set_ylabel("precision (pixels)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_object_size(stats: dict, title: str):
    """Left: fraction of labeled objects detected, by object area. Right: fraction
    of predicted objects with no labeled pixel (false positives), by area. Bar
    labels give the number of objects in each bin."""
    bins = stats["bins_m2"]
    names = [f"{int(a)}-{int(b)}" if np.isfinite(b) else f">{int(a)}" for a, b in zip(bins[:-1], bins[1:])]
    xs = np.arange(len(names))
    w = 0.8 / len(CLASSES)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for j, c in enumerate(CLASSES):
        off = (j - 1) * w
        for a, frac, n in [(ax[0], "fraction_detected", "n_labeled"), (ax[1], "fraction_false", "n_predicted")]:
            v = np.nan_to_num(np.array(stats[c][frac], float), nan=0.0)
            bars = a.bar(xs + off, v, w, color=CLASS_COLORS[c], label=c)
            a.bar_label(bars, labels=[str(k) if k else "" for k in stats[c][n]], fontsize=6, padding=1)
    ax[0].set_ylabel(f"fraction of labeled objects detected\n(at least {stats['detect_frac']:.0%} of pixels predicted)")
    ax[1].set_ylabel("fraction of predicted objects\nwith no labeled pixel")
    for a in ax:
        a.set_xticks(xs, names, rotation=30)
        a.set_xlabel("object area (m²)")
        a.set_ylim(0, 1.1)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
