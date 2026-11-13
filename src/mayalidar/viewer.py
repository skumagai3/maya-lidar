"""Interactive tile viewer for a finished run (Jupyter or Colab, needs ipywidgets).

Reads only what a run saved: val_outputs.npz (sigmoid outputs), scores.json
(validation-chosen thresholds) and the tiles on disk. Nothing is recomputed by the
model, so a run can be inspected without a GPU.

    from mayalidar.viewer import Viewer
    Viewer(run_dir, data_dir).show()
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import data as D
from . import evaluate as E
from . import viz as V
from .train import MASK_PRESENT_VALUE

LAYERS = ("labels", "predictions", "errors", "sigmoid output")


class Viewer:
    def __init__(self, run_dir: str | Path, data_dir: str | Path, outputs: str = "val_outputs.npz"):
        self.run_dir = Path(run_dir)
        ids, self.S = E.load_outputs(self.run_dir / outputs)
        by_id = {t.tile_id: t for t in D.index_tiles(data_dir)}
        self.tiles = [by_id[int(i)] for i in ids]
        self.Y = np.stack([D.load_mask(t, MASK_PRESENT_VALUE) for t in self.tiles]).astype(np.uint8)
        scores = D.load_json(self.run_dir / "scores.json")
        self.chosen = {c: float(v) for c, v in scores["validation"]["thresholds"].items()}
        self.experiment = self.run_dir.parent.name

    # ---------------------------------------------------------------- ordering

    def order(self, mode: str, thresholds: dict) -> np.ndarray:
        """Tile indices for a sort mode: 'tile id', 'worst mean IoU',
        'most missed <class>' or 'most false <class>'."""
        if mode == "tile id":
            return np.argsort([t.tile_id for t in self.tiles])
        if mode == "worst mean IoU":
            return np.argsort(np.nan_to_num(E.per_tile_mean_iou(self.Y, self.S, thresholds), nan=1.0))
        kind, cls = mode.split(" ", 2)[1:]
        err = E.per_tile_errors(self.Y, self.S, thresholds, cls)
        return np.argsort(-err["fn" if kind == "missed" else "fp"])

    def sort_modes(self) -> list[str]:
        return (["worst mean IoU", "tile id"] + [f"most missed {c}" for c in D.CLASSES]
                + [f"most false {c}" for c in D.CLASSES])

    # ---------------------------------------------------------------- drawing

    def draw(self, k: int, thresholds: dict, band: int = 0, layers=("labels", "predictions"), cls: str = "building"):
        """One large panel for tile index k. 'errors' and 'sigmoid output' apply to cls;
        'labels' and 'predictions' draw outlines for every class."""
        x = D.read_tif(self.tiles[k].lidar)
        y, s = self.Y[k], self.S[k].astype(np.float32)
        i = D.CLASSES.index(cls)
        fig, ax = plt.subplots(figsize=(8.5, 8))
        V._background(ax, x, band)
        if "sigmoid output" in layers:
            im = ax.imshow(np.ma.masked_less(s[..., i], 0.05), cmap="viridis", vmin=0, vmax=1, alpha=0.75)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=f"{cls} sigmoid output")
        if "errors" in layers:
            V._error_fill(ax, y[..., i] > 0, s[..., i] > thresholds[cls])
        for j, c in enumerate(D.CLASSES):
            if "labels" in layers:
                V._outlines(ax, y[..., j] > 0, V.CLASS_COLORS[c])
            if "predictions" in layers:
                V._outlines(ax, s[..., j] > thresholds[c], V.CLASS_COLORS[c], dashed=True)
        V._scale_bar(ax)
        if "labels" in layers or "predictions" in layers:
            legend = V._class_legend(ax, loc="upper right", framealpha=0.8, frameon=True)
            ax.add_artist(legend)  # keep it when a second legend is drawn
        if "errors" in layers:
            V._error_legend(ax, loc="lower right", framealpha=0.8, frameon=True)
        iou = V._tile_iou(y, s, thresholds)
        sub = ", ".join(f"{c} {v:.2f}" if np.isfinite(v) else f"{c} n/a" for c, v in iou.items())
        th = ", ".join(f"{c} {thresholds[c]:.2f}" for c in D.CLASSES)
        ax.set_title(f"{self.experiment}, validation tile {self.tiles[k].tile_id}: {V.BAND_NAMES[band]}\n"
                     f"IoU on this tile: {sub}\nthresholds: {th}", fontsize=9)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------- widgets

    def show(self):
        import ipywidgets as W
        from IPython.display import display

        sort = W.Dropdown(options=self.sort_modes(), description="sort by")
        pos = W.IntSlider(0, 0, len(self.tiles) - 1, description="rank")
        prev, nxt = W.Button(description="previous"), W.Button(description="next")
        band = W.Dropdown(options=[(n, i) for i, n in enumerate(V.BAND_NAMES)], description="background")
        layers = W.SelectMultiple(options=LAYERS, value=("labels", "predictions"), description="layers", rows=4)
        cls = W.Dropdown(options=D.CLASSES, description="class")
        sliders = {c: W.FloatSlider(self.chosen[c], min=0.05, max=0.95, step=0.05, description=f"{c} thr",
                                    continuous_update=False) for c in D.CLASSES}
        reset = W.Button(description="reset thresholds")
        save = W.Button(description="save view")
        status = W.Label()
        out = W.Output()
        state = {"order": self.order(sort.value, self.chosen), "fig": None}

        def thresholds():
            return {c: sl.value for c, sl in sliders.items()}

        def redraw(*_):
            k = int(state["order"][pos.value])
            with out:
                out.clear_output(wait=True)
                plt.close("all")
                state["fig"] = self.draw(k, thresholds(), band.value, layers.value, cls.value)
                plt.show()
            status.value = f"tile {self.tiles[k].tile_id} ({pos.value + 1} of {len(self.tiles)})"

        def resort(*_):
            state["order"] = self.order(sort.value, thresholds())
            pos.value = 0
            redraw()

        def step(d):
            pos.value = int(np.clip(pos.value + d, 0, len(self.tiles) - 1))

        def do_reset(_):
            for c, sl in sliders.items():
                sl.value = self.chosen[c]

        def do_save(_):
            k = int(state["order"][pos.value])
            th = "_".join(f"{sliders[c].value:.2f}" for c in D.CLASSES)
            name = f"viewer_tile{self.tiles[k].tile_id}_{band.value}_{'-'.join(l[:3] for l in layers.value)}_{cls.value}_th{th}"
            path = V.save(state["fig"], self.run_dir, name)
            status.value = f"saved {path}"

        sort.observe(resort, "value")
        for wdg in [pos, band, layers, cls, *sliders.values()]:
            wdg.observe(redraw, "value")
        prev.on_click(lambda _: step(-1))
        nxt.on_click(lambda _: step(1))
        reset.on_click(do_reset)
        save.on_click(do_save)

        controls = W.VBox([
            W.HBox([sort, prev, nxt, pos]),
            W.HBox([band, cls, layers]),
            W.HBox(list(sliders.values()) + [reset]),
            W.HBox([save, status]),
        ])
        display(controls, out)
        redraw()
