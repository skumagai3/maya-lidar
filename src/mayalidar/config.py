"""Experiment configs, project paths and run folders."""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


def in_colab() -> bool:
    return "google.colab" in sys.modules


@dataclass
class Paths:
    root: Path      # where runs/ live: the repo locally, a Drive folder on Colab
    data_dir: Path  # extracted tiles: repo/data/chactun locally, /content/data/chactun on Colab
    runs_dir: Path


def project_paths(drive_dir: str | Path | None = None) -> Paths:
    """On Colab, tiles go on the VM's local disk (reading thousands of small files
    from Drive is slow) and runs go to drive_dir so they persist. Locally,
    everything is under the repo root."""
    if in_colab():
        root = Path(drive_dir or "/content/drive/MyDrive/maya-lidar")
        data_dir = Path("/content/data/chactun")
    else:
        cwd = Path.cwd()
        root = cwd.parent if cwd.name == "notebooks" else cwd
        data_dir = root / "data" / "chactun"
    return Paths(root, data_dir, root / "runs")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(name_or_path: str | Path, overrides: dict | None = None) -> dict:
    """Load configs/<name>.yaml (searched in the repo, then next to the installed
    package) or an explicit path, with optional nested overrides."""
    p = Path(name_or_path)
    if not p.suffix:
        here = Path(__file__).resolve()
        candidates = [Path.cwd() / "configs", Path.cwd().parent / "configs",
                      here.parents[2] / "configs", here.parent / "configs"]
        p = next((c / f"{name_or_path}.yaml" for c in candidates if (c / f"{name_or_path}.yaml").exists()), p)
    cfg = yaml.safe_load(Path(p).read_text())
    return _merge(cfg, overrides or {})


def new_run_dir(paths: Paths, experiment: str, tag: str = "") -> Path:
    """runs/<experiment>/<YYYYmmdd-HHMMSS>[-tag]/"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = paths.runs_dir / experiment / (f"{stamp}-{tag}" if tag else stamp)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg: dict, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
