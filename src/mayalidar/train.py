"""Build, compile and train a model from an experiment config."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data as D
from .config import save_config

MASK_PRESENT_VALUE = 0  # Kokalj et al. 2023: 0 = feature, 255 = background


def gpu_report(require: bool = True) -> list[str]:
    """Print the GPUs TensorFlow sees. With require=True, raise if there are none,
    so a CPU-only Colab runtime fails before any training starts."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    names = [tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus]
    print("TensorFlow", tf.__version__, "| GPUs:", names or "none")
    if require and not gpus:
        raise RuntimeError("No GPU visible. On Colab: Runtime > Change runtime type > GPU.")
    return names


def peak_gpu_memory_gb() -> float | None:
    """Peak memory allocated on GPU:0 since startup, in GB, or None without a GPU."""
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        return None
    return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9


def build_model(cfg: dict):
    from . import model as M

    m = cfg["model"]
    if m["name"] == "unet":
        return M.unet(base_filters=m["base_filters"], depth=m["depth"], dropout=m.get("dropout", 0.0))
    if m["name"] == "resnet50_unet":
        return M.resnet50_unet(weights=m.get("encoder_weights", "imagenet"), freeze_encoder=True)
    raise ValueError(f"unknown model {m['name']}")


def dice_class_weights(cfg: dict, pixel_fraction: list[float]) -> list[float]:
    """'uniform' gives every class weight 1 in the Dice term. 'inv_sqrt_frequency'
    weights class c by 1/sqrt(f_c), normalized to mean 1, where f_c is the fraction
    of training pixels labeled c."""
    mode = cfg["loss"].get("dice_class_weights", "uniform")
    if mode == "uniform":
        return [1.0] * len(D.CLASSES)
    if mode == "inv_sqrt_frequency":
        w = 1 / np.sqrt(np.asarray(pixel_fraction) + 1e-6)
        return (w / w.mean()).tolist()
    raise ValueError(f"unknown dice_class_weights {mode}")


def compile_model(net, cfg: dict, learning_rate: float, class_weights: list[float]):
    import keras
    from . import model as M

    net.compile(
        optimizer=keras.optimizers.AdamW(learning_rate, weight_decay=cfg["train"]["weight_decay"]),
        loss=M.dice_bce_loss(class_weights, bce_weight=cfg["loss"]["bce_weight"]),
        metrics=M.iou_metrics(),
    )


def _best_logged(run_dir: Path, *logs: str) -> float | None:
    """Lowest val_loss in existing CSV logs, so a resumed run does not overwrite
    best.keras with a worse model."""
    import pandas as pd

    vals = [pd.read_csv(run_dir / n)["val_loss"].min() for n in logs
            if (run_dir / n).exists() and (run_dir / n).stat().st_size > 0]
    return float(min(vals)) if vals else None


def _callbacks(run_dir: Path, cfg: dict, phase: str, best_so_far: float | None):
    import keras

    t = cfg["train"]
    log = "log.csv" if phase == "main" else f"log_{phase}.csv"
    return [
        # restores model, optimizer and epoch from run_dir/backup_<phase> if the session died mid-phase
        keras.callbacks.BackupAndRestore(str(run_dir / f"backup_{phase}")),
        keras.callbacks.ModelCheckpoint(run_dir / "best.keras", monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=best_so_far),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=t["lr_patience"], min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["early_stop_patience"], restore_best_weights=True),
        keras.callbacks.CSVLogger(run_dir / log, append=True),
    ]


def run(cfg: dict, train_tiles, val_tiles, run_dir: Path, smoke: bool = False):
    """Train one experiment and save everything needed to read it back into run_dir:
    config.yaml, split.json, normalization.json, log.csv (and log_finetune.csv),
    best.keras. With smoke=True, uses a few tiles and epochs to check the pipeline.

    Calling run again with the same run_dir resumes: a phase that finished is
    skipped (marker file phase_<name>.done), and a phase that was interrupted
    continues from its last completed epoch. Early stopping and learning-rate
    plateau counters restart from zero on resume."""
    import keras
    from . import model as M

    t = cfg["train"]
    epochs = t["epochs"]
    if smoke:
        s = cfg["smoke"]
        train_tiles, val_tiles, epochs = train_tiles[: s["n_train"]], val_tiles[: s["n_val"]], s["epochs"]

    keras.mixed_precision.set_global_policy("mixed_float16" if t.get("mixed_precision", False) else "float32")

    stats = D.channel_stats(train_tiles)
    counts = D.class_counts(train_tiles, MASK_PRESENT_VALUE)
    cw = dice_class_weights(cfg, list(counts["pixel_fraction"].values()))
    save_config(cfg, run_dir)
    D.save_json({"train": [x.tile_id for x in train_tiles], "val": [x.tile_id for x in val_tiles]}, run_dir / "split.json")
    D.save_json({**stats, "train_class_counts": counts, "dice_class_weights": cw}, run_dir / "normalization.json")

    keras.utils.set_random_seed(t["seed"])
    ds_train = D.make_dataset(train_tiles, stats, MASK_PRESENT_VALUE, t["batch_size"], augment=True, shuffle=True,
                              cache=t.get("cache_train", True), seed=t["seed"])
    ds_val = D.make_dataset(val_tiles, stats, MASK_PRESENT_VALUE, t["batch_size"], cache=t.get("cache_val", True))

    main_done = run_dir / "phase_main.done"
    if main_done.exists():
        net = keras.models.load_model(run_dir / "best.keras", compile=False)
        print("main phase already finished; loaded best.keras")
    else:
        net = build_model(cfg)
        compile_model(net, cfg, t["learning_rate"], cw)
        net.fit(ds_train, validation_data=ds_val, epochs=epochs,
                callbacks=_callbacks(run_dir, cfg, "main", _best_logged(run_dir, "log.csv")))
        main_done.touch()

    ft = cfg.get("finetune")
    if ft and cfg["model"]["name"] == "resnet50_unet" and not (run_dir / "phase_finetune.done").exists():
        M.unfreeze(net, ft["unfreeze_from"])
        compile_model(net, cfg, t["learning_rate"] * ft["lr_factor"], cw)
        ft_epochs = cfg["smoke"]["epochs"] if smoke else ft["epochs"]
        # best.keras is only overwritten if fine-tuning beats every earlier epoch
        net.fit(ds_train, validation_data=ds_val, epochs=ft_epochs,
                callbacks=_callbacks(run_dir, cfg, "finetune", _best_logged(run_dir, "log.csv", "log_finetune.csv")))
        (run_dir / "phase_finetune.done").touch()
    return net, stats
