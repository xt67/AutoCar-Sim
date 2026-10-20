"""Train a behavioral-cloning steering model (baseline or augmented)."""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import utils
from src.model import MODEL_REGISTRY
from src.preprocess import preprocess_image, preprocess_depth, augment
from src.utils import ensure_dir, set_seed, save_json


def load_dataset(data_dir, use_depth=False):
    data_dir = os.path.abspath(data_dir)
    csv_path = os.path.join(data_dir, "telemetry.csv")
    img_dir = os.path.join(data_dir, "images")
    df = pd.read_csv(csv_path)
    samples, steerings = [], []
    has_depth = "depth_image" in df.columns
    for _, row in df.iterrows():
        img = cv2.imread(os.path.join(img_dir, str(row["image"])))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if use_depth and has_depth and pd.notna(row.get("depth_image", None)) and str(row["depth_image"]).strip():
            dpath = os.path.join(img_dir, str(row["depth_image"]))
            dimg = cv2.imread(dpath, cv2.IMREAD_COLOR)
            if dimg is not None:
                from src.preprocess import carla_depth_to_array, preprocess_depth
                depth_arr = carla_depth_to_array(dimg)
                samples.append((rgb, depth_arr))
            else:
                samples.append((rgb, None))
        else:
            samples.append((rgb, None))
        steerings.append(float(row["steering"]))
    print(f"[train] loaded {len(samples)} samples from {data_dir} (depth={use_depth and has_depth})")
    return samples, np.array(steerings, dtype=np.float32)


def get_split(n, seed=config.RANDOM_SEED):
    """Deterministic 70/15/15 split, persisted to data/splits.json."""
    splits_path = os.path.join(config.DATA_DIR, "splits.json") if hasattr(config, "DATA_DIR") else None
    # resolve relative to project root
    splits_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            sp = json.load(f)
        if sp.get("n") == n and sp.get("seed") == seed:
            print(f"[train] reusing saved split from {splits_path}")
            return np.array(sp["train"]), np.array(sp["val"]), np.array(sp["test"])
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump({"n": n, "seed": seed,
                   "train": train_idx.tolist(),
                   "val": val_idx.tolist(),
                   "test": test_idx.tolist()}, f)
    print(f"[train] saved split to {splits_path}")
    return train_idx, val_idx, test_idx


def augment_train(X, y):
    out_x, out_y = [], []
    for img, st in zip(X, y):
        out_x.append(img)
        out_y.append(st)
        # flip
        out_x.append(np.fliplr(img).copy())
        out_y.append(-st)
        # brightness
        if np.random.rand() < 0.5:
            f = np.random.uniform(0.7, 1.3)
            out_x.append(np.clip(img * f, -1.0, 1.0).astype(np.float32))
            out_y.append(st)
    return np.array(out_x, dtype=np.float32), np.array(out_y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train steering model")
    parser.add_argument("--data", default=str(config.DATA_DIR / "synthetic"))
    parser.add_argument("--model", default="nvidia", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--depth", action="store_true",
                        help="Use depth channel (multi-sensor model)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    exp_name = args.experiment
    out_dir = args.out_dir or str(config.MODELS_DIR / exp_name)
    ensure_dir(out_dir)
    ensure_dir(config.LOGS_DIR / exp_name)
    ensure_dir(config.REPORTS_DIR / exp_name)

    samples, steerings = load_dataset(args.data, use_depth=args.depth)
    if args.depth:
        X = np.array([np.concatenate([preprocess_image(rgb),
                                      preprocess_depth(dep)[:, :, None]], axis=-1)
                      for (rgb, dep) in samples], dtype=np.float32)
    else:
        X = np.array([preprocess_image(rgb) for (rgb, _) in samples], dtype=np.float32)
    y = steerings.astype(np.float32)

    train_idx, val_idx, test_idx = get_split(len(X), config.RANDOM_SEED)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if args.augment:
        X_train, y_train = augment_train(X_train, y_train)
        print(f"[train] augmented train size: {len(X_train)}")

    if args.depth and args.model == "nvidia":
        args.model = "multisensor"
    model = MODEL_REGISTRY[args.model](lr=args.lr)
    model.summary()

    ckpt_path = os.path.join(out_dir, "model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                       monitor="val_loss", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=4, verbose=1),
        keras.callbacks.CSVLogger(str(config.LOGS_DIR / exp_name / "training.csv")),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    model.save(ckpt_path)

    # Evaluation on val + test
    def ev(split_name, Xs, ys):
        pred = model.predict(Xs, verbose=0).flatten()
        mae = float(np.mean(np.abs(pred - ys)))
        mse = float(np.mean((pred - ys) ** 2))
        return {"mae": mae, "mse": mse, "n": len(ys)}

    metrics = {
        "experiment": exp_name,
        "augment": bool(args.augment),
        "model": args.model,
        "epochs_run": len(history.history.get("val_loss", [])),
        "elapsed_seconds": elapsed,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "val": ev("val", X_val, y_val),
        "test": ev("test", X_test, y_test),
        "best_val_loss": float(min(history.history.get("val_loss", [float("inf")]))),
    }
    save_json(metrics, str(config.LOGS_DIR / exp_name / "metrics.json"))
    save_json({k: [float(x) for x in v] for k, v in history.history.items()},
              str(config.LOGS_DIR / exp_name / "history.json"))

    # Save experiment config
    cfg = vars(args).copy()
    cfg["random_seed"] = config.RANDOM_SEED
    cfg["data_dir"] = os.path.abspath(args.data)
    cfg["split_sizes"] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    save_json(cfg, str(config.LOGS_DIR / exp_name / "config.json"))

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = config.REPORTS_DIR / exp_name
    ensure_dir(rep)
    # loss curve
    fig, ax = plt.subplots()
    ax.plot(history.history.get("loss", []), label="train")
    ax.plot(history.history.get("val_loss", []), label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.legend(); ax.set_title("Loss")
    fig.savefig(rep / "loss_curve.png"); plt.close(fig)
    # prediction vs ground truth (test)
    pred_test = model.predict(X_test, verbose=0).flatten()
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred_test, s=8, alpha=0.5)
    ax.plot([-1, 1], [-1, 1], "r--")
    ax.set_xlabel("ground truth"); ax.set_ylabel("prediction")
    ax.set_title("Prediction vs Ground Truth (test)")
    fig.savefig(rep / "pred_vs_gt.png"); plt.close(fig)
    # steering distribution
    fig, ax = plt.subplots()
    ax.hist(y_test, bins=30, alpha=0.6, label="ground truth")
    ax.hist(pred_test, bins=30, alpha=0.6, label="prediction")
    ax.legend(); ax.set_title("Steering distribution (test)")
    fig.savefig(rep / "steering_dist.png"); plt.close(fig)

    print(f"[train] experiment '{exp_name}' done in {elapsed:.1f}s")
    print(f"[train] metrics: {json.dumps(metrics['test'])}")
    print(f"[train] model: {ckpt_path}")


if __name__ == "__main__":
    main()
