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
from src.preprocess import preprocess_image, preprocess_depth, augment, brighten_rgb
from src.utils import ensure_dir, set_seed, save_json


from src.dataset import load_dataset, get_split, save_split


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
            out_x.append(brighten_rgb(img, f))
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
    parser.add_argument("--split-gap", type=int, default=30, help="Frames excluded after each temporal split boundary")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    exp_name = args.experiment
    out_dir = args.out_dir or str(config.MODELS_DIR / exp_name)
    ensure_dir(out_dir)
    ensure_dir(config.LOGS_DIR / exp_name)
    ensure_dir(config.REPORTS_DIR / exp_name)

    if args.model == "multisensor":
        args.depth = True
    if args.depth and args.model not in ("nvidia", "multisensor"):
        parser.error("--depth requires nvidia or multisensor")
    X, y, rows, fingerprint = load_dataset(args.data, use_depth=args.depth)
    sessions = rows["session_id"].astype(str).values if "session_id" in rows else None
    parts = get_split(len(X), sessions=sessions, gap=args.split_gap)
    train_idx, val_idx, test_idx = parts
    save_split(os.path.join(out_dir, "splits.json"), parts, fingerprint, args.depth, args.split_gap, rows)
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
    model = keras.models.load_model(ckpt_path)

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
