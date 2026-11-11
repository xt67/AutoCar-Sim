"""Evaluate a trained model on a dataset and produce plots."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import preprocess_image
from src.utils import ensure_dir


from pathlib import Path
from src.dataset import load_dataset, held_out_indices


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained steering model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(config.REPORTS_DIR / "eval"))
    parser.add_argument("--external-dataset", action="store_true", help="Explicitly evaluate all samples from a separate unseen dataset")
    args = parser.parse_args()

    ensure_dir(args.out)
    model = tf.keras.models.load_model(args.model, safe_mode=False)
    use_depth = model.input_shape[-1] == 4
    X, y, rows, fingerprint = load_dataset(args.data, use_depth=use_depth)
    partition = "external" if args.external_dataset else "test"
    if not args.external_dataset:
        manifest = Path(args.model).with_name("splits.json")
        if not manifest.exists():
            parser.error("Missing training splits.json; retrain with an auditable split or supply --external-dataset for genuinely unseen data")
        indices = held_out_indices(manifest, fingerprint, len(X))
        X, y = X[indices], y[indices]
    pred = model.predict(X, verbose=0).flatten()
    mae = float(np.mean(np.abs(pred - y)))
    mse = float(np.mean((pred - y) ** 2))
    rmse = float(np.sqrt(mse))
    print(f"MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  n={len(y)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(y, pred, s=8, alpha=0.5)
    ax.plot([-1, 1], [-1, 1], "r--")
    ax.set_xlabel("ground truth"); ax.set_ylabel("prediction")
    fig.savefig(os.path.join(args.out, "pred_vs_gt.png")); plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(y, bins=30, alpha=0.6, label="ground truth")
    ax.hist(pred, bins=30, alpha=0.6, label="prediction")
    ax.legend()
    fig.savefig(os.path.join(args.out, "steering_dist.png")); plt.close(fig)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"mae": mae, "mse": mse, "rmse": rmse, "n": len(y), "partition": partition, "fingerprint": fingerprint}, f, indent=2)
    print(f"Plots saved to {args.out}")


if __name__ == "__main__":
    main()
