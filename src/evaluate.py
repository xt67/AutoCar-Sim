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


def load_data(data_dir):
    csv_path = os.path.join(data_dir, "telemetry.csv")
    img_dir = os.path.join(data_dir, "images")
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        img = cv2.imread(os.path.join(img_dir, str(row["image"])))
        if img is None:
            continue
        X.append(preprocess_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        y.append(float(row["steering"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained steering model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(config.REPORTS_DIR / "eval"))
    args = parser.parse_args()

    ensure_dir(args.out)
    X, y = load_data(args.data)
    if len(X) == 0:
        print("No data found. Provide a dataset with telemetry.csv + images/.")
        return
    model = tf.keras.models.load_model(args.model)
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
        json.dump({"mae": mae, "mse": mse, "rmse": rmse, "n": len(y)}, f, indent=2)
    print(f"Plots saved to {args.out}")


if __name__ == "__main__":
    main()
