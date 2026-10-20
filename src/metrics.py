"""Performance metrics for training and driving runs."""
import json
import os

import numpy as np
import pandas as pd


def training_metrics(history_path):
    with open(history_path) as f:
        h = json.load(f)
    val_loss = h.get("val_loss", [])
    val_mae = h.get("val_mae", [])
    return {
        "final_val_loss": val_loss[-1] if val_loss else None,
        "best_val_loss": min(val_loss) if val_loss else None,
        "final_val_mae": val_mae[-1] if val_mae else None,
        "best_val_mae": min(val_mae) if val_mae else None,
        "epochs_run": len(val_loss),
        "elapsed_seconds": h.get("elapsed_seconds"),
        "train_samples": h.get("train_samples"),
        "val_samples": h.get("val_samples"),
    }


def driving_metrics(telemetry_csv):
    df = pd.read_csv(telemetry_csv)
    if df.empty:
        return {}
    steer = df["steering_pred"].values
    speed = df["speed_kmh"].values
    # smoothness: mean absolute change in steering
    steer_diff = np.abs(np.diff(steer))
    smoothness = float(np.mean(steer_diff)) if len(steer_diff) else 0.0
    # trajectory length
    dx = np.diff(df["x"].values)
    dy = np.diff(df["y"].values)
    dist = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    return {
        "frames": int(len(df)),
        "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) > 1 else 0.0,
        "avg_speed_kmh": float(np.mean(speed)),
        "max_speed_kmh": float(np.max(speed)),
        "speed_std": float(np.std(speed)),
        "steering_mean": float(np.mean(steer)),
        "steering_std": float(np.std(steer)),
        "steering_smoothness_mad": smoothness,
        "distance_m": dist,
        "collisions": int(df["collision"].sum()),
        "lane_invasions": int(df["lane_invasion"].sum()),
        "collision_rate": float(df["collision"].mean()),
        "lane_invasion_rate": float(df["lane_invasion"].mean()),
    }


def compare_models(history_paths):
    rows = []
    for name, path in history_paths.items():
        m = training_metrics(path)
        m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows)
