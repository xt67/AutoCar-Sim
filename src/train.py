"""Train a behavioral-cloning steering model."""
import argparse
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
from src.model import build_nvidia_model, build_light_model, MODEL_REGISTRY
from src.preprocess import preprocess_image
from src.utils import ensure_dir, set_seed, save_json


def load_dataset(data_dir):
    data_dir = os.path.abspath(data_dir)
    csv_path = os.path.join(data_dir, "telemetry.csv")
    img_dir = os.path.join(data_dir, "images")
    df = pd.read_csv(csv_path)
    images = []
    steerings = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, str(row["image"]))
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        steerings.append(float(row["steering"]))
    print(f"[train] loaded {len(images)} samples from {data_dir}")
    return images, np.array(steerings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train steering model")
    parser.add_argument("--data", default=str(config.DATA_DIR / "synthetic"))
    parser.add_argument("--model", default="nvidia", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--out", default=str(config.MODELS_DIR / "steering_model.keras"))
    parser.add_argument("--history", default=str(config.LOGS_DIR / "history.json"))
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data first if missing")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    ensure_dir(config.MODELS_DIR)
    ensure_dir(config.LOGS_DIR)

    if args.synthetic or not os.path.exists(os.path.join(args.data, "telemetry.csv")):
        print("[train] generating synthetic dataset...")
        from src.synthetic_data import main as gen
        # emulate
        import subprocess
        subprocess.run([sys.executable, "-m", "src.synthetic_data",
                        "--frames", "2000", "--out", args.data], check=False)
        # fallback direct
        if not os.path.exists(os.path.join(args.data, "telemetry.csv")):
            from src.synthetic_data import make_frame
            # direct generate
            import csv
            os.makedirs(os.path.join(args.data, "images"), exist_ok=True)
            with open(os.path.join(args.data, "telemetry.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "image", "timestamp", "speed_kmh", "steering",
                            "throttle", "brake", "x", "y", "z", "yaw",
                            "collision", "lane_invasion"])
                for i in range(2000):
                    st = random.uniform(-0.8, 0.8) * 0.6
                    img = make_frame(st)
                    fname = f"{i:06d}.png"
                    cv2.imwrite(os.path.join(args.data, "images", fname),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    w.writerow([i, fname, 0.0, 25.0, round(st, 4), 0.5, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0, 0])

    images, steerings = load_dataset(args.data)

    # preprocess
    X = np.array([preprocess_image(img) for img in images], dtype=np.float32)
    y = steerings.astype(np.float32)

    # augmentation via in-memory expansion
    if args.augment:
        aug_imgs = []
        aug_ys = []
        for img, st in zip(X, y):
            aug_imgs.append(img)
            aug_ys.append(st)
            # flip
            aug_imgs.append(np.fliplr(img).copy())
            aug_ys.append(-st)
        X = np.array(aug_imgs, dtype=np.float32)
        y = np.array(aug_ys, dtype=np.float32)
        print(f"[train] augmented dataset size: {len(X)}")

    # split
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - config.VAL_SPLIT))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = MODEL_REGISTRY[args.model](lr=args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, save_best_only=True,
                                       monitor="val_loss", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=4, verbose=1),
        keras.callbacks.CSVLogger(str(config.LOGS_DIR / "training.csv")),
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
    print(f"[train] finished in {elapsed:.1f}s")

    # save final model (best already saved via checkpoint)
    model.save(args.out)
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    hist["elapsed_seconds"] = elapsed
    hist["train_samples"] = len(X_train)
    hist["val_samples"] = len(X_val)
    hist["model"] = args.model
    save_json(hist, args.history)
    print(f"[train] model saved to {args.out}")
    print(f"[train] history saved to {args.history}")


if __name__ == "__main__":
    main()
