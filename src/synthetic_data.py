"""Generate a synthetic driving dataset so the ML pipeline runs without CARLA."""
import argparse
import csv
import math
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_dir


def make_frame(steering, w=config.CAMERA_WIDTH, h=config.CAMERA_HEIGHT):
    """Create a synthetic road image whose lane offset encodes steering."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # sky
    img[:h // 3, :] = (135, 206, 235)
    # road
    img[h // 3:, :] = (60, 60, 60)
    # lane center line shifted by steering
    cx = w // 2 + int(steering * (w // 3))
    for y in range(h // 3, h):
        x = cx + int((y - h // 3) * 0.0)
        if 0 <= x < w:
            img[y, max(0, x - 2):min(w, x + 3)] = (255, 255, 255)
    # side lanes
    for off in (-80, 80):
        for y in range(h // 3, h):
            x = w // 2 + off + int(steering * (w // 4))
            if 0 <= x < w:
                img[y, max(0, x - 1):min(w, x + 2)] = (200, 200, 200)
    # noise
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic driving data")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--out", default=str(config.DATA_DIR / "synthetic"))
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    img_dir = ensure_dir(out_dir / "images")
    csv_path = out_dir / "telemetry.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "image", "timestamp", "speed_kmh", "steering",
                         "throttle", "brake", "x", "y", "z", "yaw",
                         "collision", "lane_invasion"])
        for i in range(args.frames):
            steering = random.uniform(-0.8, 0.8)
            # smooth-ish: bias toward 0
            steering = steering * 0.6
            img = make_frame(steering)
            fname = f"{i:06d}.png"
            cv2.imwrite(str(img_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.writerow([i, fname, 0.0, 25.0, round(steering, 4), 0.5, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0, 0])
    print(f"[synthetic] generated {args.frames} frames in {out_dir}")


if __name__ == "__main__":
    main()
