"""Shared dataset loading and auditable, temporally separated evaluation splits."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocess import carla_depth_to_array, preprocess_depth, preprocess_image


def load_dataset(data_dir, use_depth=False):
    root = Path(data_dir)
    csv_bytes = (root / "telemetry.csv").read_bytes()
    digest = hashlib.sha256(csv_bytes)
    rows = pd.read_csv(root / "telemetry.csv")
    if rows.empty or not {"image", "steering"}.issubset(rows.columns):
        raise ValueError("telemetry.csv must contain image and steering samples")
    if rows["image"].duplicated().any():
        raise ValueError("Duplicate image identifiers in dataset")
    images = []
    for _, row in rows.iterrows():
        channels = []
        for column in (["image", "depth_image"] if use_depth else ["image"]):
            name = row.get(column)
            if pd.isna(name) or not str(name).strip():
                raise ValueError(f"Missing {column} for {row['image']}; recollect paired depth data")
            path = (root / "images" / str(name)).resolve()
            if not path.is_relative_to((root / "images").resolve()):
                raise ValueError("Image path must stay inside dataset images directory")
            if not path.is_file():
                raise ValueError(f"Missing {column}: {name}")
            raw = path.read_bytes()
            digest.update(raw)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable {column}: {name}")
            channels.append(preprocess_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            if column == "image" else preprocess_depth(carla_depth_to_array(image))[..., None])
        images.append(np.concatenate(channels, axis=-1))
    targets = rows["steering"].to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all() or (np.abs(targets) > 1).any():
        raise ValueError("Steering labels must be finite values in [-1, 1]")
    return np.array(images, dtype=np.float32), targets, rows, digest.hexdigest()


def get_split(n, seed=42, sessions=None, gap=30):
    """Whole recording splits when possible; otherwise chronological with an embargo.

    `seed` remains accepted for callers, but adjacent frames are never shuffled.
    A single recording is weaker evidence than independent held-out drives.
    """
    if n < 3 or gap < 0:
        raise ValueError("Need at least three samples and a nonnegative embargo")
    if sessions is not None:
        sessions = np.asarray(sessions, dtype=str)
        if len(sessions) != n:
            raise ValueError("Session labels must match dataset length")
        groups = list(dict.fromkeys(sessions))
        if len(groups) >= 3:
            a = min(max(1, int(len(groups) * .7)), len(groups) - 2)
            b = min(a + max(1, int(len(groups) * .15)), len(groups) - 1)
            return tuple(np.flatnonzero(np.isin(sessions, g)) for g in
                         (groups[:a], groups[a:b], groups[b:]))
    a, b = int(n * .7), int(n * .85)
    parts = (np.arange(a), np.arange(a + gap, b), np.arange(b + gap, n))
    if any(len(p) == 0 for p in parts):
        raise ValueError("Dataset too small for temporal embargo; collect more data or explicitly reduce --split-gap")
    return parts


def save_split(path, parts, fingerprint, use_depth, gap, rows):
    manifest = {"version": 1, "fingerprint": fingerprint, "use_depth": use_depth,
                "gap": gap, "samples": len(rows),
                "strategy": "recording" if "session_id" in rows and rows.session_id.nunique() >= 3 else "chronological-embargo",
                **{name: ids.tolist() for name, ids in zip(("train", "val", "test"), parts)}}
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def held_out_indices(path, fingerprint, n):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("fingerprint") != fingerprint or manifest.get("samples") != n:
        raise ValueError("Dataset differs from training manifest; use the original dataset or explicitly evaluate an external dataset")
    parts = [manifest[name] for name in ("train", "val", "test")]
    all_ids = [i for part in parts for i in part]
    if not all(parts) or len(set(all_ids)) != len(all_ids) or any(type(i) is not int or i < 0 or i >= n for i in all_ids):
        raise ValueError("Invalid or overlapping split indices")
    return np.array(manifest["test"], dtype=int)
