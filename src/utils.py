"""Utility helpers."""
import os
import random
import json
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))
