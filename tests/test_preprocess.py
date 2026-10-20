import numpy as np

import config
from src.preprocess import preprocess_image, augment


def test_preprocess_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = preprocess_image(img)
    assert out.shape == (config.MODEL_INPUT_HEIGHT, config.MODEL_INPUT_WIDTH, 3)
    assert out.dtype == np.float32
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_augment_flip_steering():
    img = np.zeros((config.MODEL_INPUT_HEIGHT, config.MODEL_INPUT_WIDTH, 3), dtype=np.float32)
    out, steer = augment(img, 0.5, rng=np.random.default_rng(0))
    # either flipped or not; steering sign consistent with flip
    # just ensure finite
    assert np.isfinite(steer)
    assert out.shape == img.shape
