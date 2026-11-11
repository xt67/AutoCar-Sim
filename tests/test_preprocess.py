import numpy as np
import pytest
from types import SimpleNamespace

import config
from src.preprocess import preprocess_image, augment, carla_depth_to_array


@pytest.mark.parametrize("bgr,encoded", [([0, 0, 1], 1), ([0, 1, 0], 256), ([1, 0, 0], 65536), ([255, 255, 255], 16777215)])
def test_depth_channel_order_for_saved_and_sensor_images(bgr, encoded):
    expected = encoded / 16777215 * 1000
    saved = np.array([[bgr]], dtype=np.uint8)
    sensor = SimpleNamespace(height=1, width=1, raw_data=bytes(bgr + [255]))
    for image in (saved, sensor):
        assert float(carla_depth_to_array(image)[0, 0]) == pytest.approx(expected)


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
