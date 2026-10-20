"""Image preprocessing and augmentation for behavioral cloning."""
import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def carla_image_to_array(image):
    """Convert carla.Image to RGB numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    array = array[:, :, :3][:, :, ::-1]  # BGR -> RGB
    return array.copy()


def preprocess_image(img, target_w=config.MODEL_INPUT_WIDTH,
                    target_h=config.MODEL_INPUT_HEIGHT):
    """Crop sky/hood, resize, normalize to [-1, 1]. Returns float32 array."""
    h, w = img.shape[:2]
    # crop top 35% and bottom 12%
    top = int(h * 0.35)
    bottom = int(h * 0.88)
    cropped = img[top:bottom, :, :]
    resized = cv2.resize(cropped, (target_w, target_h))
    norm = resized.astype(np.float32) / 127.5 - 1.0
    return norm


def augment(img, steering, rng=None):
    """Simple augmentation: flip, brightness, shadow."""
    if rng is None:
        rng = np.random
    out = img.copy()
    # horizontal flip
    if rng.rand() < 0.5:
        out = np.fliplr(out).copy()
        steering = -steering
    # brightness
    if rng.rand() < 0.5:
        factor = rng.uniform(0.6, 1.4)
        out = np.clip(out * factor, -1.0, 1.0)
    # gaussian blur
    if rng.rand() < 0.2:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out, steering


def batch_preprocess(images, steerings, augment_data=False):
    xs = []
    ys = []
    for img, st in zip(images, steerings):
        p = preprocess_image(img)
        if augment_data:
            p, st = augment(p, st)
        xs.append(p)
        ys.append(st)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
