import numpy as np

import config
from src.model import build_nvidia_model, build_light_model


def test_models_predict():
    for builder in (build_nvidia_model, build_light_model):
        m = builder()
        x = np.zeros((2, config.MODEL_INPUT_HEIGHT, config.MODEL_INPUT_WIDTH, 3), dtype=np.float32)
        y = m.predict(x, verbose=0)
        assert y.shape == (2, 1)
