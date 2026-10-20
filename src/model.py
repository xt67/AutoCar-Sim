"""NVIDIA end-to-end CNN for steering prediction."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_nvidia_model(input_shape=(config.MODEL_INPUT_HEIGHT,
                                    config.MODEL_INPUT_WIDTH, 3),
                       lr=config.LEARNING_RATE):
    inputs = keras.Input(shape=input_shape)
    x = layers.Lambda(lambda t: t)(inputs)  # placeholder for normalization (done in preprocess)
    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", padding="valid")(x)
    x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu", padding="valid")(x)
    x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu", padding="valid")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(10, activation="relu")(x)
    outputs = layers.Dense(1, name="steering")(x)
    model = keras.Model(inputs, outputs, name="nvidia_steering")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae"])
    return model


def build_light_model(input_shape=(config.MODEL_INPUT_HEIGHT,
                                  config.MODEL_INPUT_WIDTH, 3),
                     lr=config.LEARNING_RATE):
    """Smaller model for fast CPU training / comparison."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, name="steering")(x)
    model = keras.Model(inputs, outputs, name="light_steering")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae"])
    return model


MODEL_REGISTRY = {
    "nvidia": build_nvidia_model,
    "light": build_light_model,
}
