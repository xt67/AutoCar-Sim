"""Reinforcement-learning baseline for autonomous driving in CARLA.

A minimal DQN agent with a CNN Q-network. State: front-camera image.
Actions: discrete steering bins. Reward: speed - collision/lane penalties.
This is a lightweight baseline for comparison with behavioral cloning, not a
production RL system.
"""
import argparse
import collections
import math
import os
import random
import sys
import time

import carla
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.carla_client import CarlaEnv
from src.preprocess import carla_image_to_array, preprocess_image
from src.utils import ensure_dir, clamp

ACTIONS = np.linspace(-1.0, 1.0, 9).astype(np.float32)  # steering bins


def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def build_qnet(lr=1e-3):
    inputs = keras.Input(shape=(config.MODEL_INPUT_HEIGHT, config.MODEL_INPUT_WIDTH, 3))
    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu")(inputs)
    x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(len(ACTIONS))(x)
    m = keras.Model(inputs, outputs)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def main():
    parser = argparse.ArgumentParser(description="RL (DQN) baseline in CARLA")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", default=str(config.MODELS_DIR / "rl_dqn"))
    parser.add_argument("--map", default=None)
    args = parser.parse_args()

    ensure_dir(args.out)
    env = CarlaEnv(host=config.CARLA_HOST, port=config.CARLA_PORT, sync=True,
                   delta=config.COLLECT_DELTA)
    if args.map:
        env.world = env.client.load_world(args.map)
        env.map = env.world.get_map()
        env.blueprint_lib = env.world.get_blueprint_library()
        s = env.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = config.COLLECT_DELTA
        env.world.apply_settings(s)
    env.spawn_vehicle()
    env.attach_camera()
    env.attach_collision()
    env.attach_lane_invasion()

    qnet = build_qnet()
    target = build_qnet()
    target.set_weights(qnet.get_weights())
    buf = ReplayBuffer()
    gamma = 0.95
    epsilon = 1.0
    batch_size = 32

    episode_rewards = []
    try:
        for ep in range(args.episodes):
            env.vehicle.set_transform(random.choice(env.map.get_spawn_points()))
            # warmup
            for _ in range(5):
                env.tick()
                env.get_image(timeout=1.0)
            total_r = 0.0
            state = None
            img = env.get_image(timeout=2.0)
            if img is not None:
                arr = carla_image_to_array(img)
                state = preprocess_image(arr)
            for step in range(args.steps):
                env.tick()
                img = env.get_image(timeout=2.0)
                if img is None:
                    continue
                arr = carla_image_to_array(img)
                nstate = preprocess_image(arr)

                # epsilon-greedy
                if random.random() < epsilon or state is None:
                    a = random.randrange(len(ACTIONS))
                else:
                    q = qnet.predict(np.expand_dims(state, 0), verbose=0)[0]
                    a = int(np.argmax(q))

                steer = float(ACTIONS[a])
                sp = speed_kmh(env.vehicle)
                throttle = 0.5 if sp < config.TARGET_SPEED_KMH else 0.2
                control = carla.VehicleControl(throttle=throttle, steer=steer)
                control.manual_gear_shift = False
                env.vehicle.apply_control(control)

                collisions = env.drain(env.collision_queue)
                lanes = env.drain(env.lane_queue)
                collision_flag = bool(collisions)
                lane_flag = bool(lanes)

                reward = sp / 10.0
                if collision_flag:
                    reward -= 10.0
                if lane_flag:
                    reward -= 2.0
                done = collision_flag
                total_r += reward

                if state is not None:
                    buf.push(state, a, reward, nstate, done)
                state = nstate

                # train
                if len(buf) >= batch_size:
                    s, a, r, s2, d = buf.sample(batch_size)
                    q_next = target.predict(s2, verbose=0)
                    q_target = r + gamma * np.max(q_next, axis=1) * (1 - d)
                    q_pred = qnet.predict(s, verbose=0)
                    q_pred[np.arange(batch_size), a] = q_target
                    qnet.train_on_batch(s, q_pred)

                if done:
                    break

            episode_rewards.append(total_r)
            epsilon = max(0.1, epsilon * 0.9)
            # update target
            target.set_weights(qnet.get_weights())
            print(f"[rl] episode {ep+1}/{args.episodes} reward={total_r:.2f} "
                  f"epsilon={epsilon:.3f}")
    except KeyboardInterrupt:
        print("\n[rl] interrupted.")
    finally:
        qnet.save(os.path.join(args.out, "dqn.keras"))
        import json
        with open(os.path.join(args.out, "rewards.json"), "w") as f:
            json.dump({"episode_rewards": episode_rewards}, f)
        env.cleanup()
    print(f"[rl] saved model to {args.out}")


if __name__ == "__main__":
    main()
