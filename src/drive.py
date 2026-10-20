"""Run autonomous driving using a trained steering model."""
import argparse
import csv
import math
import os
import sys
import time

import carla
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.carla_client import CarlaEnv
from src.preprocess import carla_image_to_array, preprocess_image
from src.utils import ensure_dir, clamp


def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


class SpeedController:
    def __init__(self, target_kmh, kp=0.15, ki=0.02, kd=0.05):
        self.target = target_kmh
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, speed_kmh, dt=0.05):
        error = self.target - speed_kmh
        self.integral = clamp(self.integral + error * dt, -10, 10)
        derivative = (error - self.prev_error) / max(dt, 1e-3)
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        if out >= 0:
            throttle = clamp(out, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = clamp(-out, 0.0, 1.0)
        return throttle, brake


def main():
    parser = argparse.ArgumentParser(description="Autonomous driving with trained model")
    parser.add_argument("--model", default=str(config.MODELS_DIR / "steering_model.keras"))
    parser.add_argument("--host", default=config.CARLA_HOST)
    parser.add_argument("--port", type=int, default=config.CARLA_PORT)
    parser.add_argument("--frames", type=int, default=config.MAX_DRIVE_FRAMES)
    parser.add_argument("--out", default=str(config.LOGS_DIR / "drive_telemetry.csv"))
    parser.add_argument("--target-speed", type=float, default=config.TARGET_SPEED_KMH)
    parser.add_argument("--map", default=None)
    parser.add_argument("--traffic", type=int, default=config.TRAFFIC_VEHICLES)
    parser.add_argument("--camera-w", type=int, default=config.CAMERA_WIDTH)
    parser.add_argument("--camera-h", type=int, default=config.CAMERA_HEIGHT)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    print(f"[drive] loading model {args.model}")
    model = tf.keras.models.load_model(args.model)

    env = CarlaEnv(host=args.host, port=args.port, sync=True,
                   delta=config.COLLECT_DELTA, no_rendering=args.no_render)
    if args.map:
        env.world = env.client.load_world(args.map)
        env.map = env.world.get_map()
        env.blueprint_lib = env.world.get_blueprint_library()
        s = env.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = config.COLLECT_DELTA
        env.world.apply_settings(s)

    env.spawn_vehicle()
    env.attach_camera(width=args.camera_w, height=args.camera_h)
    env.attach_collision()
    env.attach_lane_invasion()
    if args.traffic > 0:
        env.spawn_traffic(args.traffic)

    speed_ctrl = SpeedController(args.target_speed)
    ensure_dir(os.path.dirname(args.out))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp", "speed_kmh", "steering_pred",
                         "throttle", "brake", "x", "y", "z", "yaw",
                         "collision", "lane_invasion"])

        # warmup
        for _ in range(10):
            env.tick()
            env.get_image(timeout=1.0)

        collected = 0
        t0 = time.time()
        stop_file = os.path.join(os.path.dirname(args.out), "STOP")
        try:
            while collected < args.frames:
                if os.path.exists(stop_file):
                    print("[drive] STOP file detected, shutting down.")
                    break
                env.tick()
                image = env.get_image(timeout=2.0)
                if image is None:
                    continue

                arr = carla_image_to_array(image)
                proc = preprocess_image(arr)
                inp = np.expand_dims(proc, axis=0)
                steer = float(model.predict(inp, verbose=0)[0][0])
                steer = clamp(steer, -1.0, 1.0)

                sp = speed_kmh(env.vehicle)
                throttle, brake = speed_ctrl.update(sp)

                control = carla.VehicleControl(throttle=throttle, steer=steer,
                                              brake=brake)
                control.manual_gear_shift = False
                env.vehicle.apply_control(control)

                collisions = env.drain(env.collision_queue)
                lanes = env.drain(env.lane_queue)
                collision_flag = 1 if collisions else 0
                lane_flag = 1 if lanes else 0

                tr = env.vehicle.get_transform()
                writer.writerow([collected, time.time(), round(sp, 3), round(steer, 4),
                                round(throttle, 4), round(brake, 4),
                                round(tr.location.x, 3), round(tr.location.y, 3),
                                round(tr.location.z, 3), round(tr.rotation.yaw, 3),
                                collision_flag, lane_flag])
                f.flush()

                collected += 1
                if collected % 50 == 0:
                    print(f"[drive] {collected}/{args.frames}  speed={sp:.1f} km/h  "
                          f"steer={steer:.3f}  collisions={collision_flag}")

                if collision_flag:
                    print("[drive] collision detected!")
        except KeyboardInterrupt:
            print("\n[drive] interrupted by user (safe shutdown).")
        finally:
            try:
                env.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            except Exception:
                pass
            print(f"[drive] done. {collected} frames. log: {args.out}")
    env.cleanup()


if __name__ == "__main__":
    main()
