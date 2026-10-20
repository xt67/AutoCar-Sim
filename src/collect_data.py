"""Collect driving data from CARLA using a BehaviorAgent as expert."""
import argparse
import csv
import math
import os
import random
import sys
import time

import carla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# CARLA PythonAPI (agents module)
_carla_api = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "PythonAPI", "carla")
if _carla_api not in sys.path:
    sys.path.insert(0, _carla_api)
import config
from src.carla_client import CarlaEnv
from src.preprocess import carla_image_to_array
from src.utils import ensure_dir


def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def main():
    parser = argparse.ArgumentParser(description="Collect CARLA driving data")
    parser.add_argument("--host", default=config.CARLA_HOST)
    parser.add_argument("--port", type=int, default=config.CARLA_PORT)
    parser.add_argument("--frames", type=int, default=config.COLLECT_FRAMES)
    parser.add_argument("--out", default=str(config.DATA_DIR / "collected"))
    parser.add_argument("--behavior", default=config.DEFAULT_BEHAVIOR,
                        choices=["careful", "normal", "aggressive"])
    parser.add_argument("--map", default=None, help="CARLA map name e.g. Town03")
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--camera-w", type=int, default=config.CAMERA_WIDTH)
    parser.add_argument("--camera-h", type=int, default=config.CAMERA_HEIGHT)
    args = parser.parse_args()

    env = CarlaEnv(host=args.host, port=args.port, sync=True, delta=config.COLLECT_DELTA)
    if args.map:
        env.world = env.client.load_world(args.map)
        env.map = env.world.get_map()
        env.blueprint_lib = env.world.get_blueprint_library()
        # re-apply sync settings
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

    from agents.navigation.behavior_agent import BehaviorAgent
    agent = BehaviorAgent(env.vehicle, behavior=args.behavior)

    spawn_points = env.map.get_spawn_points()
    agent.set_destination(random.choice(spawn_points).location)

    out_dir = ensure_dir(args.out)
    img_dir = ensure_dir(out_dir / "images")
    csv_path = out_dir / "telemetry.csv"

    print(f"[collect] spawning expert agent, target {args.frames} frames -> {out_dir}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "image", "timestamp", "speed_kmh", "steering",
                         "throttle", "brake", "x", "y", "z", "yaw",
                         "collision", "lane_invasion"])

        # warmup
        for _ in range(10):
            env.tick()
            env.get_image(timeout=1.0)

        collected = 0
        t0 = time.time()
        while collected < args.frames:
            env.tick()
            image = env.get_image(timeout=2.0)
            if image is None:
                continue

            if agent.done():
                agent.set_destination(random.choice(spawn_points).location)

            control = agent.run_step()
            control.manual_gear_shift = False
            env.vehicle.apply_control(control)

            # collision / lane
            collisions = env.drain(env.collision_queue)
            lanes = env.drain(env.lane_queue)
            collision_flag = 1 if collisions else 0
            lane_flag = 1 if lanes else 0

            fname = f"{collected:06d}.png"
            image.save_to_disk(str(img_dir / fname))

            tr = env.vehicle.get_transform()
            row = [collected, fname, time.time(), round(speed_kmh(env.vehicle), 3),
                   round(float(control.steer), 4), round(float(control.throttle), 4),
                   round(float(control.brake), 4),
                   round(tr.location.x, 3), round(tr.location.y, 3),
                   round(tr.location.z, 3), round(tr.rotation.yaw, 3),
                   collision_flag, lane_flag]
            writer.writerow(row)
            f.flush()

            collected += 1
            if collected % 100 == 0:
                elapsed = time.time() - t0
                print(f"[collect] {collected}/{args.frames}  "
                      f"{collected/elapsed:.1f} fps  steer={control.steer:.3f}")

    env.cleanup()
    print(f"[collect] done. {collected} frames saved to {out_dir}")


if __name__ == "__main__":
    main()
