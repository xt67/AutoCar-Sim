"""Collect driving data.

Two modes:

* ``manual`` (default): attach a front camera to the vehicle that
  ``manual_control.py`` is already driving (role_name == "hero"). No vehicle is
  spawned and no control is applied by this script. Steering/throttle/brake/speed
  labels are read from the vehicle with a documented one-tick delay.

* ``autopilot``: spawn a vehicle and drive it with CARLA's BehaviorAgent as an
  automated expert. Useful for generating large datasets without a human driver.
"""
import argparse
import csv
import math
import os
import random
import sys
import time

import carla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_carla_api = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "PythonAPI", "carla")
if _carla_api not in sys.path:
    sys.path.insert(0, _carla_api)
import config
from src.carla_client import CarlaEnv
from src.utils import ensure_dir


def speed_kmh(vehicle):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def find_hero(world, role_name="hero", timeout=30.0):
    """Wait for a vehicle with the given role_name to appear."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == role_name:
                return actor
        time.sleep(0.5)
    return None


def run_manual(args):
    # Connect WITHOUT forcing sync mode: manual_control.py owns the clock.
    client = carla.Client(args.host, args.port)
    client.set_timeout(config.CARLA_TIMEOUT)
    world = client.get_world()

    print(f"[collect:manual] waiting for hero vehicle (role_name='{args.role_name}')...")
    vehicle = find_hero(world, args.role_name, timeout=args.wait)
    if vehicle is None:
        raise RuntimeError(
            f"No vehicle with role_name='{args.role_name}' found. "
            "Start manual_control.py first (it spawns a 'hero' vehicle). "
            "This collector never spawns a vehicle or applies control."
        )
    print(f"[collect:manual] found vehicle id={vehicle.id}")

    # Attach a front camera to the hero (read-only; does not affect control).
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.camera_w))
    cam_bp.set_attribute("image_size_y", str(args.camera_h))
    cam_bp.set_attribute("fov", str(config.CAMERA_FOV))
    transform = carla.Transform(carla.Location(x=2.0, z=1.4),
                                carla.Rotation(pitch=-15))
    camera = world.spawn_actor(cam_bp, transform, attach_to=vehicle)

    image_queue = __import__("queue").Queue()
    camera.listen(image_queue.put)

    out_dir = ensure_dir(args.out)
    img_dir = ensure_dir(out_dir / "images")
    csv_path = out_dir / "telemetry.csv"

    print("[collect:manual] recording. Ctrl+C to stop.")
    print("NOTE: labels come from vehicle.get_control() (last applied by "
          "manual_control.py) with a ~one-tick delay; see README limitation note.")

    collected = 0
    t0 = time.time()
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "image", "timestamp", "speed_kmh",
                             "steering", "throttle", "brake", "x", "y", "z",
                             "yaw", "collision", "lane_invasion"])
            while collected < args.frames:
                # Wait for next world tick (driven by manual_control.py).
                world.wait_for_tick()
                try:
                    image = image_queue.get(timeout=2.0)
                except Exception:
                    continue

                control = vehicle.get_control()
                tr = vehicle.get_transform()
                fname = f"{collected:06d}.png"
                image.save_to_disk(str(img_dir / fname))
                writer.writerow([collected, fname, time.time(),
                                 round(speed_kmh(vehicle), 3),
                                 round(float(control.steer), 4),
                                 round(float(control.throttle), 4),
                                 round(float(control.brake), 4),
                                 round(tr.location.x, 3), round(tr.location.y, 3),
                                 round(tr.location.z, 3), round(tr.rotation.yaw, 3),
                                 0, 0])
                f.flush()
                collected += 1
                if collected % 50 == 0:
                    print(f"[collect:manual] {collected}/{args.frames}")
    except KeyboardInterrupt:
        print("\n[collect:manual] interrupted by user.")
    finally:
        try:
            camera.stop()
            camera.destroy()
        except Exception:
            pass
    print(f"[collect:manual] saved {collected} frames to {out_dir}")


def run_autopilot(args):
    env = CarlaEnv(host=args.host, port=args.port, sync=True,
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

    print(f"[collect:autopilot] target {args.frames} frames -> {out_dir}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "image", "timestamp", "speed_kmh", "steering",
                         "throttle", "brake", "x", "y", "z", "yaw",
                         "collision", "lane_invasion"])
        for _ in range(10):
            env.tick()
            env.get_image(timeout=1.0)
        collected = 0
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
            collisions = env.drain(env.collision_queue)
            lanes = env.drain(env.lane_queue)
            fname = f"{collected:06d}.png"
            image.save_to_disk(str(img_dir / fname))
            tr = env.vehicle.get_transform()
            writer.writerow([collected, fname, time.time(),
                             round(speed_kmh(env.vehicle), 3),
                             round(float(control.steer), 4),
                             round(float(control.throttle), 4),
                             round(float(control.brake), 4),
                             round(tr.location.x, 3), round(tr.location.y, 3),
                             round(tr.location.z, 3), round(tr.rotation.yaw, 3),
                             1 if collisions else 0, 1 if lanes else 0])
            f.flush()
            collected += 1
            if collected % 100 == 0:
                print(f"[collect:autopilot] {collected}/{args.frames}")
    env.cleanup()
    print(f"[collect:autopilot] done. {collected} frames saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect CARLA driving data")
    parser.add_argument("--mode", default="manual", choices=["manual", "autopilot"])
    parser.add_argument("--host", default=config.CARLA_HOST)
    parser.add_argument("--port", type=int, default=config.CARLA_PORT)
    parser.add_argument("--frames", type=int, default=config.COLLECT_FRAMES)
    parser.add_argument("--out", default=str(config.DATA_DIR / "collected"))
    parser.add_argument("--behavior", default=config.DEFAULT_BEHAVIOR,
                        choices=["careful", "normal", "aggressive"])
    parser.add_argument("--map", default=None)
    parser.add_argument("--traffic", type=int, default=0)
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--wait", type=float, default=30.0)
    parser.add_argument("--camera-w", type=int, default=config.CAMERA_WIDTH)
    parser.add_argument("--camera-h", type=int, default=config.CAMERA_HEIGHT)
    args = parser.parse_args()

    if args.mode == "manual":
        run_manual(args)
    else:
        run_autopilot(args)


if __name__ == "__main__":
    main()
