"""Record labels from the vehicle already driven with CARLA manual_control.py.

This collector never creates or controls a vehicle.  It attaches an RGB camera
to the existing ``hero`` vehicle and saves images plus control labels under the
college project's ``data/manual_collected`` folder.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
from pathlib import Path

import carla


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manual_collected" / "run_001"
# This schema is intentionally compatible with src.train.  The training loader
# needs the ``image`` filename and ``steering`` label; the remaining fields are
# useful for quality checks and the dashboard.
COLUMNS = [
    "frame", "image", "timestamp", "speed_kmh", "steering", "throttle",
    "brake", "x", "y", "z", "yaw", "collision", "lane_invasion",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--vehicle-id", type=int, help="existing CARLA vehicle actor ID")
    selector.add_argument("--role-name", default="hero", help="existing vehicle role name (default: hero)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fov", type=float, default=90.0)
    return parser.parse_args()


def select_vehicle(world: carla.World, args: argparse.Namespace) -> carla.Vehicle:
    """Find an existing vehicle only; do not create an actor."""
    if args.vehicle_id is not None:
        actor = world.get_actor(args.vehicle_id)
        if actor is None or not isinstance(actor, carla.Vehicle):
            raise RuntimeError(f"Vehicle actor {args.vehicle_id} was not found.")
        return actor

    vehicles = list(world.get_actors().filter("vehicle.*"))
    matches = [v for v in vehicles if v.attributes.get("role_name") == args.role_name]
    if len(matches) == 1:
        return matches[0]
    choices = "\n  ".join(
        f"id={v.id} type={v.type_id} role_name={v.attributes.get('role_name', '<none>')}"
        for v in vehicles
    ) or "<none>"
    raise RuntimeError(
        f"Expected exactly one vehicle with role_name={args.role_name!r}; found {len(matches)}."
        f"\nUse --vehicle-id to select a vehicle. Available vehicles:\n  {choices}"
    )


def vehicle_speed(vehicle: carla.Vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


class Recorder:
    def __init__(self, vehicle: carla.Vehicle, out_dir: Path, csv_file) -> None:
        self.vehicle = vehicle
        self.out_dir = out_dir
        self.writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        self.csv_file = csv_file
        self.lock = threading.Lock()
        self.count = 0

    def on_image(self, image: carla.Image) -> None:
        """Save one camera frame and the current manual-control state."""
        try:
            control = self.vehicle.get_control()
            speed = vehicle_speed(self.vehicle)
            filename = f"{image.frame:08d}.png"
            image.save_to_disk(str(self.out_dir / "images" / filename))
            transform = self.vehicle.get_transform()
            with self.lock:
                self.writer.writerow({
                    "frame": image.frame,
                    "image": filename,
                    "timestamp": f"{image.timestamp:.6f}",
                    "speed_kmh": f"{speed * 3.6:.6f}",
                    "steering": f"{control.steer:.6f}",
                    "throttle": f"{control.throttle:.6f}",
                    "brake": f"{control.brake:.6f}",
                    "x": f"{transform.location.x:.6f}",
                    "y": f"{transform.location.y:.6f}",
                    "z": f"{transform.location.z:.6f}",
                    "yaw": f"{transform.rotation.yaw:.6f}",
                    "collision": 0,
                    "lane_invasion": 0,
                })
                self.csv_file.flush()
                self.count += 1
                if self.count % 100 == 0:
                    print(f"Recorded {self.count} frames", flush=True)
        except Exception as exc:
            print(f"Warning: could not record frame {image.frame}: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    args = arguments()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive.")
    if args.output.exists():
        raise FileExistsError(f"Output folder exists: {args.output}. Use a new run name to avoid overwriting data.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = select_vehicle(world, args)
    print(f"Using existing vehicle id={vehicle.id} ({vehicle.type_id}); no vehicle will be spawned.")

    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(args.width))
    blueprint.set_attribute("image_size_y", str(args.height))
    blueprint.set_attribute("fov", str(args.fov))
    blueprint.set_attribute("sensor_tick", "0.0")

    args.output.mkdir(parents=True)
    (args.output / "images").mkdir()
    sensor = None
    try:
        with (args.output / "telemetry.csv").open("w", newline="", encoding="utf-8") as csv_file:
            recorder = Recorder(vehicle, args.output, csv_file)
            recorder.writer.writeheader()
            csv_file.flush()
            sensor = world.spawn_actor(
                blueprint,
                carla.Transform(carla.Location(x=1.5, z=1.7)),
                attach_to=vehicle,
            )
            sensor.listen(recorder.on_image)
            print(f"Recording to {args.output}")
            print("Drive in manual_control.py; press Ctrl+C in this terminal when finished.")
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                print("\nStopping recorder...")
            finally:
                sensor.stop()
            print(f"Saved {recorder.count} labeled frames.")
    finally:
        if sensor is not None and sensor.is_alive:
            sensor.destroy()
            print("Camera sensor removed. The manually controlled vehicle was not changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
