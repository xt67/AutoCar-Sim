"""Reusable CARLA connection + vehicle/sensor helpers."""
import queue
import random
import time

import carla

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CarlaEnv:
    def __init__(self, host=config.CARLA_HOST, port=config.CARLA_PORT,
                 timeout=config.CARLA_TIMEOUT, sync=True, delta=config.COLLECT_DELTA,
                 no_rendering=False):
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.tm = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        self.sync = sync
        if sync:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = delta
        if no_rendering:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(sync)
        self.map = self.world.get_map()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.image_queue = queue.Queue()
        self.collision_queue = queue.Queue()
        self.lane_queue = queue.Queue()
        self.depth_camera = None
        self.depth_queue = queue.Queue()
        self.frame = 0

    def spawn_vehicle(self, vehicle_filter="vehicle.tesla.model3", spawn_index=None):
        bps = self.blueprint_lib.filter(vehicle_filter)
        if not bps:
            bps = self.blueprint_lib.filter("vehicle.*")
        bp = bps[0]
        bp.set_attribute("role_name", "ego")
        if bp.has_attribute("color"):
            bp.set_attribute("color", "255,0,0")
        spawn_points = self.map.get_spawn_points()
        if spawn_index is not None and spawn_index < len(spawn_points):
            sp = spawn_points[spawn_index]
        else:
            sp = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.spawn_actor(bp, sp)
        try:
            pc = self.vehicle.get_physics_control()
            pc.use_sweep_wheel_collision = True
            self.vehicle.apply_physics_control(pc)
        except Exception:
            pass
        return self.vehicle

    def attach_camera(self, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT,
                      fov=config.CAMERA_FOV, transform=None):
        bp = self.blueprint_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        if transform is None:
            transform = carla.Transform(carla.Location(x=2.0, z=1.4),
                                         carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.camera.listen(self.image_queue.put)
        return self.camera

    def attach_collision(self):
        bp = self.blueprint_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(),
                                                       attach_to=self.vehicle)
        self.collision_sensor.listen(self.collision_queue.put)

    def attach_lane_invasion(self):
        bp = self.blueprint_lib.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(bp, carla.Transform(),
                                                  attach_to=self.vehicle)
        self.lane_sensor.listen(self.lane_queue.put)

    def attach_depth_camera(self, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT,
                            fov=config.CAMERA_FOV, transform=None):
        bp = self.blueprint_lib.find("sensor.camera.depth")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        if transform is None:
            transform = carla.Transform(carla.Location(x=2.0, z=1.4),
                                         carla.Rotation(pitch=-15))
        self.depth_camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        self.depth_queue = __import__("queue").Queue()
        self.depth_camera.listen(self.depth_queue.put)
        return self.depth_camera

    def get_depth_image(self, timeout=5.0):
        try:
            return self.depth_queue.get(timeout=timeout)
        except Exception:
            return None

    def tick(self):
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        self.frame += 1

    def get_image(self, timeout=5.0):
        try:
            return self.image_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, q):
        items = []
        try:
            while True:
                items.append(q.get_nowait())
        except queue.Empty:
            pass
        return items

    def cleanup(self):
        for a in (self.camera, self.depth_camera, self.collision_sensor, self.lane_sensor, self.vehicle):
            if a is not None:
                try:
                    a.destroy()
                except Exception:
                    pass
        self.camera = None
        self.depth_camera = None
        self.collision_sensor = None
        self.lane_sensor = None
        self.vehicle = None
        if self.sync:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.tm.set_synchronous_mode(False)
            except Exception:
                pass

    def spawn_traffic(self, n=20, filter="vehicle.*"):
        bps = self.blueprint_lib.filter(filter)
        sps = self.map.get_spawn_points()
        random.shuffle(sps)
        count = 0
        for sp in sps:
            if count >= n:
                break
            bp = random.choice(bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(
                    bp.get_attribute("color").recommended_values))
            actor = self.world.try_spawn_actor(bp, sp)
            if actor is not None:
                actor.set_autopilot(True, self.tm.get_port())
                count += 1
        return count
