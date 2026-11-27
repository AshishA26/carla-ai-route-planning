"""Drive one committed route in CARLA and record the chase camera plus a per-tick log.

Heat control: the world runs in synchronous mode, so CARLA renders only when we tick it.
--max-rate caps ticks per wall-clock second, and a watchdog pauses ticking when the GPU
passes --gpu-hot degrees C until it cools to --gpu-cool.
"""
import json
import math
import os
import queue
import random
import subprocess
import sys
import time

import carla
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from importlib.metadata import version as _pkg_version
CARLA_09 = _pkg_version("carla").startswith("0.9")
sys.path.insert(0, os.path.join(HERE, "vendor_0915" if CARLA_09 else "vendor"))
from agents.navigation.basic_agent import BasicAgent  # noqa: E402

EGO_BLUEPRINTS = ["vehicle.tesla.model3", "vehicle.dodge.charger_2020", "vehicle.dodge.charger", "vehicle.lincoln.mkz"]
EGO_COLOR = "220,218,217"
NO_TRAFFIC = ("firetruck", "ambulance", "dodgecop", "police")


# --------------------------------------------------------------------------- heat
class Thermal:
    def __init__(self, hot=82, cool=72, every_s=3.0):
        self.hot, self.cool, self.every, self.last, self.temp = hot, cool, every_s, 0.0, None
        self.paused_s = 0.0

    def read(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=3).stdout
            return int(out.split()[0])
        except Exception:
            return None

    def check(self):
        now = time.time()
        if now - self.last < self.every:
            return
        self.last = now
        self.temp = self.read()
        if self.temp is not None and self.temp >= self.hot:
            print(f"  GPU at {self.temp} C, pausing until {self.cool} C", flush=True)
            t0 = time.time()
            while True:
                time.sleep(5)
                self.temp = self.read()
                if self.temp is None or self.temp <= self.cool:
                    break
            self.paused_s += time.time() - t0
            print(f"  resumed at {self.temp} C", flush=True)


# --------------------------------------------------------------------------- chase camera
class Chase:
    """Third-person camera behind the car; yaw follows the car with a short critically damped lag."""

    def __init__(self, ego_tf, back=7.5, up=3.0, pitch=-10.0, lag_s=0.35):
        self.back, self.up, self.pitch, self.lag = back, up, pitch, lag_s
        self.yaw = ego_tf.rotation.yaw
        self.loc = ego_tf.location

    def update(self, ego_tf, dt):
        err = (ego_tf.rotation.yaw - self.yaw + 180.0) % 360.0 - 180.0
        self.yaw += err * (1.0 - math.exp(-dt / self.lag))
        self.loc = ego_tf.location
        return self.transform()

    def transform(self):
        r = math.radians(self.yaw)
        loc = carla.Location(x=self.loc.x - self.back * math.cos(r), y=self.loc.y - self.back * math.sin(r),
                             z=self.loc.z + self.up)
        return carla.Transform(loc, carla.Rotation(pitch=self.pitch, yaw=self.yaw))


# --------------------------------------------------------------------------- helpers
def connect(host, port, map_name):
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    if map_name and map_name not in world.get_map().name:
        print(f"loading {map_name}", flush=True)
        world = client.load_world(map_name)
    return client, world


def pick_blueprint(lib, names):
    for n in names:
        found = lib.filter(n)
        if found:
            return found[0]
    return [b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4][0]


def spawn_traffic(client, world, ego_loc, n, tm_port, seed):
    rng = random.Random(seed)
    lib = world.get_blueprint_library()
    bps = [b for b in lib.filter("vehicle.*") if b.has_attribute("number_of_wheels")
           and int(b.get_attribute("number_of_wheels")) == 4 and not any(k in b.id for k in NO_TRAFFIC)]
    spots = [s for s in world.get_map().get_spawn_points() if s.location.distance(ego_loc) > 25]
    rng.shuffle(spots)
    batch = []
    for sp in spots[:n]:
        bp = rng.choice(bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(carla.command.SpawnActor(bp, sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))
    ids = []
    for k in range(0, len(batch), 4):  # small batches: each new model's textures stream in before the next
        ids += [r.actor_id for r in client.apply_batch_sync(batch[k:k + 4], True) if not r.error]
        for _ in range(8):
            world.tick()
    return ids


def lead_vehicle(ego, world_map, vehicles, max_dist=60.0):
    """Nearest vehicle ahead in the ego's lane: returns (gap_m, speed_mps) or (None, None)."""
    tf = ego.get_transform()
    fwd = tf.get_forward_vector()
    ego_wp = world_map.get_waypoint(tf.location)
    best = (None, None)
    for v in vehicles:
        if v.id == ego.id:
            continue
        loc = v.get_location()
        d = loc - tf.location
        ahead = d.x * fwd.x + d.y * fwd.y
        if ahead <= 0 or ahead > max_dist:
            continue
        lateral = abs(-d.x * fwd.y + d.y * fwd.x)
        if lateral > 2.2:
            continue
        wp = world_map.get_waypoint(loc)
        if not ego_wp.is_junction and wp and (wp.road_id, wp.lane_id) != (ego_wp.road_id, ego_wp.lane_id) and not wp.is_junction:
            continue
        gap = ahead - ego.bounding_box.extent.x - v.bounding_box.extent.x
        if best[0] is None or gap < best[0]:
            vel = v.get_velocity()
            best = (max(gap, 0.0), math.hypot(vel.x, vel.y))
    return best


def cam_matrix(tf):
    """World-from-camera 4x4 for a carla.Transform (UE axes: x fwd, y right, z up)."""
    return np.array(tf.get_matrix(), dtype=np.float64)


# --------------------------------------------------------------------------- main run
def drive(world_map_name, route, outdir, *, target_kmh=30, driver=None, host="127.0.0.1", port=2200,
          tm_port=8200, fps=25, width=1920, height=1080, fov=80, traffic=30, seed=7,
          preroll_s=4.0, tail_s=2.5, max_rate=12.0, gpu_hot=82, gpu_cool=72, timeout_s=None,
          weather_sun=(150.0, 82.0)):
    """route: {"start_xy", "dest_xy", "preference", "closed_roads"}; re-planned on the live map, which is
    the same OpenDRIVE file the offline planner used, so the route is identical."""
    os.makedirs(os.path.join(outdir, "frames"), exist_ok=True)
    driver = driver or {}
    client, world = connect(host, port, world_map_name)
    wmap = world.get_map()
    original = world.get_settings()
    tm = client.get_trafficmanager(tm_port)
    actors, sensors = [], []
    thermal = Thermal(gpu_hot, gpu_cool)
    log, events = [], []
    try:
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, 1.0 / fps
        world.apply_settings(s)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)
        world.set_weather(carla.WeatherParameters(
            cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=10.0,
            sun_azimuth_angle=weather_sun[0], sun_altitude_angle=weather_sun[1], fog_density=0.0, wetness=0.0))
        world.tick()

        from world_map import Planner
        sx, sy = route["start_xy"]
        dx, dy = route.get("dest_xy", route["start_xy"])
        hold = route.get("hold", False)
        if hold:  # the car never moves: a short stretch of lane is enough to place it
            from agents.navigation.local_planner import RoadOption
            wp0 = wmap.get_waypoint(carla.Location(x=sx, y=sy))
            live_trace = [(wp0, RoadOption.LANEFOLLOW)] + [(w, RoadOption.LANEFOLLOW) for w in wp0.next_until_lane_end(2.0)[:20]]
        else:
            live_trace, _ = Planner(wmap).plan(
                wmap.get_waypoint(carla.Location(x=sx, y=sy)).transform.location,
                wmap.get_waypoint(carla.Location(x=dx, y=dy)).transform.location,
                route.get("preference", "balanced"), closed_roads=route.get("closed_roads", ()))
        if not live_trace:
            raise RuntimeError("route is not reachable on the live map")
        start_tf = live_trace[0][0].transform
        start_tf.location.z += 0.4

        lib = world.get_blueprint_library()
        ego_bp = pick_blueprint(lib, EGO_BLUEPRINTS)
        ego_bp.set_attribute("role_name", "hero")
        if ego_bp.has_attribute("color"):
            ego_bp.set_attribute("color", EGO_COLOR)
        ego = world.try_spawn_actor(ego_bp, start_tf)
        if ego is None:  # something is parked there: nudge forward along the route
            for wp, _ in live_trace[3:20]:
                tf = wp.transform
                tf.location.z += 0.4
                ego = world.try_spawn_actor(ego_bp, tf)
                if ego:
                    break
        if ego is None:
            raise RuntimeError("could not spawn the ego vehicle at the route start")
        actors.append(ego)
        world.tick()
        traffic_ids = spawn_traffic(client, world, ego.get_location(), traffic, tm_port, seed)
        print(f"ego {ego.type_id}, {len(traffic_ids)} traffic vehicles", flush=True)

        # sensors
        cam_bp = lib.find("sensor.camera.rgb")
        for k, v in {"image_size_x": width, "image_size_y": height, "fov": fov, "sensor_tick": 0.0}.items():
            cam_bp.set_attribute(k, str(v))
        chase = Chase(ego.get_transform())
        cam = world.spawn_actor(cam_bp, chase.transform())  # not attached: moved by hand every tick
        sensors.append(cam)
        frames = queue.Queue()
        cam.listen(frames.put)

        col = world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        sensors.append(col)
        col.listen(lambda e: events.append({"kind": "collision", "frame": e.frame,
                                            "other": e.other_actor.type_id if e.other_actor else "static"}))
        lane = world.spawn_actor(lib.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
        sensors.append(lane)
        lane.listen(lambda e: events.append({"kind": "lane_invasion", "frame": e.frame,
                                             "markings": sorted({str(m.type) for m in e.crossed_lane_markings})}))

        agent = BasicAgent(ego, target_speed=driver.get("target_speed_kmh", target_kmh), opt_dict={
            "ignore_traffic_lights": driver.get("ignore_traffic_lights", False),
            "base_vehicle_threshold": driver.get("follow_threshold_m", 5.0),
            "max_brake": driver.get("max_brake", 0.6),
        }, map_inst=wmap)
        agent.follow_speed_limits(driver.get("follow_speed_limits", False))
        agent.set_global_plan(live_trace, stop_waypoint_creation=True, clean_queue=True)

        K = intrinsics(width, height, fov)
        dt = 1.0 / fps
        preroll_n, tail_n = int(preroll_s * fps), int(tail_s * fps)
        limit_n = int((timeout_s or 240) * fps)
        prev_speed, prev_junction, last_light = 0.0, False, None
        done_at, i = None, 0
        wall_t0 = time.time()
        while True:
            tick_start = time.time()
            if i < preroll_n or hold:
                ctrl = carla.VehicleControl(brake=1.0, hand_brake=True)
            else:
                ctrl = agent.run_step()
            ego.apply_control(ctrl)
            cam_pose = chase.update(ego.get_transform(), dt)
            cam.set_transform(cam_pose)
            frame_id = world.tick()
            img = frames.get(timeout=30.0)
            while img.frame < frame_id:
                img = frames.get(timeout=30.0)

            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]
            cv2.imwrite(os.path.join(outdir, "frames", f"{i:05d}.jpg"), arr, [cv2.IMWRITE_JPEG_QUALITY, 93])

            # telemetry
            tf, vel = ego.get_transform(), ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)
            wp = wmap.get_waypoint(tf.location)
            vehicles = world.get_actors().filter("vehicle.*")
            gap, lead_v = lead_vehicle(ego, wmap, vehicles)
            light = ego.get_traffic_light()
            light_state = str(ego.get_traffic_light_state()).split(".")[-1] if light else None
            if light:
                last_light = light_state
            entering = wp.is_junction and not prev_junction
            if entering and last_light == "Red" and i >= preroll_n:
                events.append({"kind": "red_light", "frame": frame_id})
            if not wp.is_junction and prev_junction:
                last_light = None
            prev_junction = wp.is_junction
            log.append({
                "i": i, "frame": frame_id, "t": round(i * dt, 3), "x": tf.location.x, "y": tf.location.y,
                "z": tf.location.z, "yaw": tf.rotation.yaw, "speed": speed,
                "accel": (speed - prev_speed) / dt, "limit_kmh": ego.get_speed_limit(),
                "light": light_state, "junction": wp.is_junction, "road": wp.road_id, "lane": wp.lane_id,
                "lead_gap": gap, "lead_speed": lead_v, "throttle": ctrl.throttle, "brake": ctrl.brake,
                "steer": ctrl.steer, "cam": cam_matrix(cam_pose).tolist(),
            })
            prev_speed = speed

            if not hold and i >= preroll_n and done_at is None and agent.done():
                done_at = i
            i += 1
            if hold and i >= preroll_n + tail_n:
                break
            if (done_at is not None and i - done_at >= tail_n) or i >= limit_n:
                break
            if i % fps == 0:
                print(f"  t={i * dt:5.1f}s  {speed * 3.6:4.1f} km/h  GPU {thermal.temp} C", flush=True)

            thermal.check()
            spare = 1.0 / max_rate - (time.time() - tick_start)
            if spare > 0:
                time.sleep(spare)

        frame_to_i = {r["frame"]: r["i"] for r in log}
        for e in events:
            e["i"] = frame_to_i.get(e["frame"], e.get("i"))
            e["t"] = round((e["i"] or 0) * dt, 3)
        meta = {"fps": fps, "width": width, "height": height, "fov": fov, "K": K.tolist(),
                "ego": ego.type_id, "traffic": len(traffic_ids), "driver": driver, "target_kmh": target_kmh,
                "preroll_s": preroll_s, "reached_goal": done_at is not None, "frames": i,
                "route_xy": [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
                             for wp, _ in live_trace],
                "wall_s": round(time.time() - wall_t0, 1), "thermal_pause_s": round(thermal.paused_s, 1)}
        with open(os.path.join(outdir, "drive.json"), "w") as f:
            json.dump({"meta": meta, "log": log, "events": events}, f)
        print(f"recorded {i} frames in {meta['wall_s']} s -> {outdir}", flush=True)
        return meta
    finally:
        for snsr in sensors:
            try:
                snsr.stop()
                snsr.destroy()
            except Exception:
                pass
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in
                                list(world.get_actors().filter("vehicle.*"))])
        except Exception:
            pass
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original)
        except Exception:
            pass


def intrinsics(w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
