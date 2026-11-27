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

# a small car drives; the Kia Soul is drawn over it afterwards (kia.py), so it must fit inside the Kia
EGO_BLUEPRINTS = ["vehicle.nissan.micra", "vehicle.mini.cooper_s", "vehicle.tesla.model3"]
EGO_COLOR = "220,218,217"
NO_TRAFFIC = ("firetruck", "ambulance", "dodgecop", "police", "cybertruck", "carlacola", "microlino", "isetta")


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


def spawn_traffic(client, world, ego_loc, n, tm_port, seed, keep_clear=(), clear_m=60.0):
    rng = random.Random(seed)
    lib = world.get_blueprint_library()
    bps = [b for b in lib.filter("vehicle.*") if b.has_attribute("number_of_wheels")
           and int(b.get_attribute("number_of_wheels")) == 4 and not any(k in b.id for k in NO_TRAFFIC)
           and (not b.has_attribute("base_type") or b.get_attribute("base_type").as_str() == "car")]
    clear = [ego_loc] + list(keep_clear)
    spots = [s for s in world.get_map().get_spawn_points()
             if all(s.location.distance(c) > clear_m for c in clear)]
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
        # geometric corridor along the car's heading, not lane ids: a car we are drifting into counts
        if lateral > ego.bounding_box.extent.y + v.bounding_box.extent.y + 0.3:
            continue
        gap = ahead - ego.bounding_box.extent.x - v.bounding_box.extent.x
        if best[0] is None or gap < best[0]:
            vel = v.get_velocity()
            best = (max(gap, 0.0), math.hypot(vel.x, vel.y))
    return best


def blinker_state(agent, director):
    """Which indicator is on: the director's (stalled car), else the next manoeuvre on the plan
    within ~25 m (turns at junctions, lane changes)."""
    from agents.navigation.local_planner import RoadOption
    if director is not None and getattr(director, "blinker", None):
        return director.blinker
    lp = agent.get_local_planner()
    try:
        _, nxt = lp.get_incoming_waypoint_and_direction(steps=12)
    except Exception:
        nxt = None
    for opt in (getattr(lp, "target_road_option", None), nxt):
        if opt in (RoadOption.LEFT, RoadOption.CHANGELANELEFT):
            return "left"
        if opt in (RoadOption.RIGHT, RoadOption.CHANGELANERIGHT):
            return "right"
    return None


def cam_matrix(tf):
    """World-from-camera 4x4 for a carla.Transform (UE axes: x fwd, y right, z up)."""
    return np.array(tf.get_matrix(), dtype=np.float64)


# --------------------------------------------------------------------------- main run
def drive(world_map_name, route, outdir, *, target_kmh=30, driver=None, host="127.0.0.1", port=2200,
          tm_port=8200, fps=25, width=1920, height=1080, fov=80, traffic=30, seed=7,
          preroll_s=4.0, tail_s=2.5, max_rate=12.0, gpu_hot=82, gpu_cool=72, timeout_s=None,
          weather_sun=(135.0, 45.0), model_turn=None, warmup_s=6.0):
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
        s.substepping, s.max_substep_delta_time, s.max_substeps = True, 0.01, 10  # physics at 100 Hz
        world.apply_settings(s)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)
        world.set_weather(carla.WeatherParameters(
            cloudiness=30.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=10.0,
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
        hz = route.get("hazards") or {}
        keep = [carla.Location(x=hz[k][0], y=hz[k][1]) for k in ("stall_xy", "closure_xy") if k in hz]
        traffic_ids = spawn_traffic(client, world, ego.get_location(), traffic, tm_port, seed, keep_clear=keep,
                                    clear_m=60.0 if not hold else 30.0)
        print(f"ego {ego.type_id}, {len(traffic_ids)} traffic vehicles", flush=True)

        # sensors
        cam_bp = lib.find("sensor.camera.rgb")
        cam_attrs = {"image_size_x": width, "image_size_y": height, "fov": fov, "sensor_tick": 0.0,
                     # steady picture: no motion blur, slow auto-exposure (default adapts every frame)
                     "motion_blur_intensity": 0.0, "exposure_speed_up": 0.4, "exposure_speed_down": 0.4}
        for k, v in cam_attrs.items():
            if cam_bp.has_attribute(k):
                cam_bp.set_attribute(k, str(v))
        # rigidly mounted chase camera: the car stays fixed in frame, and Unreal's temporal
        # anti-aliasing sees continuous motion instead of a camera teleported every tick
        cam_tf = carla.Transform(carla.Location(x=-7.5, z=3.0), carla.Rotation(pitch=-10.0))
        cam = world.spawn_actor(cam_bp, cam_tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
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
            # softer, correctly timed steering: the default gains assume 20 Hz and snap at lane changes
            "dt": 1.0 / fps,
            "lateral_control_dict": {"K_P": 1.2, "K_I": 0.02, "K_D": 0.12, "dt": 1.0 / fps},
        }, map_inst=wmap)
        agent.follow_speed_limits(driver.get("follow_speed_limits", False))
        agent.set_global_plan(live_trace, stop_waypoint_creation=True, clean_queue=True)

        from director import Reroute, StopSigns, trace_xyz
        planner = Planner(wmap)
        plans = [{"t": 0.0, "reason": "initial", "by": "model", "route_xy": trace_xyz(live_trace)}]
        timeline = []
        stops = StopSigns(ego, events, obey=driver.get("stop_signs", True))
        director = None
        if route.get("hazards"):
            director = Reroute(world, wmap, ego, agent, planner, route["hazards"],
                               wmap.get_waypoint(carla.Location(x=dx, y=dy)).transform.location,
                               route.get("preference", "balanced"), events, plans, timeline,
                               lambda loc, d: model_turn(wmap, planner, loc, d, ego))
            director.setup()
            world.tick()

        # warm-up: Unreal streams high-resolution textures and levels of detail in after a camera appears,
        # and auto-exposure settles; tick (car held) and throw these frames away before recording
        for _ in range(int(warmup_s * fps)):
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            world.tick()
        while not frames.empty():
            frames.get_nowait()

        K = intrinsics(width, height, fov)
        dt = 1.0 / fps
        preroll_n, tail_n = int(preroll_s * fps), int(tail_s * fps)
        limit_n = int((timeout_s or 240) * fps)
        prev_speed, prev_junction, last_light = 0.0, False, None
        done_at, i, stuck_n = None, 0, 0
        prev_steer, wheel_angle = 0.0, 0.0
        wall_t0 = time.time()
        while True:
            tick_start = time.time()
            t_now = i * (1.0 / fps)
            stop_d = None
            if i < preroll_n or hold:
                ctrl = carla.VehicleControl(brake=1.0, hand_brake=True)
            else:
                if director:
                    v = ego.get_velocity()
                    director.step(t_now, i, math.hypot(v.x, v.y), plans[-1]["route_xy"])
                ctrl = agent.run_step()
                v = ego.get_velocity()
                ctrl, stop_d = stops.step(ctrl, wmap.get_waypoint(ego.get_location()), math.hypot(v.x, v.y), t_now, i)
            # steering slew limit: no instant swings (the PID otherwise snaps at lane changes)
            max_step = 1.1 / fps
            ctrl.steer = float(prev_steer + max(-max_step, min(max_step, ctrl.steer - prev_steer)))
            prev_steer = ctrl.steer
            ego.apply_control(ctrl)
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
                "steer": ctrl.steer, "cam": cam_matrix(cam.get_transform()).tolist(),
                "plan": len(plans) - 1, "stop_d": stop_d,
                "pitch": tf.rotation.pitch, "roll": tf.rotation.roll, "wheel": wheel_angle,
                "blinker": blinker_state(agent, director) if (i >= preroll_n and not hold) else None,
            })
            prev_speed = speed
            wheel_angle += speed * dt / 0.31235  # Kia wheel radius, for the drawn wheels

            if not hold and i >= preroll_n:
                explained = (light_state == "Red") or (last_light == "Red") or (stop_d is not None and stop_d < 10) or \
                    (director is not None and director.pending is not None)
                stuck_n = stuck_n + 1 if (speed < 0.1 and not explained and done_at is None) else 0
                if stuck_n > 15 * fps:
                    print(f"  STUCK at ({tf.location.x:.1f},{tf.location.y:.1f}) for 15 s, lead gap {gap}; stopping",
                          flush=True)
                    break
            if not hold and i >= preroll_n and done_at is None and agent.done():
                done_at = i
            i += 1
            if hold and i >= preroll_n + tail_n:
                break
            if (done_at is not None and i - done_at >= tail_n) or i >= limit_n:
                break
            if i % (2 * fps) == 0:
                print(f"  t={i * dt:5.1f}s  ({tf.location.x:6.1f},{tf.location.y:6.1f}) road {wp.road_id:3d} "
                      f"{speed * 3.6:4.1f} km/h  plan {len(plans) - 1}  stop_d {stop_d}  GPU {thermal.temp} C", flush=True)

            thermal.check()
            spare = 1.0 / max_rate - (time.time() - tick_start)
            if spare > 0:
                time.sleep(spare)

        frame_to_i = {r["frame"]: r["i"] for r in log}
        for e in events:
            e["i"] = frame_to_i.get(e.get("frame"), e.get("i"))
            e["t"] = round((e["i"] or 0) * dt, 3)
        meta = {"fps": fps, "width": width, "height": height, "fov": fov, "K": K.tolist(),
                "ego": ego.type_id, "traffic": len(traffic_ids), "driver": driver, "target_kmh": target_kmh,
                "preroll_s": preroll_s, "reached_goal": done_at is not None, "frames": i,
                "route_xy": plans[0]["route_xy"], "plans": plans, "timeline": timeline,
                "hazards": route.get("hazards"), "kia": True, "sun": list(weather_sun),
                "ego_extent": [ego.bounding_box.extent.x, ego.bounding_box.extent.y, ego.bounding_box.extent.z],
                "wall_s": round(time.time() - wall_t0, 1), "thermal_pause_s": round(thermal.paused_s, 1)}
        with open(os.path.join(outdir, "drive.json"), "w") as f:
            json.dump({"meta": meta, "log": log, "events": events}, f)
        print(f"recorded {i} frames in {meta['wall_s']} s -> {outdir}", flush=True)
        return meta
    finally:
        if log and not os.path.exists(os.path.join(outdir, "drive.json")):
            with open(os.path.join(outdir, "drive_partial.json"), "w") as f:
                json.dump({"log": [{k: v for k, v in r.items() if k != "cam"} for r in log], "events": events,
                           "plans": [{k: v for k, v in p.items() if k != "route_xy"} for p in plans]
                           if "plans" in locals() else [], "timeline": timeline if "timeline" in locals() else []}, f)
        if "director" in locals() and director:
            director.destroy()
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
