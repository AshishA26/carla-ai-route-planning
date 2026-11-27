"""Live mode: drive CARLA from a prompt, in real time.

    python live.py                    # starts CARLA with its window, opens the control window
    python live.py --start east_side  # where the car starts
    python live.py --no-launch        # use a CARLA that is already running on port 2200

Two windows:
  * CARLA's own window: a chase view of the car. The planned path is drawn into the 3D world (blue line),
    with arrows at the next turns and a marker at the destination.
  * The control window (this script): type a request and press Enter. The model (OpenAI if
    OPENAI_API_KEY is set in .env, otherwise a small offline parser) calls the same tools, the same
    deterministic validator checks the plan, and the car drives it. Speed, status and a map are on the right.

Keys in the control window: Enter = send, Esc = quit, F1 = toggle the chase camera, F2 = stop the car.
This file does not change the record-then-composite pipeline (make_demo.py / run.py).
"""
import argparse
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import envfile  # noqa: E402,F401  (loads .env)
import carla  # noqa: E402
import pygame  # noqa: E402

from agent import ROUTE_SYSTEM, run_turn  # noqa: E402
from director import Safety, StopSigns, smooth_lane_changes, trace_xyz, trim_to  # noqa: E402
from scenes import CITY_LIMIT_KMH, PLACES  # noqa: E402
from sim import EGO_COLOR, blinker_state, spawn_traffic  # noqa: E402
from tools import TOOL_SPECS, RouteSession, guardrail  # noqa: E402
from world_map import Planner  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "vendor_0915"))
from agents.navigation.basic_agent import BasicAgent  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402

PORT, TM_PORT = 2200, 8200
MAP_NAME = "Town10HD_Opt"
FONTS = os.path.join(HERE, "fonts")

# palette (same as the video overlay)
BG = (14, 16, 19)
PANEL = (22, 25, 29)
RULE = (44, 48, 54)
TEXT = (237, 237, 234)
DIM = (139, 144, 150)
FAINT = (88, 93, 99)
BLUE = (47, 107, 255)
PASS = (63, 184, 127)
WARN = (232, 161, 58)
BAD = (229, 72, 77)

EXAMPLES = ["Take me to the east side, no rush", "Get me to the north loop fast", "Go to the south boulevard",
            "Run the red lights to the west side"]


# --------------------------------------------------------------------------- offline request parser
ALIASES = {
    "west_side": ["west side", "west"], "southeast": ["southeast", "south east", "south-east"],
    "south_blvd": ["south boulevard", "boulevard", "south blvd"], "north_loop": ["north loop", "north"],
    "east_side": ["east side", "east"], "centre": ["central avenue", "centre", "center", "downtown", "museum", "avenue"],
}


def rules_script(text, at_key=None):
    """Without an API key: turn plain requests into the same tool calls a model would make."""
    t = text.lower()
    g = guardrail(text)
    if g:
        return [("report_unsafe_request", {"reason": f"Rider asked for {g['label']}."}),
                ("say", f"I won't do that ({g['label']}). Ask me for a destination and I'll take the legal route.")]
    dest = None
    for key in sorted(ALIASES, key=lambda k: -max(len(a) for a in ALIASES[k])):
        if key != at_key and any(re.search(r"\b" + re.escape(a) + r"\b", t) for a in ALIASES[key]):
            dest = key
            break
    if dest is None:
        names = ", ".join(PLACES[k]["name"] for k in PLACES)
        return [("get_vehicle_state", {}), ("say", f"Where to? I know: {names}.")]
    pref = "balanced"
    if re.search(r"\b(fast|quick|hurry|late|asap|rush(?!.*no))\b", t) and "no rush" not in t:
        pref = "fastest"
    if re.search(r"\b(no rush|safe|calm|careful|sketchy|relax|easy)\b", t):
        pref = "safest"
    cap = None
    m = re.search(r"\b(?:go|at|do|like)\s*(\d{2,3})\b", t)
    if m:
        cap = float(m.group(1))
    prefs = ["fastest", "safest"] if pref != "balanced" else ["balanced"]
    script = [("get_vehicle_state", {}), ("plan_routes", {"destination": dest, "preferences": prefs}),
              ("commit_route", {"route_id": pref, "max_speed_kmh": cap, "reason": f"Rider asked for {pref}."})]
    if cap is not None and cap > CITY_LIMIT_KMH:  # the validator will refuse this; retry legally
        script.append(("commit_route", {"route_id": pref, "max_speed_kmh": CITY_LIMIT_KMH,
                                        "reason": "Validator refused the speed; retrying at the posted limit."}))
    say = {"fastest": "Taking the fastest legal route", "safest": "Taking the calmer route",
           "balanced": "On my way"}[pref]
    script.append(("say", f"{say} to {PLACES[dest]['name']}."))
    return script


# --------------------------------------------------------------------------- CARLA side
class LiveCar:
    def __init__(self, start_key, traffic, fps):
        self.fps = fps
        self.client = carla.Client("127.0.0.1", PORT)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        if MAP_NAME not in self.world.get_map().name:
            self.world = self.client.load_world(MAP_NAME)
        self.map = self.world.get_map()
        self.original = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, 1.0 / fps
        s.substepping, s.max_substep_delta_time, s.max_substeps = True, 0.01, 10
        self.world.apply_settings(s)
        self.tm = self.client.get_trafficmanager(TM_PORT)
        self.tm.set_synchronous_mode(True)
        self.world.set_weather(carla.WeatherParameters(cloudiness=30.0, sun_azimuth_angle=250.0,
                                                       sun_altitude_angle=12.0, wind_intensity=10.0))
        self.planner = Planner(self.map)
        x, y = PLACES[start_key]["xy"]
        wp = self.map.get_waypoint(carla.Location(x=x, y=y))
        lib = self.world.get_blueprint_library()
        bp = lib.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute("color", EGO_COLOR)
        self.ego = None
        for w in [wp] + wp.next_until_lane_end(3.0)[:15]:
            tf = w.transform
            tf.location.z += 0.4
            self.ego = self.world.try_spawn_actor(bp, tf)
            if self.ego:
                break
        if self.ego is None:
            raise RuntimeError("could not spawn the car at " + start_key)
        self.world.tick()
        self.traffic = spawn_traffic(self.client, self.world, self.ego.get_location(), traffic, TM_PORT, 7)
        for v in self.world.get_actors(self.traffic):
            self.tm.update_vehicle_lights(v, True)  # traffic switches its lights on at dusk
        self.agent = BasicAgent(self.ego, target_speed=CITY_LIMIT_KMH, opt_dict={
            "dt": 1.0 / fps, "lateral_control_dict": {"K_P": 1.2, "K_I": 0.02, "K_D": 0.12, "dt": 1.0 / fps}},
            map_inst=self.map)
        self.events = []
        self.stops = StopSigns(self.ego, self.events)
        self.safety = Safety(self.world, self.ego, self.map)
        self.route_xy = None
        self.trace = None       # active plan: list of (waypoint, RoadOption)
        self.dest_key = None
        self.driving = False
        self.arrived = False
        self.prev_steer = 0.0
        self.follow = True
        self.chase_yaw = self.ego.get_transform().rotation.yaw
        self.t = 0.0
        self.light = None
        self.stop_d = None

    # --- plans -----------------------------------------------------------------------------
    def apply_plan(self, trace, dest_key, cap_kmh):
        loc = self.ego.get_location()
        ego_wp = self.map.get_waypoint(loc)
        near = min(trace, key=lambda p: p[0].transform.location.distance(loc))[0]
        if near.transform.location.distance(loc) > 5.0 or (near.road_id, near.lane_id) != (ego_wp.road_id, ego_wp.lane_id):
            x, y = PLACES[dest_key]["xy"]
            fresh, _ = self.planner.plan(loc, self.map.get_waypoint(carla.Location(x=x, y=y)).transform.location)
            if fresh:  # the car has moved since the model planned: same request, from here
                trace = fresh
        trace = smooth_lane_changes(trim_to(trace, loc))
        self.trace, self.dest_key = trace, dest_key
        self.route_xy = trace_xyz(trace)
        self.agent.set_target_speed(min(cap_kmh or CITY_LIMIT_KMH, CITY_LIMIT_KMH))
        self.agent.set_global_plan(trace, stop_waypoint_creation=True, clean_queue=True)
        self.driving, self.arrived = True, False

    def stop(self):
        self.driving = False

    def remaining_m(self):
        if not self.trace:
            return None
        loc = self.ego.get_location()
        k = min(range(len(self.trace)), key=lambda j: self.trace[j][0].transform.location.distance(loc))
        pts = [w.transform.location for w, _ in self.trace[k:]]
        return sum(a.distance(b) for a, b in zip(pts, pts[1:]))

    # --- per frame -------------------------------------------------------------------------
    def step(self):
        v = self.ego.get_velocity()
        speed = math.hypot(v.x, v.y)
        if self.driving and not self.agent.done():
            ctrl = self.agent.run_step()
            ctrl, self.stop_d = self.stops.step(ctrl, self.map.get_waypoint(self.ego.get_location()), speed, self.t, 0)
            ctrl = self.safety.step(ctrl, self.route_xy, speed, self.t)
        else:
            if self.driving and self.agent.done():
                self.driving, self.arrived = False, True
            ctrl = carla.VehicleControl(brake=1.0, hand_brake=speed < 0.3)
            self.stop_d = None
            self.safety.why = None
        step = 1.1 / self.fps
        ctrl.steer = float(self.prev_steer + max(-step, min(step, ctrl.steer - self.prev_steer)))
        self.prev_steer = ctrl.steer
        self.ego.apply_control(ctrl)
        self.set_lights(ctrl)
        self.world.tick()
        self.t += 1.0 / self.fps
        light = self.ego.get_traffic_light()
        self.light = str(self.ego.get_traffic_light_state()).split(".")[-1] if light else None
        if self.follow:
            self.chase()
        self.draw_path()
        return speed

    def set_lights(self, ctrl):
        ls = carla.VehicleLightState
        state = ls.Position | ls.LowBeam
        if ctrl.brake > 0.1:
            state |= ls.Brake
        side = blinker_state(self.agent, None) if self.driving else None
        if side == "left":
            state |= ls.LeftBlinker
        elif side == "right":
            state |= ls.RightBlinker
        self.ego.set_light_state(ls(state))

    def chase(self):
        tf = self.ego.get_transform()
        err = (tf.rotation.yaw - self.chase_yaw + 180.0) % 360.0 - 180.0
        self.chase_yaw += err * (1.0 - math.exp(-(1.0 / self.fps) / 0.35))
        r = math.radians(self.chase_yaw)
        loc = carla.Location(tf.location.x - 8.0 * math.cos(r), tf.location.y - 8.0 * math.sin(r), tf.location.z + 3.4)
        self.world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-13.0, yaw=self.chase_yaw)))

    def draw_path(self):
        """Path, turn arrows and the destination, drawn into the 3D world (visible in CARLA's window)."""
        if not self.trace or not self.driving:
            return
        dbg = self.world.debug
        life = 2.5 / self.fps
        loc = self.ego.get_location()
        k = min(range(len(self.trace)), key=lambda j: self.trace[j][0].transform.location.distance(loc))
        ahead = self.trace[k + 2:k + 60]
        # debug lines are drawn self-lit and the dusk exposure blooms them: keep the colour dark and the line thin
        col = carla.Color(12, 40, 120)
        pts = [w.transform.location + carla.Location(z=0.08) for w, _ in ahead]
        for a, b in zip(pts, pts[1:]):
            dbg.draw_line(a, b, thickness=0.06, color=col, life_time=life)
        # arrows where the plan turns (RoadOption LEFT/RIGHT), along the exit direction
        last = None
        for j, (w, opt) in enumerate(ahead):
            if opt in (RoadOption.LEFT, RoadOption.RIGHT) and opt != last and j + 6 < len(ahead):
                a = ahead[j][0].transform.location + carla.Location(z=0.3)
                b = ahead[j + 6][0].transform.location + carla.Location(z=0.3)
                dbg.draw_arrow(a, b, thickness=0.07, arrow_size=0.4, color=carla.Color(90, 90, 95), life_time=life)
            last = opt if opt != RoadOption.LANEFOLLOW else last
        end = self.trace[-1][0].transform.location
        if end.distance(loc) < 150:
            dbg.draw_point(end + carla.Location(z=1.0), size=0.15, color=carla.Color(20, 90, 50), life_time=life)
            dbg.draw_string(end + carla.Location(z=2.0), PLACES[self.dest_key]["name"], draw_shadow=True,
                            color=carla.Color(255, 255, 255), life_time=life)

    def add_test_camera(self):
        """Testing only: a camera at the chase position, to see what CARLA's window shows."""
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in {"image_size_x": 960, "image_size_y": 540, "fov": 90}.items():
            bp.set_attribute(k, str(v))
        self.test_cam = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=-8.0, z=3.4), carla.Rotation(pitch=-13.0)),
                                               attach_to=self.ego)
        self.last_view = None
        self.test_cam.listen(lambda img: setattr(self, "last_view", img))

    def save_view(self, path):
        img = getattr(self, "last_view", None)
        if img is not None:
            import numpy as np
            from PIL import Image
            a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, 2::-1]
            Image.fromarray(a).save(path, quality=90)

    def close(self):
        try:
            if getattr(self, "test_cam", None):
                self.test_cam.stop()
                self.test_cam.destroy()
            self.client.apply_batch([carla.command.DestroyActor(a) for a in [self.ego.id] + list(self.traffic)])
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self.original)
        except Exception:
            pass


# --------------------------------------------------------------------------- UI
class Fonts:
    def __init__(self):
        f = lambda name, size: pygame.font.Font(os.path.join(FONTS, f"{name}.ttf"), size)  # noqa: E731
        self.title = f("Geist-500", 22)
        self.label = f("Geist-500", 12)
        self.body = f("Geist-400", 16)
        self.say = f("Geist-400", 17)
        self.mono = f("GeistMono-400", 14)
        self.mono_s = f("GeistMono-400", 13)
        self.speed = f("Geist-300", 84)
        self.unit = f("Geist-400", 15)
        self.small = f("Geist-400", 14)
        self.status = f("Geist-500", 15)
        self.input = f("Geist-400", 17)
        self.limit = f("Geist-600", 20)


def wrap(font, text, width):
    lines, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if font.size(trial)[0] <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


class App:
    W, H = 1240, 780
    LEFT = 720

    def __init__(self, car, real):
        pygame.init()
        pygame.display.set_caption("Route planning · live")
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.key.set_repeat(400, 35)  # hold Backspace to delete
        self.f = Fonts()
        self.car, self.real = car, real
        self.rows = []          # (kind, text, color)
        self.input = ""
        self.busy = False
        self.results = queue.Queue()
        self.scroll = 0
        self.roads = self._roads()
        self.chip_rects = []
        self.mode = os.environ.get("OPENAI_MODEL", "gpt-5-mini") if real else "offline rules (no API key)"
        self.add("label", "READY")
        self.add("result", "Type where you want to go and press Enter. The validator checks every plan "
                           "before the car moves.")

    def _roads(self):
        from overlay import road_polylines
        lines = road_polylines(self.car.map, step=3.0)
        xs = [p[0] for s in lines for p in s]
        ys = [p[1] for s in lines for p in s]
        return lines, (min(xs), max(xs), min(ys), max(ys))

    def add(self, kind, text, color=None):
        self.rows.append((kind, text, color))
        self.scroll = 0

    # --- requests -------------------------------------------------------------------------
    def send(self, text):
        if self.busy or not text.strip():
            return
        self.busy = True
        self.add("label", "RIDER")
        self.add("body", text)
        loc = self.car.ego.get_location()
        v = self.car.ego.get_velocity()
        at_key = min(PLACES, key=lambda k: math.hypot(PLACES[k]["xy"][0] - loc.x, PLACES[k]["xy"][1] - loc.y))

        def work():
            try:
                sess = RouteSession(self.car.map, self.car.planner, None, text, start_loc=loc,
                                    speed_kmh=math.hypot(v.x, v.y) * 3.6)
                script = None if self.real else rules_script(text, at_key)
                tr = run_turn(sess, TOOL_SPECS, ROUTE_SYSTEM, text, script=script, fake=not self.real)
                self.results.put(("ok", sess, tr))
            except Exception as e:  # network or API errors end up in the panel, not a crash
                self.results.put(("error", None, str(e)))

        threading.Thread(target=work, daemon=True).start()

    def take_result(self):
        try:
            kind, sess, tr = self.results.get_nowait()
        except queue.Empty:
            return
        self.busy = False
        if kind == "error":
            self.add("check", "ERROR", BAD)
            self.add("result", tr)
            return
        self.add("label", "MODEL  ·  " + self.mode)
        for ev in tr[1:]:
            k = ev["kind"]
            if k == "tool_call":
                from overlay import fmt_call
                self.add("mono", fmt_call(ev["name"], ev["args"]))
            elif k == "tool_result":
                r = ev["result"]
                if ev["name"] == "commit_route":
                    if r.get("accepted"):
                        self.add("check", "VALIDATOR  pass", PASS)
                        self.add("result", f"driving to {r['driving']}, about {r['eta_s']} s")
                    else:
                        self.add("check", "VALIDATOR  reject", WARN)
                        for issue in r.get("issues", []):
                            self.add("result", issue)
                elif ev["name"] in ("plan_routes", "get_vehicle_state"):
                    from overlay import fmt_tool_result
                    for line in fmt_tool_result(ev["name"], None, r):
                        self.add("result", line)
                elif ev["name"] == "report_unsafe_request":
                    self.add("check", "DECLINED", BAD)
            elif k == "say":
                self.add("label", "CAR")
                self.add("say", ev["text"])
        g = guardrail(sess.command)
        if g and not sess.refused:
            self.add("check", "GUARDRAIL  match", BAD)
            self.add("result", f"“{g['match']}”: {g['label']}")
        if sess.committed:
            c = sess.committed
            self.car.apply_plan(c["trace"], c["destination"], c["max_speed_kmh"])
        elif sess.refused:
            self.car.stop()

    # --- drawing --------------------------------------------------------------------------
    def text(self, s, font, color, pos):
        self.screen.blit(font.render(s, True, color), pos)

    def draw_left(self):
        sc, f = self.screen, self.f
        pygame.draw.rect(sc, PANEL, (0, 0, self.LEFT, self.H))
        pygame.draw.line(sc, RULE, (self.LEFT, 0), (self.LEFT, self.H))
        self.text("LIVE  ·  CARLA TOWN 10", f.label, BLUE, (32, 26))
        self.text("Route planning", f.title, TEXT, (32, 44))
        chip = f"model: {self.mode}"
        w = f.small.size(chip)[0] + 24
        pygame.draw.rect(sc, (34, 38, 44), (self.LEFT - 32 - w, 40, w, 28), border_radius=14)
        self.text(chip, f.small, DIM, (self.LEFT - 32 - w + 12, 45))
        pygame.draw.line(sc, RULE, (32, 84), (self.LEFT - 32, 84))
        # conversation, newest at the bottom, clipped above the input area
        top, bottom = 96, self.H - 150
        area = pygame.Rect(0, top, self.LEFT, bottom - top)
        sc.set_clip(area)
        layout = []
        width = self.LEFT - 64
        for kind, text, color in self.rows:
            font = {"label": f.label, "body": f.body, "say": f.say, "mono": f.mono, "result": f.mono_s,
                    "check": f.label}[kind]
            lines = wrap(font, text, width - 18) if kind in ("body", "say", "result") else [text]
            lh = {"label": 20, "body": 23, "say": 25, "mono": 21, "result": 19, "check": 20}[kind]
            gap = {"label": 2, "body": 14, "say": 14, "mono": 2, "result": 8, "check": 2}[kind]
            layout.append((kind, lines, color, font, lh, gap))
        total = sum(lh * len(ln) + gap for _, ln, _, _, lh, gap in layout)
        y = bottom - total - 8 + self.scroll if total > bottom - top else top + 8
        for kind, lines, color, font, lh, gap in layout:
            for ln in lines:
                x = 32
                col = color or {"label": DIM, "body": TEXT, "say": TEXT, "mono": TEXT, "result": DIM,
                                "check": TEXT}[kind]
                if kind == "mono":
                    self.text("›", font, BLUE, (x, y))
                    x += 16
                elif kind == "result":
                    x += 16
                elif kind == "check":
                    pygame.draw.circle(sc, col, (x + 4, y + 9), 4)
                    x += 16
                self.text(ln, font, col, (x, y))
                y += lh
            y += gap
        sc.set_clip(None)
        if self.busy:
            dots = "." * (1 + int(time.time() * 3) % 3)
            self.text("thinking" + dots, f.small, DIM, (32, bottom - 2))
        # example chips
        self.chip_rects = []
        x, y = 32, self.H - 136
        for ex in EXAMPLES:
            w = f.small.size(ex)[0] + 22
            if x + w > self.LEFT - 32:
                x, y = 32, y + 34
            r = pygame.Rect(x, y, w, 28)
            pygame.draw.rect(sc, (32, 36, 42), r, border_radius=14)
            self.text(ex, f.small, DIM, (x + 11, y + 5))
            self.chip_rects.append((r, ex))
            x += w + 8
        # input box
        box = pygame.Rect(32, self.H - 64, self.LEFT - 64, 44)
        pygame.draw.rect(sc, (12, 14, 17), box, border_radius=10)
        pygame.draw.rect(sc, BLUE if not self.busy else RULE, box, width=1, border_radius=10)
        shown = self.input or "Where to?  e.g. “Take me to the east side, no rush”"
        col = TEXT if self.input else FAINT
        while f.input.size(shown)[0] > box.width - 40 and len(shown) > 1:
            shown = shown[1:]
        self.text(shown, f.input, col, (box.x + 16, box.y + 11))
        if self.input and int(time.time() * 2) % 2 == 0:
            cx = box.x + 16 + f.input.size(shown)[0] + 2
            pygame.draw.line(sc, TEXT, (cx, box.y + 12), (cx, box.y + 32), 2)
        self.text("Enter", f.label, FAINT, (box.right - 48, box.y + 15))

    def status(self, speed):
        c = self.car
        if self.busy:
            return "Planning", DIM
        if c.arrived:
            return f"Arrived at {PLACES[c.dest_key]['name']}", PASS
        if not c.driving:
            return ("Holding: request declined" if any(r[1] == "DECLINED" for r in self.rows[-6:]) else
                    "Waiting for a destination"), DIM
        if c.stop_d is not None and c.stop_d < 25:
            return ("Stop sign: full stop" if speed < 0.2 else "Stop sign ahead"), WARN
        if c.safety.why == "light" or (speed < 0.2 and c.light == "Red"):
            return ("Red light: stopping" if speed > 0.2 else "Red light"), BAD
        if c.safety.why == "vehicle" and speed < 4.0:
            return "Yielding to traffic", WARN
        side = blinker_state(c.agent, None)
        if side:
            return f"Signalling {side}", WARN
        return f"Driving to {PLACES[c.dest_key]['name']}", BLUE

    def draw_right(self, speed):
        sc, f, c = self.screen, self.f, self.car
        x0 = self.LEFT + 32
        kmh = speed * 3.6
        s = f"{kmh:.0f}"
        self.text(s, f.speed, BAD if kmh > CITY_LIMIT_KMH + 2 else TEXT, (x0, 20))
        sw = f.speed.size(s)[0]
        self.text("km/h", f.unit, DIM, (x0 + sw + 10, 86))
        lim = pygame.Rect(self.W - 32 - 54, 34, 54, 48)
        pygame.draw.rect(sc, TEXT, lim, width=2, border_radius=7)
        lt = f"{CITY_LIMIT_KMH}"
        self.text(lt, f.limit, TEXT, (lim.centerx - f.limit.size(lt)[0] // 2, lim.y + 11))
        st, col = self.status(speed)
        pill = pygame.Rect(x0, 128, f.status.size(st)[0] + 46, 34)
        pygame.draw.rect(sc, (30, 34, 40), pill, border_radius=17)
        pygame.draw.circle(sc, col, (pill.x + 18, pill.centery), 5)
        self.text(st, f.status, TEXT, (pill.x + 32, pill.y + 7))
        rem = c.remaining_m() if c.driving else None
        info = f"{rem:.0f} m to go" if rem is not None else " "
        self.text(info, f.small, DIM, (x0, 176))
        # map: north-up, whole town
        m = pygame.Rect(x0, 210, self.W - x0 - 32, self.H - 210 - 32)
        pygame.draw.rect(sc, (12, 14, 17), m, border_radius=14)
        (lines, (x_min, x_max, y_min, y_max)) = self.roads
        pad = 18
        scale = min((m.width - 2 * pad) / (x_max - x_min), (m.height - 2 * pad) / (y_max - y_min))
        ox = m.x + pad + ((m.width - 2 * pad) - (x_max - x_min) * scale) / 2
        oy = m.y + pad + ((m.height - 2 * pad) - (y_max - y_min) * scale) / 2
        P = lambda x, y: (ox + (x - x_min) * scale, oy + (y - y_min) * scale)  # noqa: E731
        for seg in lines:
            if len(seg) > 1:
                pygame.draw.lines(sc, (52, 57, 64), False, [P(*p) for p in seg], 3)
        for k, pl in PLACES.items():
            px, py = P(*pl["xy"])
            pygame.draw.circle(sc, FAINT, (px, py), 3)
            lw = f.label.size(pl["name"])[0]
            lx = px + 6 if px + 6 + lw < m.right - 6 else px - 6 - lw  # flip left at the right edge
            self.text(pl["name"], f.label, FAINT, (lx, py - 7))
        if c.trace and c.driving:
            pts = [P(w.transform.location.x, w.transform.location.y) for w, _ in c.trace]
            if len(pts) > 1:
                pygame.draw.lines(sc, BLUE, False, pts, 4)
            ex, ey = pts[-1]
            pygame.draw.circle(sc, TEXT, (ex, ey), 7, 2)
        tf = c.ego.get_transform()
        cx, cy = P(tf.location.x, tf.location.y)
        a = math.radians(tf.rotation.yaw)
        tri = [(cx + 10 * math.cos(a), cy + 10 * math.sin(a)),
               (cx + 7 * math.cos(a + 2.5), cy + 7 * math.sin(a + 2.5)),
               (cx + 7 * math.cos(a - 2.5), cy + 7 * math.sin(a - 2.5))]
        pygame.draw.polygon(sc, TEXT, tri)
        self.text("F1 chase camera  ·  F2 stop  ·  Esc quit", f.label, FAINT, (x0, self.H - 24))

    # --- loop -----------------------------------------------------------------------------
    def run(self, autotest=None, autotest_s=45.0):
        """autotest: send this prompt by itself after 3 s, save screenshots every 8 s, quit after autotest_s."""
        clock = pygame.time.Clock()
        t0, shots = time.time(), 0
        while True:
            if autotest is not None:
                el = time.time() - t0
                if el > 3.0 and not self.rows[-1][1] == autotest and not any(r[1] == autotest for r in self.rows):
                    self.send(autotest)
                if el > 8.0 * (shots + 1):
                    shots += 1
                    os.makedirs(os.path.join(HERE, "out", "check"), exist_ok=True)
                    pygame.image.save(self.screen, os.path.join(HERE, "out", "check", f"live_{shots:02d}.png"))
                    self.car.save_view(os.path.join(HERE, "out", "check", f"live_carla_{shots:02d}.jpg"))
                if el > autotest_s:
                    return
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_RETURN:
                        self.send(self.input)
                        self.input = ""
                    elif ev.key == pygame.K_BACKSPACE:
                        self.input = self.input[:-1]
                    elif ev.key == pygame.K_F1:
                        self.car.follow = not self.car.follow
                    elif ev.key == pygame.K_F2:
                        self.car.stop()
                        self.add("check", "STOPPED  by you", WARN)
                elif ev.type == pygame.TEXTINPUT:
                    self.input += ev.text
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    for r, ex in self.chip_rects:
                        if r.collidepoint(ev.pos) and not self.busy:
                            self.send(ex)
                elif ev.type == pygame.MOUSEWHEEL:
                    self.scroll = max(0, self.scroll + ev.y * 30)
            self.take_result()
            speed = self.car.step()
            self.screen.fill(BG)
            self.draw_left()
            self.draw_right(speed)
            pygame.display.flip()
            clock.tick(self.car.fps)


def carla_running():
    try:
        c = carla.Client("127.0.0.1", PORT)
        c.set_timeout(3.0)
        c.get_world()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="west_side", choices=list(PLACES))
    ap.add_argument("--traffic", type=int, default=15)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--level", type=int, default=2, help="CARLA graphics preset 0-3 (2 = High)")
    ap.add_argument("--no-launch", action="store_true", help="connect to a CARLA that is already running")
    ap.add_argument("--offline", action="store_true", help="use the offline parser even if an API key is set")
    ap.add_argument("--autotest", help=argparse.SUPPRESS)  # testing: send this prompt by itself, screenshot, quit
    ap.add_argument("--autotest-secs", type=float, default=45.0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    launched = False
    if not carla_running():
        if a.no_launch:
            sys.exit("CARLA is not running on port 2200 (start it with ./carla4.sh gui)")
        subprocess.run([os.path.join(HERE, "carla4.sh"), "gui"], env={**os.environ, "LEVEL": str(a.level)}, check=True)
        launched = True
        t0 = time.time()
        while not carla_running():
            if time.time() - t0 > 180:
                sys.exit("CARLA did not start (see out/carla4.log)")
            time.sleep(3)
    real = bool(os.environ.get("OPENAI_API_KEY")) and not a.offline
    car = LiveCar(a.start, a.traffic, a.fps)
    try:
        if a.autotest:
            car.add_test_camera()
        App(car, real).run(a.autotest, a.autotest_secs)
    finally:
        car.close()
        pygame.quit()
        if launched:
            subprocess.run([os.path.join(HERE, "carla4.sh"), "stop"])


if __name__ == "__main__":
    main()
