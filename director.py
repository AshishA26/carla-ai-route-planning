"""Things that happen during a drive: stop signs, a stalled car, and road works that force a reroute.

Each controller is called once per simulation step from sim.drive. Plan changes are recorded as
plan versions (with the time and reason) so the overlay can show the route changing on the fly.
"""
import math

import carla
import numpy as np

STOP_SIGN = "206"


class StopSigns:
    """Full stop at every stop sign on the route: brake smoothly to the line, hold, then go.
    A sign passed without stopping is logged as a stop_sign violation either way."""

    def __init__(self, ego, events, obey=True, hold_s=2.0, decel=2.2):
        self.ego, self.events, self.obey = ego, events, obey
        self.hold_s, self.decel = hold_s, decel
        self.served, self.min_speed, self.hold_until = set(), {}, None
        self.active = None

    def step(self, ctrl, ego_wp, speed, t, i):
        """Returns (control, distance to the next stop line or None)."""
        lms = sorted(((lm.distance - self.ego.bounding_box.extent.x - 1.0, lm.id)
                      for lm in ego_wp.get_landmarks_of_type(50.0, STOP_SIGN, False)))
        # a sign and its painted line (sometimes a second line a few metres on) are one stop: group
        # landmarks less than 12 m apart and stop at the last one, the one at the junction mouth
        groups = []
        for d, sid in lms:
            if groups and d - groups[-1][0] < 12.0:
                groups[-1][0] = d
                groups[-1][1].add(sid)
            else:
                groups.append([d, {sid}])
        ahead = {}
        for d, ids in groups:
            key = min(ids)
            if any(x in self.served for x in ids):
                self.served |= ids
            ahead[key] = (d, ids)
        # violations: a stop we were approaching is gone (passed) and we never stopped for it
        for sid in list(self.min_speed):
            if sid not in ahead:
                if sid not in self.served and self.min_speed[sid] > 0.5:
                    self.events.append({"kind": "stop_sign", "i": i, "t": round(t, 3),
                                        "min_speed_kmh": round(self.min_speed[sid] * 3.6, 1)})
                del self.min_speed[sid]
        for sid, (d, _) in ahead.items():
            if d < 8.0:
                self.min_speed[sid] = min(self.min_speed.get(sid, 99.0), speed)
        todo = sorted((d, sid) for sid, (d, _) in ahead.items() if sid not in self.served)
        if not todo:
            return ctrl, None
        d, sid = todo[0]
        if not self.obey:
            return ctrl, d
        if self.hold_until is not None:
            if t >= self.hold_until:
                self.served |= ahead[sid][1]
                self.events.append({"kind": "stop_served", "i": i, "t": round(t, 3)})
                self.hold_until = None
                return ctrl, d
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=ctrl.steer), d
        if speed < 0.15 and d < 3.5:
            self.hold_until = t + self.hold_s
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=ctrl.steer), d
        desired = math.sqrt(2.0 * self.decel * max(d, 0.0))
        if d < 35.0 and speed > desired - 0.3:
            brake = min(1.0, max(0.2, 0.25 + (speed - desired) * 0.6))
            return carla.VehicleControl(throttle=0.0, brake=brake, steer=ctrl.steer), d
        if d < 3.5:  # creeping up to the line
            return carla.VehicleControl(throttle=min(ctrl.throttle, 0.25), brake=0.0, steer=ctrl.steer), d
        return ctrl, d


def light_stop_lines(world, wmap):
    """(x, y, fx, fy, light) for every lane a traffic light controls, at the lane's entry to the junction.
    Built from each light's trigger volume (what CARLA itself uses to say a car is at a light), sampled
    across its width so every lane it covers is found; get_stop_waypoints() misses lanes on Town 10."""
    out = []
    for tl in world.get_actors().filter("traffic.traffic_light"):
        base = tl.get_transform()
        yaw = math.radians(base.rotation.yaw)
        centre = base.transform(tl.trigger_volume.location)
        ext = tl.trigger_volume.extent
        seen = set()
        for ax in np.linspace(-ext.x, ext.x, max(2, int(ext.x * 2) + 1)):
            for ay in np.linspace(-ext.y, ext.y, max(2, int(ext.y * 2) + 1)):
                px = centre.x + ax * math.cos(yaw) - ay * math.sin(yaw)
                py = centre.y + ax * math.sin(yaw) + ay * math.cos(yaw)
                p = carla.Location(x=px, y=py, z=centre.z)
                wp = wmap.get_waypoint(p, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is None or wp.is_junction or wp.transform.location.distance(p) > 2.5:
                    continue
                key = (wp.road_id, wp.section_id, wp.lane_id)
                if key in seen:
                    continue
                seen.add(key)
                cur = wp
                for _ in range(80):  # walk to where the lane enters the junction: that is the stop line
                    nx = cur.next(0.5)
                    if not nx or nx[0].is_junction:
                        break
                    cur = nx[0]
                loc, f = cur.transform.location, cur.transform.get_forward_vector()
                out.append((loc.x, loc.y, f.x, f.y, tl))
    return out


class Safety:
    """A supervisor over the CARLA agent, which only sees a light ~5 m out and only checks cars in its
    own lane. Every step, along the path the car is about to drive:
      - signals: brake smoothly to the light's real stop line on red, and on yellow when a comfortable
        stop is possible; wait for green (no right on red);
      - path guard: brake for any vehicle that is, or within ~3 s will be, in the car's path (cars
        crossing a junction, a car turning in) -- this is also what makes a stop-sign start wait for
        cross traffic.
    The result can only lower throttle / raise brake on the agent's control, never add speed."""

    def __init__(self, world, ego, wmap=None, obey_lights=True, guard=True, decel=2.5):
        self.world, self.ego = world, ego
        self.obey_lights, self.guard, self.decel = obey_lights, guard, decel
        self.stop_lines = light_stop_lines(world, wmap or world.get_map())
        self._route = None
        self._k = 0
        self.committed = set()  # lights we decided to go through on yellow
        self.why = None         # what the supervisor braked for this step: "light", "vehicle" or None
        self.light = None       # state of the next light on the path, if within 60 m

    def _index(self, route_xy):
        raw = np.array([[p[0], p[1]] for p in route_xy], dtype=float)
        pts = [raw[0]]
        for a, b in zip(raw[:-1], raw[1:]):  # resample to <= 1 m: plans can have long straight gaps
            n = int(np.hypot(*(b - a)) // 1.0)
            pts += [a + (b - a) * (j / (n + 1)) for j in range(1, n + 1)] + [b]
        pts = np.array(pts)
        keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-3])
        pts = pts[keep] if keep.sum() >= 2 else pts
        seg = np.diff(pts, axis=0)
        ln = np.maximum(np.hypot(seg[:, 0], seg[:, 1]), 1e-6)
        self.pts, self.seg, self.seglen = pts, seg, ln
        self.s = np.concatenate([[0.0], np.cumsum(ln)])
        hdg = seg / ln[:, None]
        self.hdg = np.vstack([hdg, hdg[-1:]])
        lights = []
        for x, y, fx, fy, tl in self.stop_lines:
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            k = int(np.argmin(d))
            if d[k] < 1.8 and self.hdg[k] @ np.array([fx, fy]) > 0.7:
                s_l = self.s[k] + self.hdg[k] @ np.array([x - pts[k, 0], y - pts[k, 1]])
                lights.append((s_l, tl))
        lights.sort(key=lambda e: e[0])
        dedup = []
        for s_l, tl in lights:
            if not dedup or s_l - dedup[-1][0] > 3.0:
                dedup.append((s_l, tl))
        self.lights = dedup
        self._route, self._k = route_xy, None

    def _project(self, q):
        """Arc length along the plan and lateral distance for points q (n x 2)."""
        a = self.pts[:-1]
        rel = q[:, None, :] - a[None, :, :]
        tt = np.clip((rel * self.seg[None]).sum(-1) / (self.seglen ** 2)[None], 0.0, 1.0)
        foot = a[None] + tt[..., None] * self.seg[None]
        dist = np.hypot(*(q[:, None, :] - foot).transpose(2, 0, 1))
        j = np.argmin(dist, axis=1)
        n = np.arange(len(q))
        return self.s[j] + tt[n, j] * self.seglen[j], dist[n, j]

    def step(self, ctrl, route_xy, speed, t):
        self.why, self.light = None, None
        if route_xy is not self._route:
            self._index(route_xy)
        if len(self.pts) < 2:
            return ctrl
        tf = self.ego.get_transform()
        ex, ey = self.ego.bounding_box.extent.x, self.ego.bounding_box.extent.y
        me = np.array([[tf.location.x, tf.location.y]])
        # where the car is on the plan: search near last time's spot so loops in the route don't confuse it
        if self._k is None:
            s_me = float(self._project(me)[0][0])
        else:
            lo = max(0, self._k - 5)
            hi = min(len(self.pts), self._k + 80)
            sub = Safety.__new__(Safety)
            sub.pts, sub.seg, sub.seglen, sub.s = self.pts[lo:hi], self.seg[lo:hi - 1], self.seglen[lo:hi - 1], self.s[lo:hi]
            s_me = float(sub._project(me)[0][0]) if hi - lo >= 2 else float(self._project(me)[0][0])
        self._k = int(np.searchsorted(self.s, s_me))
        need = 0.0  # brake asked for

        if True:  # the light state is tracked for the log even when the driver ignores lights
            for s_l, tl in self.lights:
                d = s_l - s_me - ex
                if d < -1.0:
                    continue
                if d > 60.0:
                    break
                st = tl.get_state()
                if d < 12.0:  # the light we are at (for the log): not one further down the road
                    self.light = str(st).split(".")[-1]
                stop = False
                if st == carla.TrafficLightState.Red:
                    stop = tl.id not in self.committed or d > 6.0
                elif st == carla.TrafficLightState.Yellow:
                    if tl.id not in self.committed:
                        if speed * speed / (2.0 * max(d, 0.1)) < 3.5:
                            stop = True
                        elif d < 15.0:
                            self.committed.add(tl.id)  # too close to stop comfortably: go through
                else:
                    self.committed.discard(tl.id)
                if stop and self.obey_lights:
                    want = math.sqrt(2.0 * self.decel * max(d - 0.5, 0.0))
                    if d < 1.5 and speed < 0.5:
                        need = 1.0
                    elif speed > want - 0.3:
                        need = max(need, min(1.0, max(0.2, 0.25 + (speed - want) * 0.6)))
                    elif d < 4.0:
                        ctrl.throttle = min(ctrl.throttle, 0.25)
                    if need > 0:
                        self.why = "light"
                break

        if self.guard:
            reach = min(45.0, max(14.0, 6.0 + speed * 3.0 + speed * speed / 6.0))
            ahead = (self.s >= s_me - 1.0) & (self.s <= s_me + reach + 3.0)
            idx = np.nonzero(ahead)[0]
            if len(idx) >= 2:
                lo, hi = idx[0], idx[-1] + 1
                sub = Safety.__new__(Safety)
                sub.pts, sub.seg, sub.seglen, sub.s = self.pts[lo:hi], self.seg[lo:hi - 1], self.seglen[lo:hi - 1], self.s[lo:hi]
                taus = (0.0, 0.75, 1.5, 2.25, 3.0) + ((3.75, 4.5) if speed < 3.0 else ())
                for v in self.world.get_actors().filter("vehicle.*"):
                    if v.id == self.ego.id:
                        continue
                    vt = v.get_transform()
                    if vt.location.distance(tf.location) > reach + 30.0:
                        continue
                    vel = v.get_velocity()
                    vx, vy = vel.x, vel.y
                    moving = math.hypot(vx, vy) > 0.5
                    f = vt.get_forward_vector()
                    r = vt.get_right_vector()
                    b = v.bounding_box.extent
                    c = np.array([vt.location.x, vt.location.y])
                    corners = np.array([c + sx * b.x * np.array([f.x, f.y]) + sy * b.y * np.array([r.x, r.y])
                                        for sx in (-1, 1) for sy in (-1, 1)] + [c])
                    s0, lat0 = sub._project(corners)
                    if lat0.min() > 25.0:
                        continue
                    k0 = min(len(self.hdg) - 1, int(np.searchsorted(self.s, float(s0[-1]))))
                    same_dir = (self.hdg[k0] @ np.array([f.x, f.y])) > 0.8
                    if s0.max() < s_me + ex * 0.5:
                        continue  # behind us
                    # straight ahead along the car's own heading, whatever the plan says (the car can sit
                    # a metre or more off the plan line in a turn)
                    hx, hy = math.cos(math.radians(tf.rotation.yaw)), math.sin(math.radians(tf.rotation.yaw))
                    rel = corners - me[0]
                    fx_ = rel @ np.array([hx, hy])
                    fy_ = rel @ np.array([-hy, hx])
                    near = (fx_ > ex) & (fx_ < 6.0 + speed * 2.5) & (np.abs(fy_) < ey + 0.35)
                    if near.any():
                        gap = float(fx_[near].min()) - ex - 2.0
                        along = max(0.0, (vx * hx + vy * hy))
                        want = math.sqrt(along * along * 0.6 + 2.0 * 3.5 * max(gap, 0.0))
                        if gap < 0.5 and along < 1.0:
                            brake = 1.0
                        elif speed > want - 0.2:
                            brake = min(1.0, max(0.4, 0.4 + (speed - want) * 0.8))
                        else:
                            brake = 0.0
                        if brake > need:
                            need, self.why = brake, "vehicle"
                    for tau in taus:
                        if tau > 0 and (not moving or same_dir):
                            break
                        q = corners + tau * np.array([vx, vy])
                        sq, lat = (s0, lat0) if tau == 0 else sub._project(q)
                        hit = lat < ey + 0.6
                        hit &= sq > s_me + ex * 0.5
                        if not hit.any():
                            continue
                        s_hit = float(sq[hit].min())
                        gap = s_hit - s_me - ex - 2.0
                        if tau > 0:
                            t_me = (s_hit - s_me) / max(speed, 2.0)
                            if not (tau - 1.0 < t_me < tau + 2.0) or gap > reach:
                                continue
                        along = 0.0
                        if tau == 0 and same_dir and moving:
                            along = max(0.0, self.hdg[k0] @ np.array([vx, vy]))
                        # the car ahead may brake hard (6 m/s2) while we brake at 3.5 m/s2
                        want = math.sqrt(along * along * 0.6 + 2.0 * 3.5 * max(gap, 0.0))
                        if gap < 0.5 and along < 1.0:
                            brake = 1.0
                        elif speed > want - 0.2:
                            brake = min(1.0, max(0.4, 0.4 + (speed - want) * 0.8))
                        else:
                            continue
                        if brake > need:
                            need, self.why = brake, "vehicle"
                        break
        if need > 0.0:
            return carla.VehicleControl(throttle=0.0, brake=max(need, ctrl.brake), steer=ctrl.steer)
        return ctrl


def trace_xyz(trace):
    return [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp, _ in trace]


def along_plan(route_xy, ego_loc, target_loc, snap=4.0):
    """Distance along a plan from the car to a point on it, or None if the point is not ahead on it."""
    def d2(p, l):
        return (p[0] - l.x) ** 2 + (p[1] - l.y) ** 2
    i0 = min(range(len(route_xy)), key=lambda k: d2(route_xy[k], ego_loc))
    acc = 0.0
    for k in range(i0, len(route_xy) - 1):
        if d2(route_xy[k], target_loc) < snap ** 2:
            return acc
        a, b = route_xy[k], route_xy[k + 1]
        acc += math.hypot(b[0] - a[0], b[1] - a[1])
    return None


def trim_to(trace, loc, ahead=1):
    """Drop the part of a plan that is already behind the car."""
    k = min(range(len(trace)), key=lambda j: trace[j][0].transform.location.distance(loc))
    return trace[min(k + ahead, len(trace) - 1):]


class Reroute:
    """Stalled car (planner changes lanes) then road works (the model replans from where the car is)."""

    PROPS_CONE = ("static.prop.constructioncone", "static.prop.trafficcone01")
    PROPS_SIGN = ("static.prop.warningconstruction", "static.prop.trafficwarning")
    PROPS_BARRIER = ("static.prop.streetbarrier",)

    def __init__(self, world, wmap, ego, agent, planner, cfg, dest_loc, preference, events, plans,
                 timeline, run_model_turn, base_trace=None):
        self.world, self.wmap, self.ego, self.agent, self.planner = world, wmap, ego, agent, planner
        self.cfg, self.dest, self.pref = cfg, dest_loc, preference
        self.events, self.plans, self.timeline = events, plans, timeline
        self.run_model_turn = run_model_turn
        self.base_trace = base_trace  # the route the car set off on (moving back rejoins it)
        self.actors = []
        self.stalled = None
        self.lane_changed = False
        self.stall_seen = None
        self.blinker = None
        self.blink_until = None
        self.returning = False
        self.returned = False
        self.return_at = None
        self.closure_seen = None
        self.pending = None  # (apply_at_t, trace, transcript)
        self.cruise = None

    # --- setup -------------------------------------------------------------------------
    def _bp(self, names):
        lib = self.world.get_blueprint_library()
        for n in names:
            found = lib.filter(n)
            if found:
                return found[0]
        return None

    def setup(self):
        c = self.cfg
        # stalled car with hazard lights, in the route's lane
        wp = self.wmap.get_waypoint(carla.Location(x=c["stall_xy"][0], y=c["stall_xy"][1]))
        tf = wp.transform
        tf.location.z += 0.3
        bp = self._bp(("vehicle.audi.a2", "vehicle.nissan.micra", "vehicle.citroen.c3"))
        if bp.has_attribute("color"):
            bp.set_attribute("color", "70,74,80")
        self.stalled = self.world.try_spawn_actor(bp, tf)
        if self.stalled:
            self.actors.append(self.stalled)
            self.stalled.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
            ls = carla.VehicleLightState
            self.stalled.set_light_state(ls(ls.Position | ls.LeftBlinker | ls.RightBlinker))
        # road works: cones across every lane in the route's direction, a sign and barriers
        cwp = self.wmap.get_waypoint(carla.Location(x=c["closure_xy"][0], y=c["closure_xy"][1]))
        lanes = [cwp]
        for get in ("get_left_lane", "get_right_lane"):
            w = cwp
            while True:
                w = getattr(w, get)()
                if not w or w.lane_type != carla.LaneType.Driving or w.lane_id * cwp.lane_id <= 0:
                    break
                lanes.append(w)
        cone, sign, barrier = self._bp(self.PROPS_CONE), self._bp(self.PROPS_SIGN), self._bp(self.PROPS_BARRIER)
        for lw in lanes:
            base = lw.transform
            right = base.get_right_vector()
            fwd = base.get_forward_vector()
            for k in range(-2, 3):  # five cones across the lane
                off = k * lw.lane_width / 5.0
                loc = carla.Location(base.location.x + right.x * off, base.location.y + right.y * off,
                                     base.location.z + 0.05)
                self._spawn(cone, carla.Transform(loc, base.rotation))
            if barrier:
                loc = carla.Location(base.location.x + fwd.x * 3.0, base.location.y + fwd.y * 3.0, base.location.z)
                self._spawn(barrier, carla.Transform(loc, carla.Rotation(yaw=base.rotation.yaw + 90)))
        if sign:
            s = lanes[0].transform
            back = s.get_forward_vector()
            loc = carla.Location(s.location.x - back.x * 12.0, s.location.y - back.y * 12.0, s.location.z)
            self._spawn(sign, carla.Transform(loc, carla.Rotation(yaw=s.rotation.yaw + 180)))
        self.closure_loc = cwp.transform.location

    def _spawn(self, bp, tf):
        if bp is None:
            return
        a = self.world.try_spawn_actor(bp, tf)
        if a:
            a.set_simulate_physics(False)
            self.actors.append(a)

    def _junction_within(self, wp, dist):
        w, acc = wp, 0.0
        while acc < dist:
            nxt = w.next(2.0)
            if not nxt:
                return False
            w = nxt[0]
            acc += 2.0
            if w.is_junction:
                return True
        return False

    def destroy(self):
        for a in self.actors:
            try:
                a.destroy()
            except Exception:
                pass

    # --- per step ----------------------------------------------------------------------
    def _set_plan(self, trace, t, reason, who, manoeuvre=None):
        """manoeuvre: an explicit lane-change path (already a gentle diagonal) kept as is; only the planner's
        route that follows it has its lane-change jumps smoothed."""
        if manoeuvre:
            trace = list(manoeuvre) + smooth_lane_changes(trace)
        loc = self.ego.get_location()
        ego_wp = self.wmap.get_waypoint(loc)
        near_wp = min(trace, key=lambda p: p[0].transform.location.distance(loc))[0]
        nearest = near_wp.transform.location.distance(loc)
        moved_lane = (near_wp.road_id, near_wp.lane_id) != (ego_wp.road_id, ego_wp.lane_id) and not ego_wp.is_junction
        if who == "model" and (nearest > 5.0 or moved_lane):
            # the plan starts somewhere the car no longer is (another lane): same request, from here
            closed = [self.cfg["closure_road"]] if who == "model" else []
            fresh, _ = self.planner.plan(loc, self.dest, self.pref, closed_roads=closed)
            if fresh:
                trace = fresh
                reason += " (re-anchored to the car's position)"
        trace = trim_to(trace if manoeuvre else smooth_lane_changes(trace), loc)
        self.agent.set_global_plan(trace, stop_waypoint_creation=True, clean_queue=True)
        self.plans.append({"t": round(t, 3), "reason": reason, "by": who, "route_xy": trace_xyz(trace)})

    def step(self, t, i, speed, active_xy):
        ego_loc = self.ego.get_location()
        ego_wp = self.wmap.get_waypoint(ego_loc)
        # 1) stalled car in our lane: the planner notices it, slows and signals, and only then changes
        #    lanes (deterministic, no model)
        side = self.cfg.get("stall_side", "right")
        if self.stalled and self.stall_seen is None:
            sl = self.stalled.get_location()
            d = ego_loc.distance(sl)
            s_wp = self.wmap.get_waypoint(sl)
            if d < 50.0 and (s_wp.road_id, s_wp.lane_id) == (ego_wp.road_id, ego_wp.lane_id):
                self.stall_seen = t
                self.cruise = self.cruise or self.agent._target_speed
                self.agent.set_target_speed(22.0)
                self.blinker = side
                self.timeline.append({"t": round(t, 3), "kind": "planner",
                                      "text": f"Stalled vehicle {d:.0f} m ahead with hazards on. Slowing and signalling, "
                                              f"then changing lanes {side}."})
                self.events.append({"kind": "hazard_seen", "i": i, "t": round(t, 3)})
        if self.stall_seen is not None and not self.lane_changed and t >= self.stall_seen + self.cfg.get("signal_s", 2.0):
            d = ego_loc.distance(self.stalled.get_location())
            path = self.agent._generate_lane_change_path(
                ego_wp, side, distance_same_lane=3, distance_other_lane=8,
                lane_change_distance=min(28.0, max(18.0, d * 0.6)), check=False, lane_changes=1, step_distance=1)
            if path:
                rest, _ = self.planner.plan(path[-1][0].transform.location, self.dest, self.pref)
                self.lane_changed = True
                self.blink_until = t + 4.5
                self.restore_at = t + 5.0
                self._set_plan(rest[1:] if rest else [], t, f"lane change {side}: stalled vehicle {d:.0f} m ahead",
                               "planner", manoeuvre=path)
                self.events.append({"kind": "lane_change", "i": i, "t": round(t, 3), "side": side})
        if self.lane_changed and self.blinker and t >= self.blink_until and not self.returning:
            self.blinker = None
        # 1b) once well past the stalled car, move back to the original lane (keep right), well before junctions
        if self.lane_changed and not self.returned and self.stalled is not None and not ego_wp.is_junction:
            sl = self.stalled.get_location()
            fwd = self.ego.get_transform().get_forward_vector()
            d = ego_loc - sl
            past = d.x * fwd.x + d.y * fwd.y
            back = {"left": "right", "right": "left"}[side]
            nb = ego_wp.get_right_lane() if back == "right" else ego_wp.get_left_lane()
            ok_lane = nb is not None and nb.lane_type == carla.LaneType.Driving and nb.lane_id * ego_wp.lane_id > 0
            if not self.returning and past > 10.0 and ok_lane and not self._junction_within(ego_wp, 35.0):
                self.returning = True
                self.blinker = back
                self.return_at = t + 1.5  # signal first
                self.timeline.append({"t": round(t, 3), "kind": "planner_return",
                                      "text": f"Past the stalled vehicle. Signalling and moving back to the {back} lane."})
            if self.returning and self.return_at is not None and t >= self.return_at:
                path = self.agent._generate_lane_change_path(
                    ego_wp, back, distance_same_lane=3, distance_other_lane=8, lane_change_distance=22.0,
                    check=False, lane_changes=1, step_distance=1)
                if path:
                    end = path[-1][0].transform.location
                    if self.base_trace:  # rejoin the original route just past the manoeuvre
                        rest = trim_to(self.base_trace, end, ahead=2)
                    else:
                        rest, _ = self.planner.plan(end, self.dest, self.pref)
                        rest = rest[1:] if rest else []
                    self._set_plan(rest, t, f"lane change {back}: back to the {back} lane", "planner", manoeuvre=path)
                    self.events.append({"kind": "lane_change", "i": i, "t": round(t, 3), "side": back})
                self.return_at = None
                self.returned = True
                self.blink_until = t + 4.0
        if self.returned and self.blinker and self.blink_until is not None and t >= self.blink_until:
            self.blinker = None
            self.returning = False
        if getattr(self, "restore_at", None) is not None and t >= self.restore_at and self.closure_seen is None:
            self.agent.set_target_speed(self.cruise or 30.0)
            self.restore_at = None
        # 2) road works reported on the current plan (traffic feed): slow down, ask the model to replan.
        #    Triggered by distance along the plan, so it happens before the junction where the plan turns
        #    onto the closed road, while there is still another way to go.
        if self.closure_seen is None and not (self.returning and not self.returned):  # not while signalling
            along = along_plan(active_xy, ego_loc, self.closure_loc)
            if along is not None and along < self.cfg.get("see_m", 80.0):
                self.closure_seen = t
                self.restore_at = None
                self.cruise = self.cruise or self.agent._target_speed
                self.agent.set_target_speed(15.0)
                name = self.cfg.get("closure_name", "the road ahead")
                self.timeline.append({"t": round(t, 3), "kind": "world",
                                      "text": f"Traffic feed: road works on {name}, all lanes closed, {along:.0f} m ahead on the route."})
                trace, transcript = self.run_model_turn(ego_loc, along)
                if trace:
                    self.pending = (t + self.cfg.get("think_s", 3.0), trace, transcript)
                self.timeline.append({"t": round(t, 3), "kind": "model_turn", "transcript": transcript})
        if self.pending and t >= self.pending[0]:
            _, trace, _ = self.pending
            self.pending = None
            self._set_plan(trace, t, "road works: rerouted by the model", "model")
            self.agent.set_target_speed(self.cruise or 30.0)
            self.events.append({"kind": "reroute", "i": i, "t": round(t, 3)})


def curb_pose(wmap, wp):
    """Where to park at the start: a parking/shoulder lane to the right if there is one, otherwise against
    the right curb of the rightmost driving lane. Returns (transform, rightmost driving waypoint)."""
    w = wp
    while True:
        r = w.get_right_lane()
        if r and r.lane_type == carla.LaneType.Driving and r.lane_id * w.lane_id > 0:
            w = r
        else:
            break
    tf = w.transform
    right = tf.get_right_vector()
    r = w.get_right_lane()
    if r and r.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder) and r.lane_width >= 2.0:
        off = (w.lane_width + r.lane_width) / 2.0
    else:
        off = max(0.0, w.lane_width / 2.0 - 1.15)  # half the car plus a little room to the curb
    loc = carla.Location(tf.location.x + right.x * off, tf.location.y + right.y * off, tf.location.z + 0.4)
    return carla.Transform(loc, tf.rotation), w


class PullOut:
    """Parked at the curb: after the model has spoken, signal left, wait at least signal_s (longer if a car
    is coming up behind in the lane), then pull out. The indicator stays on until the car has merged."""

    def __init__(self, ego, world, lane_wp, t_signal, signal_s=2.0, merge_s=3.0, max_wait_s=10.0):
        self.ego, self.world, self.lane_wp = ego, world, lane_wp
        self.t_signal, self.signal_s, self.merge_s, self.max_wait_s = t_signal, signal_s, merge_s, max_wait_s
        self.t_go = None
        self.blinker = None

    def _clear_behind(self):
        tf = self.ego.get_transform()
        fwd = tf.get_forward_vector()
        for v in self.world.get_actors().filter("vehicle.*"):
            if v.id == self.ego.id:
                continue
            d = v.get_location() - tf.location
            along = d.x * fwd.x + d.y * fwd.y
            lateral = abs(-d.x * fwd.y + d.y * fwd.x)
            vel = v.get_velocity()
            if -30.0 < along < 6.0 and lateral < 4.5 and math.hypot(vel.x, vel.y) > 1.0:
                return False
        return True

    def waiting(self, t):
        """True while the car must stay parked (the caller holds the brake)."""
        if t < self.t_signal:
            return True
        if self.t_go is None:
            self.blinker = "left"
            ready = t >= self.t_signal + self.signal_s
            if ready and (self._clear_behind() or t >= self.t_signal + self.max_wait_s):
                self.t_go = t
                return False
            return True
        if t >= self.t_go + self.merge_s:
            self.blinker = None
        return False


def smooth_lane_changes(trace, before_m=4.0, after_m=11.0):
    """CARLA plans a lane change as a jump between neighbouring waypoints, which the controller follows as a
    sudden swerve. Drop the waypoints just before and after each jump, so the car aims diagonally across
    ~15 m of road instead."""
    if len(trace) < 3:
        return trace
    locs = [wp.transform.location for wp, _ in trace]
    drop = set()
    for k in range(1, len(trace)):
        a, b = trace[k - 1][0], trace[k][0]
        if a.road_id == b.road_id and a.lane_id != b.lane_id and not a.is_junction and not b.is_junction:
            acc = 0.0
            j = k - 1
            while j > 0 and acc < before_m:
                acc += locs[j].distance(locs[j - 1])
                drop.add(j)
                j -= 1
            acc = 0.0
            j = k
            while j < len(trace) - 1 and acc < after_m:
                acc += locs[j].distance(locs[j + 1])
                drop.add(j)
                j += 1
    return [p for i, p in enumerate(trace) if i not in drop]


def pullout_curve(park_tf, join_wp, step=1.0):
    """Ribbon points (x, y, z) from the parked car into the lane: a cubic curve that leaves along the car's
    heading and arrives along the lane."""
    p0 = park_tf.location
    p3 = join_wp.transform.location
    f0 = park_tf.get_forward_vector()
    f3 = join_wp.transform.get_forward_vector()
    L = p0.distance(p3)
    c1 = (p0.x + f0.x * L * 0.4, p0.y + f0.y * L * 0.4)
    c2 = (p3.x - f3.x * L * 0.4, p3.y - f3.y * L * 0.4)
    n = max(4, int(L / step))
    pts = []
    for k in range(n + 1):
        t = k / n
        u = 1 - t
        x = u ** 3 * p0.x + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p3.x
        y = u ** 3 * p0.y + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p3.y
        pts.append([x, y, p0.z - 0.4 + (p3.z - p0.z + 0.4) * t])
    return pts
