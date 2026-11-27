"""Things that happen during a drive: stop signs, a stalled car, and road works that force a reroute.

Each controller is called once per simulation step from sim.drive. Plan changes are recorded as
plan versions (with the time and reason) so the overlay can show the route changing on the fly.
"""
import math

import carla

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
        lms = [lm for lm in ego_wp.get_landmarks_of_type(40.0, STOP_SIGN, False)]
        ahead = {}
        for lm in lms:  # a sign and its painted line are two landmarks at one stop: keep one per stop
            d = lm.distance - self.ego.bounding_box.extent.x - 1.0
            if all(abs(d - other) > 4.0 for other in ahead.values()):
                ahead[lm.id] = d
        # violations: a sign we were approaching is gone (passed) and we never stopped for it
        for sid in list(self.min_speed):
            if sid not in ahead:
                if sid not in self.served and self.min_speed[sid] > 0.5:
                    self.events.append({"kind": "stop_sign", "i": i, "t": round(t, 3),
                                        "min_speed_kmh": round(self.min_speed[sid] * 3.6, 1)})
                del self.min_speed[sid]
        for sid, d in ahead.items():
            if d < 8.0:
                self.min_speed[sid] = min(self.min_speed.get(sid, 99.0), speed)
        todo = sorted((d, sid) for sid, d in ahead.items() if sid not in self.served)
        if not todo:
            return ctrl, None
        d, sid = todo[0]
        if not self.obey:
            return ctrl, d
        if self.hold_until is not None:
            if t >= self.hold_until:
                self.served.add(sid)
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
                 timeline, run_model_turn):
        self.world, self.wmap, self.ego, self.agent, self.planner = world, wmap, ego, agent, planner
        self.cfg, self.dest, self.pref = cfg, dest_loc, preference
        self.events, self.plans, self.timeline = events, plans, timeline
        self.run_model_turn = run_model_turn
        self.actors = []
        self.stalled = None
        self.lane_changed = False
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

    def destroy(self):
        for a in self.actors:
            try:
                a.destroy()
            except Exception:
                pass

    # --- per step ----------------------------------------------------------------------
    def _set_plan(self, trace, t, reason, who):
        loc = self.ego.get_location()
        nearest = min(wp.transform.location.distance(loc) for wp, _ in trace)
        if nearest > 5.0:  # the plan starts somewhere the car no longer is: same request, from here
            closed = [self.cfg["closure_road"]] if who == "model" else []
            fresh, _ = self.planner.plan(loc, self.dest, self.pref, closed_roads=closed)
            if fresh:
                trace = fresh
                reason += " (re-anchored to the car's position)"
        trace = trim_to(trace, loc)
        self.agent.set_global_plan(trace, stop_waypoint_creation=True, clean_queue=True)
        self.plans.append({"t": round(t, 3), "reason": reason, "by": who, "route_xy": trace_xyz(trace)})

    def step(self, t, i, speed, active_xy):
        ego_loc = self.ego.get_location()
        ego_wp = self.wmap.get_waypoint(ego_loc)
        # 1) stalled car in our lane: the planner changes lanes (deterministic, no model)
        if self.stalled and not self.lane_changed:
            sl = self.stalled.get_location()
            d = ego_loc.distance(sl)
            s_wp = self.wmap.get_waypoint(sl)
            if d < 34.0 and (s_wp.road_id, s_wp.lane_id) == (ego_wp.road_id, ego_wp.lane_id):
                side = self.cfg.get("stall_side", "right")
                path = self.agent._generate_lane_change_path(
                    ego_wp, side, distance_same_lane=1, distance_other_lane=4,
                    lane_change_distance=max(10.0, d * 0.45), check=False, lane_changes=1, step_distance=2)
                if path:
                    rest, _ = self.planner.plan(path[-1][0].transform.location, self.dest, self.pref)
                    trace = path + (rest[1:] if rest else [])
                    self.lane_changed = True
                    self._set_plan(trace, t, f"stalled vehicle {d:.0f} m ahead in lane", "planner")
                    self.timeline.append({"t": round(t, 3), "kind": "planner",
                                          "text": f"Stalled vehicle {d:.0f} m ahead with hazards on. Changing lanes {side}."})
                    self.events.append({"kind": "lane_change", "i": i, "t": round(t, 3), "side": side})
        # 2) road works reported on the current plan (traffic feed): slow down, ask the model to replan.
        #    Triggered by distance along the plan, so it happens before the junction where the plan turns
        #    onto the closed road, while there is still another way to go.
        if self.closure_seen is None:
            along = along_plan(active_xy, ego_loc, self.closure_loc)
            if along is not None and along < self.cfg.get("see_m", 80.0):
                self.closure_seen = t
                self.cruise = self.agent._target_speed
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
