"""Tools the model can call for route planning, and the deterministic validator behind them.

The model only proposes. commit_route runs the validator; nothing reaches the car unless it passes.
"""
import re

import carla

from scenes import CITY_LIMIT_KMH, PLACES

# Raw-text guardrail: checked on the rider's words whether or not the model flags them.
UNSAFE = [
    (r"\brun (the |a |any )?(red|reds|light|lights)\b", "running red lights"),
    (r"\bignore (the )?(lights?|signals?|stop signs?)\b", "ignoring traffic control"),
    (r"\b(on|onto|via) the sidewalk\b", "driving on the sidewalk"),
    (r"\bwrong way\b", "driving against traffic"),
    (r"\b(hit|ram|chase)\b", "deliberate contact"),
]


def guardrail(text):
    for pattern, label in UNSAFE:
        m = re.search(pattern, text, re.I)
        if m:
            return {"match": m.group(0), "label": label}
    return None


TOOL_SPECS = [
    {"name": "get_vehicle_state", "description": "Current position, speed, posted limit and any road closures.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "list_places", "description": "Named destinations on the map.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "plan_routes",
     "description": "Plan candidate routes to a place. Returns one route per preference with distance, ETA, "
                    "turns across traffic and signalised junctions.",
     "parameters": {"type": "object", "properties": {
         "destination": {"type": "string", "enum": list(PLACES)},
         "preferences": {"type": "array", "items": {"type": "string", "enum": ["fastest", "balanced", "safest"]}},
     }, "required": ["destination", "preferences"], "additionalProperties": False}},
    {"name": "commit_route",
     "description": "Ask the vehicle to drive a planned route. A deterministic validator checks it first and "
                    "may reject it; read the result.",
     "parameters": {"type": "object", "properties": {
         "route_id": {"type": "string"},
         "max_speed_kmh": {"type": ["number", "null"]},
         "reason": {"type": "string"},
     }, "required": ["route_id", "max_speed_kmh", "reason"], "additionalProperties": False}},
    {"name": "report_unsafe_request", "description": "Decline a request that would break traffic law or endanger people.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}},
                    "required": ["reason"], "additionalProperties": False}},
]


class RouteSession:
    """State for one command: where the car is, what has been planned, what was committed."""

    def __init__(self, wmap, planner, start_key, command, closed_roads=(), start_loc=None, closures=(),
                 speed_kmh=0.0):
        self.map, self.planner, self.command = wmap, planner, command
        self.start_key = start_key
        self.start = start_loc or self._loc(start_key)
        self.closures = list(closures)  # [{"road": id, "name": str, "ahead_m": float}]
        self.closed_roads = list(closed_roads) + [c["road"] for c in self.closures]
        self.speed_kmh = speed_kmh
        self.routes = {}       # route_id -> {"trace", "summary", "destination"}
        self.committed = None  # {"route_id", "max_speed_kmh", "trace", ...}
        self.refused = None
        self.checks = []       # every validator decision, for the overlay

    def _loc(self, key):
        x, y = PLACES[key]["xy"]
        return self.map.get_waypoint(carla.Location(x=x, y=y)).transform.location

    # --- tools ------------------------------------------------------------------------
    def get_vehicle_state(self):
        at = PLACES[self.start_key]["name"] if self.start_key else "en route"
        return {"at": at, "speed_kmh": round(self.speed_kmh), "posted_limit_kmh": CITY_LIMIT_KMH,
                "road_closures": [f"{c['name']}: road works, all lanes closed {c['ahead_m']:.0f} m ahead"
                                  for c in self.closures], "weather": "clear, dry"}

    def list_places(self):
        return [{"id": k, "name": v["name"], "about": v["about"]} for k, v in PLACES.items() if k != self.start_key]

    def plan_routes(self, destination, preferences):
        if destination not in PLACES:
            return {"error": f"unknown place {destination!r}"}
        out = []
        for pref in preferences or ["balanced"]:
            trace, summary = self.planner.plan(self.start, self._loc(destination), pref, closed_roads=self.closed_roads)
            if trace is None:
                out.append({"route_id": pref, "reachable": False})
                continue
            self.routes[pref] = {"trace": trace, "summary": summary, "destination": destination}
            out.append({"route_id": pref, **{k: summary[k] for k in
                        ("distance_m", "eta_s", "left_turns", "right_turns", "signalised_junctions", "risk_score")}})
        return {"destination": PLACES[destination]["name"], "routes": out}

    def commit_route(self, route_id, max_speed_kmh, reason):
        issues = validate(self, route_id, max_speed_kmh)
        decision = {"route_id": route_id, "max_speed_kmh": max_speed_kmh, "accepted": not issues, "issues": issues}
        self.checks.append(decision)
        if not issues:
            r = self.routes[route_id]
            self.committed = {"route_id": route_id, "max_speed_kmh": max_speed_kmh or CITY_LIMIT_KMH,
                              "reason": reason, **r}
            return {"accepted": True, "driving": PLACES[r["destination"]]["name"], "eta_s": r["summary"]["eta_s"]}
        return {"accepted": False, "issues": issues}

    def report_unsafe_request(self, reason):
        self.refused = reason
        return {"acknowledged": True, "vehicle": "holding position"}

    def call(self, name, args):
        if name not in {t["name"] for t in TOOL_SPECS}:
            return {"error": f"unknown tool {name}"}
        return getattr(self, name)(**args)


def validate(s, route_id, max_speed_kmh):
    """Deterministic checks. The model cannot argue with these."""
    issues = []
    g = guardrail(s.command)
    if g:
        issues.append(f"Guardrail: the command contains \"{g['match']}\" ({g['label']}).")
    if s.refused:
        issues.append("The request was already declined as unsafe.")
    if route_id not in s.routes:
        issues.append(f"Route {route_id!r} was never planned; plan it first.")
        return issues
    r = s.routes[route_id]
    if max_speed_kmh is not None:
        if max_speed_kmh > CITY_LIMIT_KMH:
            issues.append(f"Speed cap {max_speed_kmh:g} km/h is above the posted {CITY_LIMIT_KMH} km/h limit.")
        elif max_speed_kmh < 10:
            issues.append(f"Speed cap {max_speed_kmh:g} km/h would block traffic.")
    closed = set(s.closed_roads)
    if any(wp.road_id in closed for wp, _ in r["trace"]):
        issues.append("Route uses a closed road.")
    if r["summary"]["distance_m"] < 15:
        issues.append("Destination is where the car already is.")
    return issues
