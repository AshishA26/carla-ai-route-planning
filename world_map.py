"""Road graph, named places and weighted route planning on a CARLA map.

Works offline (from the .xodr file) or against a live server's map, so routes can be
planned and checked without starting the simulator.
"""
import math
import os
import sys

import carla
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
from importlib.metadata import version as _pkg_version
CARLA_09 = _pkg_version("carla").startswith("0.9")
sys.path.insert(0, os.path.join(HERE, "vendor_0915" if CARLA_09 else "vendor"))
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402

XODR = os.path.join(HERE, "maps_0915" if CARLA_09 else "maps", "Town10HD.xodr")

# Expected cost of each manoeuvre in seconds, on top of driving time.
SIGNAL_WAIT_S = 9.0
TURN_COST_S = {RoadOption.LEFT: 6.0, RoadOption.RIGHT: 2.0, RoadOption.STRAIGHT: 1.0}
# Risk points per manoeuvre (used by "safest"): unprotected lefts dominate urban crash data.
TURN_RISK = {RoadOption.LEFT: 3.0, RoadOption.RIGHT: 1.0, RoadOption.STRAIGHT: 0.5,
             RoadOption.CHANGELANELEFT: 1.0, RoadOption.CHANGELANERIGHT: 1.0}


def load_offline_map():
    with open(XODR) as f:
        return carla.Map("Town10HD", f.read())


class Planner(GlobalRoutePlanner):
    """GlobalRoutePlanner whose search cost depends on the route preference."""

    def __init__(self, wmap, resolution=2.0):
        super().__init__(wmap, resolution)
        self._cost = None
        self._label_turns()

    def _label_turns(self):
        """CARLA labels every road edge LANEFOLLOW; tag junction edges with the turn they make."""
        for _, _, d in self._graph.edges(data=True):
            d["turn"], d["junction"] = None, None
            if d["type"] != RoadOption.LANEFOLLOW:
                continue
            jid = self.junction_of(d)
            if jid is None:
                continue
            a = d["entry_waypoint"].transform.rotation.yaw
            b = d["exit_waypoint"].transform.rotation.yaw
            delta = (b - a + 180) % 360 - 180  # CARLA yaw grows clockwise: + is a right turn
            d["junction"] = jid
            d["turn"] = RoadOption.RIGHT if delta > 35 else RoadOption.LEFT if delta < -35 else RoadOption.STRAIGHT

    # --- edge costs -----------------------------------------------------------------
    def edge_seconds(self, u, v, d, speed_mps):
        length = d.get("length", 0) * self._sampling_resolution
        t = length / speed_mps
        kind = d.get("turn") or d["type"]
        if kind in TURN_COST_S:
            t += TURN_COST_S[kind]
            if self._junction_signalised(d):
                t += SIGNAL_WAIT_S * 0.5  # half the time it is green
        if kind in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            t += 1.5
        return t

    def edge_risk(self, d):
        kind = d.get("turn") or d["type"]
        return TURN_RISK.get(kind, 0.0) * (1.5 if kind == RoadOption.LEFT and not self._junction_signalised(d) else 1.0)

    def junction_of(self, d):
        for wp in [d["entry_waypoint"], *d.get("path", []), d["exit_waypoint"]]:
            if wp.is_junction:
                return wp.get_junction().id
        return None

    def signal_junctions(self):
        if not hasattr(self, "_signal_jids"):
            jids = set()
            for lm in self._wmap.get_all_landmarks_of_type("1000001"):
                wp = self._wmap.get_waypoint(lm.transform.location)
                for _ in range(15):  # walk up to 30 m forward to the junction this signal guards
                    if wp.is_junction:
                        jids.add(wp.get_junction().id)
                        break
                    nxt = wp.next(2.0)
                    if not nxt:
                        break
                    wp = nxt[0]
            self._signal_jids = jids
        return self._signal_jids

    def _junction_signalised(self, d):
        jid = d["junction"] if "junction" in d else self.junction_of(d)
        return jid is not None and jid in self.signal_junctions()

    def _path_search(self, origin, destination):
        start, end = self._localize(origin), self._localize(destination)
        weight = self._cost or (lambda u, v, d: d.get("length", 1))
        route = nx.dijkstra_path(self._graph, source=start[0], target=end[0], weight=weight)
        route.append(end[1])
        self._last_route = route
        return route

    # --- public -----------------------------------------------------------------------
    def plan(self, origin, destination, preference="balanced", avoid=(), closed_roads=(), cap_kmh=None):
        """Returns (trace, summary). trace is a list of (carla.Waypoint, RoadOption)."""
        speed = (min(cap_kmh, 30) if cap_kmh else 30) / 3.6
        risk_w = {"fastest": 0.0, "balanced": 4.0, "safest": 15.0}[preference]
        closed = set(closed_roads)

        def cost(u, v, d):
            ew = d["entry_waypoint"]
            if ew.road_id in closed or d["exit_waypoint"].road_id in closed:
                return None  # networkx: edge is hidden
            if "left_turns" in avoid and d.get("turn") == RoadOption.LEFT:
                return None
            if d.get("junction") is not None and f"junction_{d['junction']}" in avoid:
                return None
            return self.edge_seconds(u, v, d, speed) + risk_w * self.edge_risk(d)

        self._cost = cost
        try:
            trace = self.trace_route(origin, destination)
        except nx.NetworkXNoPath:
            return None, {"reachable": False}
        finally:
            self._cost = None
        return trace, self.summarize(trace, speed)

    def summarize(self, trace, speed_mps):
        edges = [self._graph.edges[a, b] for a, b in zip(self._last_route, self._last_route[1:-1])]
        dist = sum(a[0].transform.location.distance(b[0].transform.location) for a, b in zip(trace, trace[1:]))
        turns = [{**d, "type": d["turn"]} for d in edges if d.get("turn")]
        eta = sum(self.edge_seconds(None, None, d, speed_mps) for d in edges)
        return {"reachable": True, "distance_m": round(dist), "eta_s": round(eta),
                "left_turns": sum(d["type"] == RoadOption.LEFT for d in turns),
                "right_turns": sum(d["type"] == RoadOption.RIGHT for d in turns),
                "signalised_junctions": sum(self._junction_signalised(d) for d in turns),
                "junctions": len(turns), "turn_sequence": [d["type"].name.lower() for d in turns],
                "risk_score": round(sum(self.edge_risk(d) for d in edges), 1)}


def route_xy(trace):
    return [(wp.transform.location.x, wp.transform.location.y) for wp, _ in trace]
