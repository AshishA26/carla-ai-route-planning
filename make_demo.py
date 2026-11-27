"""Demo pipeline: model turn -> validator -> CARLA recording -> composited video.

  python make_demo.py plan                 # offline: run every model turn, save transcripts + routes
  python make_demo.py preview [scene]      # offline: overlay stills on a placeholder background
  python make_demo.py record <scene>       # needs ./carla_up.sh: drive and record frames
  python make_demo.py review               # offline: behaviour review of the recorded review drive
  python make_demo.py compose <scene>      # offline: composite final frames + encode <scene>.mp4
  python make_demo.py video                # concatenate every composed scene into demo.mp4

Scenes: route, speed, refuse, review. Add --real to use the OpenAI API (needs OPENAI_API_KEY).
"""
import argparse
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)

import overlay as ov  # noqa: E402
from scenes import PLACES, REVIEW, SCENES, CITY_LIMIT_KMH  # noqa: E402

ORDER = ["route", "speed", "refuse", "review"]
MAP_NAME = "Town10HD_Opt"
REVIEW_DRIVER = {"target_speed_kmh": 38, "follow_threshold_m": 2.5, "max_brake": 1.0,
                 "follow_speed_limits": False, "label": "candidate build: target 38 km/h, 2.5 m follow distance"}


def sdir(scene):
    d = os.path.join(OUT, scene)
    os.makedirs(d, exist_ok=True)
    return d


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsave(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


# --------------------------------------------------------------------------- plan (offline)
def cmd_plan(args):
    from agent import ROUTE_SYSTEM, run_turn
    from tools import TOOL_SPECS, RouteSession, guardrail
    from world_map import Planner, load_offline_map

    wmap = load_offline_map()
    planner = Planner(wmap)
    for key in [k for k in ORDER if k != "review"]:
        sc = SCENES[key]
        s = RouteSession(wmap, planner, sc["start"], sc["command"])
        tr = run_turn(s, TOOL_SPECS, ROUTE_SYSTEM, sc["command"], script=sc["script"], fake=not args.real)
        route = {"start_xy": PLACES[sc["start"]]["xy"], "hold": s.committed is None}
        if s.committed:
            route.update(dest_xy=PLACES[s.committed["destination"]]["xy"], preference=s.committed["route_id"],
                         cap_kmh=s.committed["max_speed_kmh"], summary=s.committed["summary"])
        timing = panel_timing(tr, guardrail(sc["command"]))
        jsave(os.path.join(sdir(key), "transcript.json"), tr)
        jsave(os.path.join(sdir(key), "route.json"), {**route, **timing})
        print(f"{key:7s} {'hold' if route['hold'] else route['preference']:8s} preroll {timing['preroll_s']:.1f} s"
              f"   {tr[-1]['text'][:70]}")
    rv = REVIEW
    jsave(os.path.join(sdir("review"), "route.json"), {
        "start_xy": PLACES[rv["start"]]["xy"], "dest_xy": PLACES[rv["destination"]]["xy"],
        "preference": "balanced", "hold": False, "preroll_s": 2.5, "ribbon_t": 0.5, "cap_kmh": CITY_LIMIT_KMH})
    print("review  drive plan saved (recorded with the candidate driver profile)")


def panel_timing(transcript, guard):
    rows = ov.route_rows(transcript, guard)
    ov.layout(ImageDraw.Draw(Image.new("RGB", (8, 8))), rows, ov.PANEL_W - 2 * ov.PAD)
    end = ov.schedule(rows, 0.6)
    ribbon_t = next((r.reveal for r in rows if r.kind == "check" and r.color == ov.C["pass"]), None)
    return {"preroll_s": round(end + 1.0, 2), "ribbon_t": ribbon_t}


# --------------------------------------------------------------------------- record (CARLA)
def cmd_record(args):
    import sim
    key = args.scene
    route = jload(os.path.join(OUT, key, "route.json"))
    driver = REVIEW_DRIVER if key == "review" else None
    meta = sim.drive(MAP_NAME, route, sdir(key), target_kmh=route.get("cap_kmh") or CITY_LIMIT_KMH,
                     driver=driver, port=args.port, tm_port=args.tm_port, fps=args.fps,
                     width=args.width, height=args.height, traffic=args.traffic,
                     preroll_s=route["preroll_s"], tail_s=4.0 if route["hold"] else 2.5,
                     max_rate=args.max_rate, gpu_hot=args.gpu_hot, gpu_cool=args.gpu_cool)
    print(json.dumps({k: meta[k] for k in ("frames", "reached_goal", "wall_s", "thermal_pause_s")}))


# --------------------------------------------------------------------------- review (offline)
def cmd_review(args):
    from agent import run_turn
    from review import REVIEW_SYSTEM, TOOL_SPECS, ReviewSession, abort_on_contact, scripted_summary, scripted_verdict
    drive = abort_on_contact(jload(os.path.join(OUT, "review", "drive.json")))
    s = ReviewSession(drive)
    script = [(n, scripted_verdict if a == "FROM_EVENTS" else a) for n, a in REVIEW["script"]]
    script.append(("say", scripted_summary))
    tr = run_turn(s, TOOL_SPECS, REVIEW_SYSTEM, REVIEW["command"], script=script, fake=not args.real)
    jsave(os.path.join(OUT, "review", "review.json"), {"events": s.events, "transcript": tr,
                                                        "verdict": s.verdict, "check": s.check})
    for e in s.events:
        print(f"  {e['id']:6s} {e['kind']:16s} {e['t_start']:6.1f}-{e['t_end']:.1f}s")
    if s.check:
        print({k: v for k, v in s.check.items() if k != "rows"})


# --------------------------------------------------------------------------- compose (offline)
_CTX = {}


def _roads():
    path = os.path.join(OUT, "roads.json")
    if not os.path.exists(path):
        from world_map import load_offline_map
        jsave(path, ov.road_polylines(load_offline_map()))
    return jload(path)


def review_rows(rv):
    from review import RULES
    rows = [ov.Row("label", "REVIEWER" + ("  ·  scripted, offline" if rv["transcript"][0]["mode"] == "fake"
                                          else f"  ·  {rv['transcript'][0]['mode']}"))]
    for ev in rv["transcript"]:
        if ev["kind"] == "tool_call" and ev["name"] != "submit_verdict":
            rows.append(ov.Row("mono", ov.fmt_call(ev["name"], ev["args"]), gap=2))
        elif ev["kind"] == "tool_result" and ev["name"] == "find_events":
            n = len(ev["result"])
            rows.append(ov.Row("result", f"{n} event{'s' if n != 1 else ''}" if n else "none", gap=10))
        elif ev["kind"] == "tool_result" and ev["name"] == "get_drive_summary":
            r = ev["result"]
            rows.append(ov.Row("result", f"{r['distance_m']} m in {r['duration_s']} s, max {r['max_speed_kmh']} km/h",
                               gap=10))
    rows.append(ov.Row("label", "FINDINGS, CHECKED AGAINST THE LOG", gap=10))
    chk = rv["check"] or {"rows": [], "missed": [], "unsupported": []}
    for f in chk["rows"]:
        ok = f["status"] == "confirmed"
        sev = RULES.get(f["rule"], {}).get("severity", "minor")
        col = ov.C["bad"] if sev == "critical" else ov.C["warn"] if sev == "major" else ov.C["dim"]
        rows.append(ov.Row("check", f"{f['event_id']}  {RULES[f['rule']]['label']}", col, gap=2))
        rows.append(ov.Row("result", f"{f['note']}  ({'confirmed in log' if ok else 'NOT in log'})", gap=12))
    for m in chk["missed"]:
        rows.append(ov.Row("check", f"{m['id']}  missed by the model", ov.C["bad"], gap=12))
    if not chk["rows"]:
        rows.append(ov.Row("result", "No rule events in this drive.", gap=12))
    agree = chk.get("agree")
    rows.append(ov.Row("check", f"CODE {chk.get('code_verdict', '?').upper()}   ·   MODEL "
                                f"{chk.get('model_verdict', '?').upper()}   ·   {'agree' if agree else 'disagree'}",
                       ov.C["pass"] if agree else ov.C["bad"], gap=14))
    say = next((e["text"] for e in rv["transcript"][::-1] if e["kind"] == "say"), None)
    if say:
        rows += [ov.Row("label", "SUMMARY", gap=6), ov.Row("say", say, gap=0)]
    return rows


def time_map(log, fps, preroll_s, max_stop_s=3.0, keep_s=1.0):
    """Output frame -> (recorded frame, rate). Everything plays in real time; the only edit is that
    a stop longer than max_stop_s after the model panel (waiting at a light) is cut down to
    keep_s on each side."""
    n = len(log)
    start = int(preroll_s * fps)
    stopped = [k >= start and log[k]["speed"] < 0.3 for k in range(n)]
    skip = set()
    k = start
    while k < n:
        if stopped[k]:
            e = k
            while e < n and stopped[e]:
                e += 1
            if e < n and (e - k) / fps > max_stop_s:  # never trim the final stop (arrival)
                skip.update(range(k + int(keep_s * fps), e - int(keep_s * fps)))
            k = e
        else:
            k += 1
    return [(k, 1.0) for k in range(n) if k not in skip]


def compose_frame(i):
    c = _CTX
    meta, log = c["meta"], c["log"]
    fps = meta["fps"]
    tmap = c["tmap"]
    n_out = len(tmap)
    n_rec = len(log)
    ri, rate = tmap[min(i, n_out - 1)]
    row = log[ri]
    t = ri / fps if i < n_out else (n_rec - 1) / fps + (i - n_out + 1) / fps
    src = os.path.join(c["dir"], "frames", f"{ri:05d}.jpg")
    img = Image.open(src).convert("RGB")
    if img.size != (ov.W, ov.H):
        img = img.resize((ov.W, ov.H), Image.LANCZOS)
    img = img.convert("RGBA")
    if i >= n_out:  # review freeze: dim the last frame
        img = Image.blend(img, Image.new("RGBA", img.size, (0, 0, 0, 255)), min(0.45, (i - n_out) / fps))
    sx, sy = ov.W / meta["width"], ov.H / meta["height"]
    rt = c["route"]
    ribbon_t = rt.get("ribbon_t")
    hit = meta.get("aborted_at_s")
    if c["route_pts"] is not None and ribbon_t is not None and t >= ribbon_t and i < n_out \
            and (hit is None or t < hit):
        grow = ov.ease((t - ribbon_t) / 1.2)
        ov.draw_ribbon(img, c["route_pts"], row, np.array(meta["K"]), sx, sy, grow=grow)
    if c["key"] == "review":
        ov.draw_flags(img, c["events"], t, c["rules"])
        ov.draw_timeline(img, c["events"], min(t, n_rec / fps), n_rec / fps, meta["preroll_s"], c["rules"])
        if i < n_out:
            seen = [e for e in c["events"] if e["t_start"] <= t]
            rows = [ov.Row("label", "PLANNER UNDER TEST"), ov.Row("body", REVIEW_DRIVER["label"], gap=20),
                    ov.Row("label", "RULE CHECKS (CODE, LIVE)", gap=10)]
            for e in seen:
                rows.append(ov.Row("check", f"{e['t_start']:5.1f} s   {e['id']}  {c['rules'][e['kind']]['label']}",
                                   ov.C["warn"] if c["rules"][e["kind"]]["severity"] != "minor" else ov.C["dim"], gap=4))
            if not seen:
                rows.append(ov.Row("result", "no events yet", gap=4))
            ov.layout(ImageDraw.Draw(img), rows, ov.PANEL_W - 2 * ov.PAD)
            for r in rows:
                r.reveal = -10
            ov.draw_panel(img, rows, t, ("04", "Behaviour validation"), "Drive under test, rules checked in code")
        else:
            ov.draw_panel(img, c["rows"], (i - n_out) / fps, ("04", "Behaviour validation"),
                          "A model reviews the log, code checks the model")
    else:
        ov.draw_panel(img, c["rows"], t, c["header"], c["sub"])
    if i < n_out:
        remaining = None
        if c["route_pts"] is not None and not rt.get("hold"):
            p = c["route_pts"]
            k = int(np.argmin(np.linalg.norm(p[:, :2] - [row["x"], row["y"]], axis=1)))
            remaining = float(np.sum(np.linalg.norm(np.diff(p[k:, :2], axis=0), axis=1)))
        label = None if rt.get("hold") else (rt.get("preference") or "").upper() + " ROUTE"
        ov.draw_hud(img, row, row.get("limit_kmh") or CITY_LIMIT_KMH, remaining, label, rate)
        ov.draw_minimap(img, c["roads"], c["route_pts"] if not rt.get("hold") else None, row)
    img.convert("RGB").save(os.path.join(c["dir"], "final", f"{i:05d}.jpg"), quality=94)
    return i


def _init(ctx):
    _CTX.update(ctx)
    # rows are rebuilt per process so fonts and layout are local
    key = ctx["key"]
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    if key == "review":
        rows = review_rows(ctx["rv"])
        ov.layout(d, rows, ov.PANEL_W - 2 * ov.PAD)
        ov.schedule(rows, 0.4, step=0.32)
    else:
        from tools import guardrail
        rows = ov.route_rows(ctx["transcript"], guardrail(SCENES[key]["command"]))
        ov.layout(d, rows, ov.PANEL_W - 2 * ov.PAD)
        ov.schedule(rows, 0.6)
    _CTX["rows"] = rows


def build_ctx(key, d):
    drive = jload(os.path.join(d, "drive.json"))
    if key == "review":
        from review import abort_on_contact
        drive = abort_on_contact(drive)
    route = jload(os.path.join(d, "route.json"))
    fps = drive["meta"]["fps"]
    tmap = time_map(drive["log"], fps, route["preroll_s"]) if not route.get("hold") else \
        [(k, 1.0) for k in range(len(drive["log"]))]
    ctx = {"key": key, "dir": d, "meta": drive["meta"], "log": drive["log"], "route": route, "tmap": tmap,
           "roads": _roads(), "route_pts": ov.dense_route(drive["meta"]["route_xy"]) if not route.get("hold") else None}
    if key == "review":
        from review import RULES
        rv = jload(os.path.join(d, "review.json"))
        ctx.update(rv=rv, events=rv["events"], rules=RULES)
        n_rows = len(review_rows(rv))
        extra = int((0.4 + n_rows * 0.5 + 5.0) * drive["meta"]["fps"])
    else:
        ctx.update(transcript=jload(os.path.join(d, "transcript.json")),
                   header=(f"0{ORDER.index(key) + 1}", SCENES[key]["title"]),
                   sub=f"From {PLACES[SCENES[key]['start']]['name']}")
        extra = 0
    return ctx, len(tmap) + extra


def cmd_compose(args):
    key = args.scene
    d = os.path.join(OUT, key)
    os.makedirs(os.path.join(d, "final"), exist_ok=True)
    ctx, n = build_ctx(key, d)
    frames = range(0, n, args.every) if args.every > 1 else range(n)
    with Pool(args.jobs, initializer=_init, initargs=(ctx,)) as pool:
        for k, _ in enumerate(pool.imap(compose_frame, frames, chunksize=8)):
            if k % 100 == 0:
                print(f"  {k}/{len(frames)}", flush=True)
    if args.every == 1:
        encode(os.path.join(d, "final"), ctx["meta"]["fps"], os.path.join(OUT, f"{key}.mp4"))


def encode(frames_dir, fps, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "%05d.jpg"), "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                    "-pix_fmt", "yuv420p", out], check=True)  # native rate: no frame duplication
    print("wrote", out)


def cmd_video(args):
    parts = [os.path.join(OUT, f"{k}.mp4") for k in ORDER if os.path.exists(os.path.join(OUT, f"{k}.mp4"))]
    lst = os.path.join(OUT, "concat.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", os.path.join(OUT, "demo.mp4")], check=True)
    print("wrote", os.path.join(OUT, "demo.mp4"), "from", [os.path.basename(p) for p in parts])


# --------------------------------------------------------------------------- preview (offline)
def cmd_preview(args):
    """Fake a short straight drive on a placeholder background to check the overlay layout."""
    from world_map import load_offline_map
    import carla
    keys = [args.scene] if args.scene else [k for k in ORDER if k != "review"]
    wmap = load_offline_map()
    for key in keys:
        d = os.path.join(OUT, "preview", key)
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        route = jload(os.path.join(OUT, key, "route.json"))
        fps, w, h, fov = 25, 1920, 1080, 80
        from world_map import Planner
        sx, sy = route["start_xy"]
        if route.get("hold"):
            pts = [[sx, sy, 0.0], [sx + 1, sy, 0.0]]
        else:
            dx, dy = route["dest_xy"]
            tr, _ = Planner(wmap).plan(wmap.get_waypoint(carla.Location(x=sx, y=sy)).transform.location,
                                       wmap.get_waypoint(carla.Location(x=dx, y=dy)).transform.location,
                                       route["preference"])
            pts = [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp, _ in tr]
        dense = ov.dense_route(pts)
        n = int((route["preroll_s"] + 6) * fps)
        log, s = [], 0.0
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1))]
        for i in range(n):
            t = i / fps
            v = 0 if (t < route["preroll_s"] or route.get("hold")) else min(8.3, (t - route["preroll_s"]) * 2.5)
            s += v / fps
            k = int(np.searchsorted(seg, s).clip(0, len(dense) - 2))
            p, q = dense[k], dense[k + 1]
            yaw = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            cam_tf = carla.Transform(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                                     carla.Rotation(yaw=yaw)).transform(carla.Location(x=-7.0, z=3.1))
            cam = carla.Transform(cam_tf, carla.Rotation(pitch=-11.0, yaw=yaw))
            log.append({"i": i, "t": t, "x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw": yaw,
                        "speed": v, "limit_kmh": 30, "cam": np.array(cam.get_matrix()).tolist()})
        bg = placeholder_bg(w, h)
        for i in range(n):
            bg.save(os.path.join(d, "frames", f"{i:05d}.jpg"), quality=90) if i == 0 else None
        # every frame points to the same background
        for i in range(1, n):
            dst = os.path.join(d, "frames", f"{i:05d}.jpg")
            if not os.path.exists(dst):
                os.link(os.path.join(d, "frames", "00000.jpg"), dst)
        K = ov.np.array([[w / (2 * math.tan(math.radians(fov) / 2)), 0, w / 2],
                         [0, w / (2 * math.tan(math.radians(fov) / 2)), h / 2], [0, 0, 1]])
        jsave(os.path.join(d, "drive.json"), {"meta": {"fps": fps, "width": w, "height": h, "K": K.tolist(),
                                                       "route_xy": pts, "preroll_s": route["preroll_s"]},
                                              "log": log, "events": []})
        jsave(os.path.join(d, "route.json"), route)
        os.link(os.path.join(OUT, key, "transcript.json"), os.path.join(d, "transcript.json")) \
            if not os.path.exists(os.path.join(d, "transcript.json")) else None
        ctx, _ = build_ctx(key, d)
        ctx["key"] = key
        os.makedirs(os.path.join(d, "final"), exist_ok=True)
        _init(ctx)
        picks = sorted({int(x * fps) for x in (1.2, route["preroll_s"] * 0.5, route["preroll_s"] - 0.2,
                                               route["preroll_s"] + 4)} & set(range(n)))
        for i in picks:
            compose_frame(i)
        print(key, "->", [os.path.join(d, "final", f"{i:05d}.jpg") for i in picks])


def placeholder_bg(w, h):
    """Neutral sky/road split so overlay contrast can be judged without a render."""
    y = np.linspace(0, 1, h)[:, None]
    sky = np.array([120, 140, 160]) * (1 - y) + np.array([170, 175, 180]) * y
    road = np.array([70, 72, 74]) * np.ones_like(y)
    img = np.where(y < 0.45, sky, road)
    img = np.repeat(img[:, None, :], w, axis=1).reshape(h, w, 3)
    return Image.fromarray(img.astype(np.uint8))


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "preview", "record", "review", "compose", "video"])
    ap.add_argument("scene", nargs="?", choices=ORDER)
    ap.add_argument("--real", action="store_true", help="use the OpenAI API instead of scripted turns")
    ap.add_argument("--port", type=int, default=2200)
    ap.add_argument("--tm-port", type=int, default=8200)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--traffic", type=int, default=30)
    ap.add_argument("--max-rate", type=float, default=12.0, help="max simulation ticks per wall-clock second")
    ap.add_argument("--gpu-hot", type=int, default=82, help="pause rendering at this GPU temperature (C)")
    ap.add_argument("--gpu-cool", type=int, default=72, help="resume rendering at this GPU temperature (C)")
    ap.add_argument("--jobs", type=int, default=4, help="compose worker processes")
    ap.add_argument("--every", type=int, default=1, help="compose every Nth frame only (quick check, no mp4)")
    args = ap.parse_args()
    if args.real and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("--real needs OPENAI_API_KEY")
    if args.cmd in ("record", "compose") and not args.scene:
        sys.exit(f"{args.cmd} needs a scene: {', '.join(ORDER)}")
    {"plan": cmd_plan, "preview": cmd_preview, "record": cmd_record, "review": cmd_review,
     "compose": cmd_compose, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
