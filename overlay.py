"""Composite recorded CARLA frames with the route ribbon, HUD and the model/validator panel.

Pure post-processing: reads frames/ + drive.json + transcript.json, writes final/ frames.
Runs without the simulator.
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

W, H = 1920, 1080
PANEL_W = 600
PAD = 40

C = {
    "panel": (11, 13, 16, 168), "text": (237, 237, 234), "dim": (139, 144, 150), "faint": (88, 93, 99),
    "rule": (44, 48, 54), "blue": (47, 107, 255), "pass": (63, 184, 127), "warn": (232, 161, 58),
    "bad": (229, 72, 77), "white": (255, 255, 255),
}


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, f"{name}.ttf"), size)


F = {
    "label": font("Geist-500", 15), "body": font("Geist-400", 23), "say": font("Geist-400", 25),
    "title": font("Geist-500", 30), "mono": font("GeistMono-400", 18), "mono_s": font("GeistMono-400", 16),
    "speed": font("Geist-300", 84), "unit": font("Geist-400", 18), "small": font("Geist-400", 17),
    "verdict": font("Geist-600", 22), "status": font("Geist-500", 20),
    "card_title": font("Geist-300", 58), "card_sub": font("Geist-400", 24),
}


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# --------------------------------------------------------------------------- panel rows
class Row:
    """One block in the side panel. kind: label | text | say | mono | result | check."""

    def __init__(self, kind, text, color=None, gap=10, tag=None):
        self.kind, self.text, self.color, self.gap, self.tag = kind, text, color, gap, tag
        self.reveal = 0.0


def fmt_tool_result(name, args, res):
    if name == "get_vehicle_state":
        return [f"{res['at']}  ·  {res['speed_kmh']} km/h  ·  limit {res['posted_limit_kmh']}"
                + (f"  ·  closed: {len(res['road_closures'])}" if res["road_closures"] else "")]
    if name == "list_places":
        return [", ".join(p["name"] for p in res)]
    if name == "plan_routes":
        out = []
        for r in res.get("routes", []):
            if not r.get("reachable", True):
                out.append(f"{r['route_id']:<8} unreachable")
                continue
            lt = r["left_turns"]
            out.append(f"{r['route_id']:<8} {r['distance_m']:>4} m  {r['eta_s']:>3} s  "
                       f"{lt} left turn{'s' if lt != 1 else ''}")
        return out
    if name == "report_unsafe_request":
        return ["vehicle holding position"]
    return []


def fmt_call(name, args):
    if name == "plan_routes":
        return f"plan_routes({args['destination']}, {'/'.join(args['preferences'])})"
    if name == "commit_route":
        cap = args.get("max_speed_kmh")
        return f"commit_route({args['route_id']}, cap={'none' if cap is None else f'{cap:g}'})"
    if name == "find_events":
        return f"find_events({args['kind']})"
    if name == "report_unsafe_request":
        return "report_unsafe_request()"
    if name == "submit_verdict":
        return f"submit_verdict({args['overall']}, {len(args['findings'])} findings)"
    return f"{name}()"


def route_rows(transcript, guard):
    rows = []
    mode = transcript[0].get("mode", "fake") if transcript else "fake"
    for ev in transcript:
        k = ev["kind"]
        if k == "user":
            rows += [Row("label", "RIDER"), Row("body", ev["text"], gap=26)]
            rows.append(Row("label", "MODEL" + ("  ·  scripted, offline" if mode == "fake" else f"  ·  {mode}")))
        elif k == "tool_call":
            rows.append(Row("mono", fmt_call(ev["name"], ev["args"]), gap=4))
        elif k == "tool_result":
            if ev["name"] == "commit_route":
                res = ev["result"]
                if res["accepted"]:
                    rows.append(Row("check", "VALIDATOR  pass", C["pass"], gap=4))
                    rows.append(Row("result", f"driving to {res['driving']}, about {res['eta_s']} s", gap=16))
                else:
                    rows.append(Row("check", "VALIDATOR  reject", C["warn"], gap=4))
                    for issue in res["issues"]:
                        rows.append(Row("result", issue, gap=16))
            else:
                lines = fmt_tool_result(ev["name"], None, ev["result"])
                for j, line in enumerate(lines):
                    rows.append(Row("result", line, gap=16 if j == len(lines) - 1 else 2))
                if ev["name"] == "report_unsafe_request" and guard:
                    rows.append(Row("check", "GUARDRAIL  match", C["bad"], gap=4))
                    rows.append(Row("result", f"“{guard['match']}”: {guard['label']}. "
                                              "Checked in code on the raw command.", gap=16))
        elif k == "say":
            rows += [Row("label", "CAR", gap=8), Row("say", ev["text"], gap=0)]
    return rows


def layout(draw, rows, width):
    """Assign line lists and heights."""
    for r in rows:
        fnt = {"label": F["label"], "body": F["body"], "say": F["say"], "mono": F["mono"],
               "result": F["mono_s"], "check": F["label"]}[r.kind]
        r.font = fnt
        r.lines = wrap(draw, r.text, fnt, width) if r.kind in ("body", "say", "result") else [r.text]
        lh = {"label": 22, "body": 32, "say": 35, "mono": 27, "result": 24, "check": 24}[r.kind]
        r.h = lh * len(r.lines) + r.gap
        r.lh = lh


def schedule(rows, t_start, step=0.55, slow=("body", "say", "check")):
    """Reveal times: rows appear one after another; text-heavy rows take longer."""
    t = t_start
    for r in rows:
        r.reveal = t
        t += step * (2.2 if r.kind in slow else 1.0)
    return t


def draw_panel(img, rows, now, header, sub=None):
    # frosted backdrop: blur what is behind the panel, then tint it
    back = img.crop((0, 0, PANEL_W, H)).filter(ImageFilter.GaussianBlur(18))
    img.paste(back, (0, 0))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rectangle([0, 0, PANEL_W, H], fill=C["panel"])
    d.line([PANEL_W, 0, PANEL_W, H], fill=C["rule"], width=1)
    y = PAD
    d.text((PAD, y), header[0], font=F["label"], fill=C["blue"])
    y += 26
    d.text((PAD, y), header[1], font=F["title"], fill=C["text"])
    y += 46
    if sub:
        d.text((PAD, y), sub, font=F["small"], fill=C["dim"])
        y += 30
    d.line([PAD, y, PANEL_W - PAD, y], fill=C["rule"], width=1)
    y += 24
    top = y
    img.alpha_composite(layer)
    # rows go on their own layer, clipped below the header
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    visible = [r for r in rows if now >= r.reveal]
    total = sum(r.h for r in visible)
    room = H - y - PAD
    if total > room:  # scroll: keep the newest rows on screen
        y -= total - room
    for r in visible:
        a = ease((now - r.reveal) / 0.35)
        color = r.color or {"label": C["dim"], "body": C["text"], "say": C["text"], "mono": C["text"],
                            "result": C["dim"], "check": C["text"]}[r.kind]
        fill = (*color[:3], int(255 * a))
        lines = r.lines
        if r.kind == "body" and rows.index(r) == 1:  # type out the rider's command
            n = int(len(r.text) * ease((now - r.reveal) / 1.4))
            lines = wrap(ImageDraw.Draw(img), r.text[:max(n, 1)], r.font, PANEL_W - 2 * PAD)
            fill = (*color[:3], 255)
        for line in lines:
            if y > PAD - 40:
                if r.kind == "check":
                    d.ellipse([PAD, y + 6, PAD + 9, y + 15], fill=fill)
                    d.text((PAD + 20, y), line, font=r.font, fill=fill)
                elif r.kind == "mono":
                    d.text((PAD, y), "›", font=r.font, fill=(*C["blue"], int(255 * a)))
                    d.text((PAD + 18, y), line, font=r.font, fill=fill)
                elif r.kind == "result":
                    d.text((PAD + 18, y), line, font=r.font, fill=fill)
                else:
                    d.text((PAD, y), line, font=r.font, fill=fill)
            y += r.lh
        y += r.gap
    mask = Image.new("L", img.size, 0)
    fade = 28  # soft edge where rows scroll under the header
    for k in range(fade):
        ImageDraw.Draw(mask).line([0, top + k, PANEL_W, top + k], fill=int(255 * k / fade))
    ImageDraw.Draw(mask).rectangle([0, top + fade, PANEL_W, H], fill=255)
    layer.putalpha(Image.composite(layer.getchannel("A"), Image.new("L", img.size, 0), mask))
    img.alpha_composite(layer)


# --------------------------------------------------------------------------- route ribbon
def project(points_w, cam_to_world, K):
    """points_w: (N,3) world xyz (CARLA). Returns (N,2) pixels and depth."""
    world_to_cam = np.linalg.inv(np.array(cam_to_world))
    P = np.c_[points_w, np.ones(len(points_w))] @ world_to_cam.T
    x, y, z = P[:, 0], P[:, 1], P[:, 2]  # UE: x forward, y right, z up
    u = K[0][0] * y / np.maximum(x, 1e-3) + K[0][2]
    v = -K[1][1] * z / np.maximum(x, 1e-3) + K[1][2]
    return np.c_[u, v], x


def dense_route(route_xyz, step=0.8):
    pts = np.array(route_xyz, dtype=np.float64)
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(np.linalg.norm(b[:2] - a[:2]) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def draw_ribbon(img, route, row, K, sx, sy, grow=1.0, ahead_m=70.0, half_w=0.85, color=None, fade_all=1.0,
                edges=True):
    if grow <= 0 or fade_all <= 0:
        return
    ego = np.array([row["x"], row["y"]])
    d = np.linalg.norm(route[:, :2] - ego, axis=1)
    i0 = int(np.argmin(d))
    seg_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))]
    s0 = seg_len[i0] + 5.5  # start past the bonnet so the ribbon never crosses the car
    s1 = min(seg_len[-1], s0 + ahead_m * grow)
    idx = np.where((seg_len >= s0) & (seg_len <= s1))[0]
    if len(idx) < 2:
        return
    p = route[idx]
    tang = np.gradient(p[:, :2], axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    z = p[:, 2:3] + 0.06
    left = np.c_[p[:, :2] + nrm * half_w, z]
    right = np.c_[p[:, :2] - nrm * half_w, z]
    (lu, ld), (ru, rd) = project(left, row["cam"], K), project(right, row["cam"], K)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    n = len(p)
    col = color or C["blue"]
    for k in range(n - 1):
        if min(ld[k], ld[k + 1], rd[k], rd[k + 1]) < 1.0:
            continue
        d_along = seg_len[idx[k]] - s0
        fade = 1.0 - d_along / max(ahead_m, 1)  # fade out with distance
        fade_in = min(1.0, d_along / 3.0)       # and in over the first 3 m
        alpha = int(135 * max(0.0, min(1.0, fade * 1.6)) * fade_in * fade_all)
        quad = [(lu[k][0] * sx, lu[k][1] * sy), (lu[k + 1][0] * sx, lu[k + 1][1] * sy),
                (ru[k + 1][0] * sx, ru[k + 1][1] * sy), (ru[k][0] * sx, ru[k][1] * sy)]
        dr.polygon(quad, fill=(*col, alpha))
        if edges:  # crisp edges read better than a flat band
            ea = min(255, int(alpha * 1.7))
            dr.line([quad[0], quad[1]], fill=(*col, ea), width=2)
            dr.line([quad[3], quad[2]], fill=(*col, ea), width=2)
    img.alpha_composite(layer)


# --------------------------------------------------------------------------- HUD
_SCRIM = {}


def scrim(size=(620, 330)):
    """Soft dark corner gradient behind the HUD so it reads over bright pavement."""
    if size not in _SCRIM:
        w, h = size
        y, x = np.mgrid[0:h, 0:w]
        r = np.sqrt(((w - x) / w) ** 2 + ((h - y) / h) ** 2)
        a = np.clip(1.15 - r, 0, 1) ** 1.6 * 175
        arr = np.zeros((h, w, 4), np.uint8)
        arr[..., 3] = a.astype(np.uint8)
        arr[..., :3] = (8, 10, 12)
        _SCRIM[size] = Image.fromarray(arr, "RGBA")
    return _SCRIM[size]


def draw_hud(img, row, limit, remaining_m=None, label=None, rate=None):
    sc = scrim()
    img.alpha_composite(sc, (W - sc.width, H - sc.height))
    d = ImageDraw.Draw(img)
    x1, y1 = W - PAD, H - PAD
    kmh = row["speed"] * 3.6
    s = f"{kmh:.0f}"
    tw = d.textlength(s, font=F["speed"])
    over = kmh > limit + 2
    d.text((x1 - tw - 70, y1 - 100), s, font=F["speed"], fill=C["bad"] if over else C["white"])
    d.text((x1 - 60, y1 - 40), "km/h", font=F["unit"], fill=C["dim"])
    # posted limit: plain outlined box, no sign art
    bx = x1 - 52
    d.rounded_rectangle([bx, y1 - 104, bx + 52, y1 - 58], radius=6, outline=C["text"], width=2)
    lt = f"{limit:.0f}"
    d.text((bx + 26 - d.textlength(lt, font=F["verdict"]) / 2, y1 - 94), lt, font=F["verdict"], fill=C["text"])
    if remaining_m is not None:
        txt = f"{remaining_m:.0f} m to go" if remaining_m > 5 else "arrived"
        d.text((x1 - d.textlength(txt, font=F["small"]), y1 - 135), txt, font=F["small"], fill=C["dim"])
    if label:
        d.text((x1 - d.textlength(label, font=F["label"]), y1 - 160), label, font=F["label"], fill=C["blue"])
    if rate and rate > 1:
        tag = f"{rate:g}\u00d7"
        d.text((x1 - d.textlength(tag, font=F["label"]), y1 - 185), tag, font=F["label"], fill=C["dim"])


def draw_minimap(img, roads_xy, route, row, size=300, scale=1.25, old=None, markers=()):
    """Track-up minimap, top right. route: remaining active plan; old: an abandoned plan (faint);
    markers: [(x, y, kind)] with kind in {"stall", "closure"}."""
    m = Image.new("RGBA", (size, size), (11, 13, 16, 205))
    d = ImageDraw.Draw(m)
    cx = cy = size / 2
    yaw = math.radians(row["yaw"])
    c, s = math.cos(-yaw - math.pi / 2), math.sin(-yaw - math.pi / 2)

    def tr(x, y):
        dx, dy = x - row["x"], y - row["y"]
        return cx + (dx * c - dy * s) * scale, cy + (dx * s + dy * c) * scale

    for seg in roads_xy:
        d.line([tr(*p) for p in seg], fill=(62, 67, 74, 255), width=3)
    if old is not None and len(old) > 1:
        pts = [tr(p[0], p[1]) for p in old[::2]]
        for k in range(0, len(pts) - 1, 2):  # dashed
            d.line([pts[k], pts[k + 1]], fill=(150, 154, 160, 150), width=3)
    if route is not None and len(route) > 1:
        d.line([tr(p[0], p[1]) for p in route[::2]], fill=(*C["blue"], 255), width=5, joint="curve")
        ex, ey = tr(route[-1][0], route[-1][1])
        d.ellipse([ex - 7, ey - 7, ex + 7, ey + 7], outline=C["white"], width=3)
    for x, y, kind in markers:
        px, py = tr(x, y)
        if kind == "closure":
            d.line([px - 7, py - 7, px + 7, py + 7], fill=C["bad"], width=4)
            d.line([px - 7, py + 7, px + 7, py - 7], fill=C["bad"], width=4)
        else:
            d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=C["warn"])
    d.polygon([(cx, cy - 10), (cx - 7, cy + 8), (cx + 7, cy + 8)], fill=C["white"])
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=14, fill=255)
    img.paste(m, (W - PAD - size, PAD), mask)


def draw_status(img, text, color, y=None):
    """One line of state at the bottom left of the video area: what the car is doing now."""
    if not text:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = PANEL_W + PAD
    y = y if y is not None else H - PAD - 44
    w = d.textlength(text, font=F["status"]) + 54
    d.rounded_rectangle([x, y, x + w, y + 44], radius=22, fill=(11, 13, 16, 190))
    d.ellipse([x + 18, y + 17, x + 28, y + 27], fill=(*color[:3], 255))
    d.text((x + 38, y + 10), text, font=F["status"], fill=(*C["text"], 255))
    img.alpha_composite(layer)


def road_polylines(wmap, step=2.0):
    """Lane centre polylines for the minimap, from the map topology."""
    lines = []
    for a, b in wmap.get_topology():
        seg = [(a.transform.location.x, a.transform.location.y)]
        wp = a
        for _ in range(400):
            nxt = wp.next(step)
            if not nxt:
                break
            wp = nxt[0]
            seg.append((wp.transform.location.x, wp.transform.location.y))
            if wp.transform.location.distance(b.transform.location) < step:
                break
        seg.append((b.transform.location.x, b.transform.location.y))
        lines.append(seg)
    return lines


# --------------------------------------------------------------------------- event tags
def draw_flags(img, events, now, rules):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y = PANEL_W + PAD, PAD
    for e in events:
        if not (e["t_start"] <= now <= e["t_end"] + 2.0):
            continue
        sev = rules[e["kind"]]["severity"]
        col = C["bad"] if sev == "critical" else C["warn"] if sev == "major" else C["dim"]
        txt = f"{e['id']}   {rules[e['kind']]['label']}"
        w = d.textlength(txt, font=F["small"]) + 40
        d.rounded_rectangle([x, y, x + w, y + 40], radius=8, fill=(11, 13, 16, 225))
        d.ellipse([x + 14, y + 16, x + 23, y + 25], fill=col)
        d.text((x + 32, y + 10), txt, font=F["small"], fill=C["text"])
        y += 50
    img.alpha_composite(layer)


def draw_timeline(img, events, now, total, t0, rules):
    d = ImageDraw.Draw(img)
    x0, x1, y = PANEL_W + PAD, W - PAD - 260, H - PAD - 8
    d.line([x0, y, x1, y], fill=(255, 255, 255, 70), width=2)
    span = max(total - t0, 1e-3)
    px = lambda t: x0 + (x1 - x0) * min(max((t - t0) / span, 0), 1)
    d.line([x0, y, px(now), y], fill=C["white"], width=2)
    for e in events:
        if e["t_start"] > now:
            continue
        sev = rules[e["kind"]]["severity"]
        col = C["bad"] if sev == "critical" else C["warn"] if sev == "major" else C["dim"]
        a, b = px(e["t_start"]), max(px(e["t_end"]), px(e["t_start"]) + 4)
        d.rounded_rectangle([a, y - 5, b, y + 5], radius=3, fill=col)
