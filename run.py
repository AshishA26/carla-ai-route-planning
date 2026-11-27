"""One command for the whole demo: start CARLA, plan, record, review, composite, build the video, stop CARLA.

    python run.py                                   # menu: pick scenes or type your own request
    python run.py --scenes route reroute            # just these scenes
    python run.py --all                             # every scene (the full demo)
    python run.py --command "Take me to the museum, no left turns" --start west_side   # your own request

Your own request needs the real model (OPENAI_API_KEY set); the built-in scenes run with the scripted
model unless you pass --real. Recording uses the GPU for a few minutes per scene; the heat watchdog
pauses it at 82 C. CARLA is stopped at the end (also on Ctrl-C) unless --keep-carla.
"""
import argparse
import os
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_demo as md  # noqa: E402
from scenes import PLACES, SCENES  # noqa: E402


def carla_up(level):
    subprocess.run([os.path.join(HERE, "carla4.sh")], env={**os.environ, "LEVEL": str(level)}, check=True)
    import carla
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            c = carla.Client("127.0.0.1", 2200)
            c.set_timeout(4.0)
            c.get_world()
            print("CARLA ready", flush=True)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("CARLA did not come up within 3 minutes (see out/carla4.log)")


def carla_down():
    subprocess.run([os.path.join(HERE, "carla4.sh"), "stop"])


def menu():
    keys = [k for k in md.ORDER]
    print("\nScenes:")
    for n, k in enumerate(keys, 1):
        title = SCENES[k]["title"] if k in SCENES else "Behaviour validation"
        print(f"  {n}. {k:8s} {title}")
    print("  c. your own request (needs OPENAI_API_KEY)")
    pick = input("\nPick scene numbers (e.g. 1 2), 'a' for all, or 'c': ").strip().lower()
    if pick == "c":
        print("\nStart places: " + ", ".join(PLACES))
        start = input("Start at [west_side]: ").strip() or "west_side"
        command = input("Rider says: ").strip()
        return [], command, start
    if pick in ("a", "all", ""):
        return keys, None, None
    chosen = [keys[int(x) - 1] for x in pick.replace(",", " ").split() if x.isdigit() and 0 < int(x) <= len(keys)]
    return chosen, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", choices=md.ORDER)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--command", help="your own rider request (uses the real model)")
    ap.add_argument("--start", default="west_side", choices=list(PLACES), help="start place for --command")
    ap.add_argument("--real", action="store_true", help="use the OpenAI model for the built-in scenes too")
    ap.add_argument("--level", type=int, default=3, help="graphics preset 0-3 (3 = Epic)")
    ap.add_argument("--light", choices=list(md.SUN), default="sunset")
    ap.add_argument("--traffic", type=int, default=20)
    ap.add_argument("--keep-carla", action="store_true", help="leave CARLA running afterwards")
    ap.add_argument("--out", help="output video path (default out/run_<time>.mp4)")
    a = ap.parse_args()

    scenes, command, start = (md.ORDER if a.all else a.scenes or []), a.command, a.start
    if not scenes and not command:
        scenes, command, start = menu()
    real = a.real or bool(command)
    if real and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("This needs the real model: set OPENAI_API_KEY first (or pick built-in scenes without --real).")

    if command:  # an ad-hoc scene from the rider's words
        SCENES["custom"] = {"title": "Your request", "start": start, "command": command, "script": None}
        if "custom" not in md.ORDER:
            md.ORDER.append("custom")
        scenes = list(scenes) + ["custom"]

    common = dict(real=real, port=2200, tm_port=8200, fps=30, width=1920, height=1080, traffic=a.traffic,
                  max_rate=16.0, gpu_hot=82, gpu_cool=72, jobs=4, every=1, light=a.light, sim_limit=180.0)
    out = a.out or os.path.join(md.OUT, f"run_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    print(f"Scenes: {', '.join(scenes)}  |  model: {'OpenAI' if real else 'scripted'}  |  output: {out}")

    md.cmd_plan(SimpleNamespace(**common, only=[s for s in scenes if s != "review"] or ["route"]))
    try:
        carla_up(a.level)
        for s in scenes:
            print(f"\n=== recording {s}", flush=True)
            md.cmd_record(SimpleNamespace(**common, scene=s))
    finally:
        if not a.keep_carla:
            carla_down()
    if "review" in scenes:
        md.cmd_review(SimpleNamespace(**common))
    for s in scenes:
        print(f"\n=== compositing {s}", flush=True)
        md.cmd_compose(SimpleNamespace(**common, scene=s))
    md.cmd_video(SimpleNamespace(**common, only=scenes, out=out))
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
