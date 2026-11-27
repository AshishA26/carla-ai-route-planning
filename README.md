# carla-ai-route-planning

An LLM plans routes and reviews drives in CARLA, while deterministic code makes the decisions.

- **Route planning:** the model turns a rider's request into tool calls (`get_vehicle_state`, `plan_routes`, `commit_route`). A validator checks every commit (speed limit, closures, guardrails on the raw text), and CARLA drives the committed route with traffic.
- **Behaviour validation:** a candidate driving profile is recorded in CARLA. Code computes the rule events (speeding, headway, hard braking, red lights, solid lines, contact). The model reviews the log through tools and must cite event ids, and code checks each finding against the log.

Output: `out/demo.mp4` (1080p, 4 scenes).

## Setup

```bash
# CARLA 0.9.15 (Unreal Engine 4), extracted into ./carla-0.9.15 (git-ignored, 19 GB)
curl -L -o downloads/CARLA_0.9.15.tar.gz https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
mkdir -p carla-0.9.15 && tar -xzf downloads/CARLA_0.9.15.tar.gz -C carla-0.9.15

conda create -n llm-carla-0915 -c conda-forge python=3.10
conda activate llm-carla-0915
pip install carla==0.9.15 numpy networkx shapely opencv-python-headless pillow openai pyyaml fonttools
```

## Run

```bash
./carla4.sh            # start CARLA headless (./carla4.sh gui for a window, ./carla4.sh stop)
python make_demo.py plan                     # model turns + validator (offline)
python make_demo.py record route             # record each scene: route, speed, refuse, review
python make_demo.py review                   # behaviour review of the recorded review drive (offline)
python make_demo.py compose route            # overlay + encode each scene (offline)
python make_demo.py video                    # out/demo.mp4
./carla4.sh stop
```

**Fake vs real model.** Every command runs a scripted model turn by default: the tool calls a model would make are replayed, and every tool, the validator and the scoring still run for real. To use OpenAI, set `OPENAI_API_KEY` (optionally `OPENAI_MODEL`, default `gpt-5-mini`) and pass `--real` to `plan` and `review`.

**GPU care (8 GB laptop).** `carla4.sh` sets Unreal's High preset and a 20 fps idle cap in `~/.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/` (where shipping builds read user settings). Recording runs in synchronous mode at 12 simulation steps per second or fewer, and pauses when the GPU reaches 82 °C (`--gpu-hot`, `--gpu-cool`). Peak use while recording at 1080p was 6.4 GB.

CARLA 0.10 (Unreal Engine 5) also works with this code (`carla_up.sh`, conda env with the 0.10 wheel), but it runs out of GPU memory on 8 GB.

## Files

| File | Role |
|---|---|
| `scenes.py` | Places, rider commands, scripted model turns |
| `tools.py` | Route tools + deterministic validator + guardrail |
| `agent.py` | Tool-calling loop: scripted or OpenAI |
| `world_map.py` | CARLA route planner with fastest / balanced / safest costs |
| `sim.py` | CARLA recording: chase camera, traffic, telemetry, heat control |
| `director.py` | Stop signs, the stalled car and road works, and `Safety`: a supervisor over CARLA's agent that stops at each signal's real stop line on red/yellow and brakes for any vehicle in, or about to cross, the car's path |
| `review.py` | Rule events, reviewer tools, check of the model's findings |
| `overlay.py`, `make_demo.py` | Compositing and the pipeline |
| `vendor_0915/`, `maps_0915/` | CARLA 0.9.15 navigation agents and Town10HD OpenDRIVE (MIT / CC-BY, from the release) |

## Live mode

```bash
python live.py                    # starts CARLA with its window + opens the control window
python live.py --start east_side  # where the car starts (default west_side)
python live.py --no-launch        # use a CARLA already running on port 2200 (./carla4.sh gui)
```

- **CARLA window:** a chase view of the car. The planned path is drawn into the 3D world (blue line), with
  arrows at upcoming turns and a marker at the destination. The car uses CARLA's own model (no Kia overlay),
  with real indicators, brake lights and headlights.
- **Control window:** type a request (or click an example) and press Enter. With `OPENAI_API_KEY` in `.env`
  the OpenAI model calls the tools; without it, a small offline parser does ("take me to the east side, no
  rush", "go 50" gets refused by the validator, "run the reds" is declined). Speed, status and a map of the town
  are on the right. F1 toggles the chase camera, F2 stops the car, Esc quits (and stops CARLA if it started it).

Live mode is separate from the recorded video pipeline (`make_demo.py`, `run.py`), which it does not change.

## Credits

- Vehicle model: "Kia Soul" by Nieve5677 on Sketchfab, licensed CC BY 4.0
  (https://sketchfab.com/3d-models/kia-soul-079c8678206c47a087d0ae47bcfe8055). Changes: repainted white with a
  black roof, sensor rig added from the WATonomous Eve URDF, wheels separated so they turn.
- CARLA 0.9.15 (MIT), Town 10 map (CC BY). Geist fonts (SIL OFL).
