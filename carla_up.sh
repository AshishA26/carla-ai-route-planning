#!/usr/bin/env bash
# Start a private CARLA 0.10 server for this demo (Medium quality, no window, no sound).
# Port 2200 (+2201/2202) so it never collides with a monorepo CARLA on 2000.
# The texture streaming pool is capped (POOL_MB) so it fits an 8 GB laptop GPU next to a camera.
# Override: QUALITY=Low POOL_MB=1000 ./carla_up.sh
set -euo pipefail
QUALITY="${QUALITY:-Epic}"   # CARLA only knows Low and Epic; SCALABILITY sets the real level
PORT="${PORT:-2200}"
POOL_MB="${POOL_MB:-1000}"
SCALABILITY="${SCALABILITY:-1}"   # Unreal preset: 0 low, 1 medium, 2 high, 3 epic
RESX="${RESX:-640}"; RESY="${RESY:-360}"   # main viewport only; camera sensors set their own size
NAME=llm-carla-sim
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "$NAME is already running"; exit 0; fi
docker rm "$NAME" >/dev/null 2>&1 || true   # a stopped container from a previous run (kept for its logs)
"$(dirname "$0")/carla_config/make_ini.sh" "$SCALABILITY" 20 "$POOL_MB"
docker run -d --name "$NAME" --runtime=nvidia --gpus all --net=host \
  -v "$(cd "$(dirname "$0")" && pwd)/carla_config/Linux:/home/carla/.config/Epic/CarlaUnreal/Saved/Config/Linux" \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.10.0 \
  bash -c "./CarlaUnreal.sh -nosound -RenderOffscreen -quality-level=$QUALITY -carla-rpc-port=$PORT -ResX=$RESX -ResY=$RESY -FullStdOutLogOutput"
echo "started $NAME (quality=$QUALITY, scalability=$SCALABILITY, port=$PORT, texture pool ${POOL_MB} MB). Logs: docker logs -f $NAME"
