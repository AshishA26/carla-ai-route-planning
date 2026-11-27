#!/usr/bin/env bash
# Start the demo's CARLA server WITH a window on your desktop (same medium settings as carla_up.sh).
set -euo pipefail
NAME=llm-carla-sim
PORT="${PORT:-2200}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
"$(dirname "$0")/carla_config/make_ini.sh" "${SCALABILITY:-1}" 20 "${POOL_MB:-1000}"
docker run -d --name "$NAME" --runtime=nvidia --gpus all --net=host \
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/tmp/.Xauthority \
  -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia -e __VK_LAYER_NV_optimus=NVIDIA_only \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "${XAUTHORITY:-$HOME/.Xauthority}:/tmp/.Xauthority:ro" \
  -v "$(cd "$(dirname "$0")" && pwd)/carla_config/Linux:/home/carla/.config/Epic/CarlaUnreal/Saved/Config/Linux" \
  carlasim/carla:0.10.0 \
  bash -c "./CarlaUnreal.sh -nosound -windowed -ResX=1280 -ResY=720 -carla-rpc-port=$PORT"
echo "started $NAME with a window (port $PORT). Logs: docker logs -f $NAME"
