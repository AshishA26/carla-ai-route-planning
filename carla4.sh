#!/usr/bin/env bash
# CARLA 0.9.15 (Unreal Engine 4), run natively from ./carla-0.9.15.
#   ./carla4.sh gui     window on your desktop
#   ./carla4.sh         no window (for recording)
#   ./carla4.sh stop
# Graphics: LEVEL 0 low, 1 medium, 2 high (default), 3 epic. Frame rate capped at FPS (default 20).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/carla-0.9.15"
PIDF="$HERE/out/carla4.pid"
LOG="$HERE/out/carla4.log"
PORT="${PORT:-2200}"; LEVEL="${LEVEL:-2}"; FPS="${FPS:-20}"
mkdir -p "$HERE/out"

stop() {
  if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    pkill -TERM -P "$(cat "$PIDF")" 2>/dev/null || true; kill -TERM "$(cat "$PIDF")" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$(cat "$PIDF")" 2>/dev/null || break; sleep 0.5; done
    echo "stopped CARLA 0.9.15"
  else
    echo "CARLA 0.9.15 launcher was not running"
  fi
  rm -f "$PIDF"
  # the engine binary can outlive its launcher script: stop any copy started from this install
  if pkill -TERM -f "$ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" 2>/dev/null; then
    for _ in $(seq 1 20); do pgrep -f "$ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" >/dev/null || break; sleep 0.5; done
    echo "stopped the CARLA engine process"
  fi
}
[[ "${1:-}" == "stop" ]] && { stop; exit 0; }
[[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null && { echo "already running (pid $(cat "$PIDF"))"; exit 0; }

# Unreal user config: quality preset per group + frame cap (read at startup)
CFG="$HOME/.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor"   # where shipping builds read user settings
mkdir -p "$CFG"
cat > "$CFG/GameUserSettings.ini" <<INI
[ScalabilityGroups]
sg.ResolutionQuality=100
sg.ViewDistanceQuality=$LEVEL
sg.AntiAliasingQuality=$LEVEL
sg.ShadowQuality=$LEVEL
sg.PostProcessQuality=$LEVEL
sg.TextureQuality=$LEVEL
sg.EffectsQuality=$LEVEL
sg.FoliageQuality=$LEVEL
sg.ShadingQuality=$LEVEL

[/Script/Engine.GameUserSettings]
bUseVSync=False
FrameRateLimit=$FPS.000000
INI
cat > "$CFG/Engine.ini" <<INI
[SystemSettings]
t.MaxFPS=$FPS
INI

ARGS=(-prefernvidia -vulkan -nosound -carla-rpc-port="$PORT")
if [[ "${1:-}" == "gui" ]]; then
  ARGS+=(-windowed -ResX=1280 -ResY=720)
  export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
else
  ARGS+=(-RenderOffscreen)
fi
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 nohup ./CarlaUE4.sh "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "started CARLA 0.9.15 ${1:-headless} (pid $!, port $PORT, level $LEVEL, max $FPS fps). Log: $LOG"
