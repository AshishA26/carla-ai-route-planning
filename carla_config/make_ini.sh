#!/usr/bin/env bash
# Writes Unreal user config for the demo's CARLA server. Usage: make_ini.sh <level 0-3> <max_fps> <texture_pool_mb>
L=${1:-1}; FPS=${2:-20}; POOL=${3:-1500}; GI=${GI:-$L}   # GI=2 turns on Lumen bounce light
D="$(dirname "$0")/Linux"
mkdir -p "$D"
cat > "$D/GameUserSettings.ini" <<INI
[ScalabilityGroups]
sg.ResolutionQuality=100
sg.ViewDistanceQuality=$L
sg.AntiAliasingQuality=$L
sg.ShadowQuality=$L
sg.GlobalIlluminationQuality=$GI
sg.ReflectionQuality=$L
sg.PostProcessQuality=$L
sg.TextureQuality=$L
sg.EffectsQuality=$L
sg.FoliageQuality=$L
sg.ShadingQuality=$L
sg.LandscapeQuality=$L

[/Script/Engine.GameUserSettings]
bUseVSync=False
FrameRateLimit=$FPS.000000
ResolutionSizeX=640
ResolutionSizeY=360
LastUserConfirmedResolutionSizeX=640
LastUserConfirmedResolutionSizeY=360
INI
cat > "$D/Engine.ini" <<INI
[SystemSettings]
t.MaxFPS=$FPS
r.Streaming.PoolSize=$POOL
r.Streaming.LimitPoolSizeToVRAM=1
INI
echo "wrote $D (level $L, GI $GI, max $FPS fps, texture pool $POOL MB)"
