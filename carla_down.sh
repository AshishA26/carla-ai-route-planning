#!/usr/bin/env bash
# Stop and remove the demo's CARLA server container.
docker stop llm-carla-sim >/dev/null 2>&1 && echo "stopped llm-carla-sim" || echo "llm-carla-sim was not running"
docker rm llm-carla-sim >/dev/null 2>&1 || true
