#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then set -- --no-los; fi
exec "$PROJECT_ROOT/build/Aerial_detection_demo" --config "$PROJECT_ROOT/config/video.yaml" "$@"
