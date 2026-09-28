#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$PROJECT_ROOT/build" -j"${BUILD_JOBS:-4}"
cmake -E chdir "$PROJECT_ROOT/build" ctest --output-on-failure
cmake --install "$PROJECT_ROOT/build"
echo "Built and tested. Run ./demo.sh for video detection/tracking without fabricated LOS."
