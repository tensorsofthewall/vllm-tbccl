#!/usr/bin/env bash
# Build and inspect the vllm-tbccl source distribution:  tools/build_release.sh <out-dir>
# The sdist carries the sources of the native backend (csrc/, setup.py) and the patches; building a wheel from it needs an installed TBCCL (TBCCL_ROOT).
# The platform wheels are built by tools/build_release_wheel.sh (Linux: inside manylinux_2_28; macOS: arm64).
set -euo pipefail
OUT=${1:?output directory}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
python -m build --sdist --no-isolation -o "$OUT" "$HERE"
python -m twine check "$OUT"/*.tar.gz
python "$HERE/tools/inspect_wheel.py" "$OUT"/*.tar.gz > "$OUT/inspect-sdist.json"
echo "release sdist in $OUT:"; ls "$OUT"
