#!/usr/bin/env bash
# Build and inspect the vllm-tbccl release artifacts (a pure-Python py3-none-any wheel and an sdist):  tools/build_release.sh <out-dir>
set -euo pipefail
OUT=${1:?output directory}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
python -m build -o "$OUT" "$HERE"
python -m twine check "$OUT"/*
python "$HERE/tools/inspect_wheel.py" "$OUT"/*.whl > "$OUT/inspect-wheel.json"
python "$HERE/tools/inspect_wheel.py" "$OUT"/*.tar.gz > "$OUT/inspect-sdist.json"
echo "release artifacts in $OUT:"; ls "$OUT"
