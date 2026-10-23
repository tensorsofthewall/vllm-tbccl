#!/usr/bin/env bash
# Runs INSIDE the manylinux_2_28 container: tools/ci/linux_wheel.sh <out-dir>
# Expects /src/vllm-tbccl (this repository) and /src/tbccl (the TBCCL core at the commit being released against). Builds the CUDA 13 core archive, then builds
# the wheel twice per supported CPython (3.12 and 3.13) with the release recipe (build, auditwheel repair against manylinux_2_28, inspect) and compares the two; one wheel per CPython is left in <out-dir>.
set -euo pipefail
OUT=${1:?output directory}
SRC=/src
git config --global --add safe.directory '*'
HERE=$(cd "$(dirname "$0")" && pwd)
# The system bundle of the image still lists the emSign roots revoked in October 2026
bash "$HERE/distrust_emsign.sh"
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y -q cuda-nvcc-13-0-13.0.88-1 cuda-cudart-devel-13-0-13.0.96-1 cuda-driver-devel-13-0-13.0.96-1
export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p "$OUT/inputs" /w
(cd "$SRC/tbccl" && scripts/package_native.sh --cuda --out /w/core)
cp /w/core/tbccl-*.tar.gz "$OUT/inputs/"
mkdir -p /w/pfx && tar -xzf /w/core/tbccl-*.tar.gz -C /w/pfx
export TBCCL_ROOT=$(ls -d /w/pfx/tbccl-*)
: > "$OUT/reproducibility.txt"; : > "$OUT/build-environment.txt"
for PYTAG in cp312-cp312 cp313-cp313; do
    PYBIN=/opt/python/$PYTAG/bin
    # PyTorch 2.13.0 from PyPI: the Linux build is CUDA 13.0 and brings the CUDA runtime as pip packages, which is what the wheel is repaired against
    bash "$HERE/bootstrap_pip.sh" "$PYBIN/python"
    "$PYBIN/pip" install -q "torch==2.13.0" "build==1.6.1" "auditwheel==6.8.2" "patchelf==0.17.2.4" numpy setuptools wheel
    { echo "# $PYTAG"; "$PYBIN/pip" freeze; } >> "$OUT/build-environment.txt"
    for n in 1 2; do
        rm -rf "$SRC/vllm-tbccl/build"
        (cd "$SRC/vllm-tbccl" && PATH=$PYBIN:$PATH tools/build_release_wheel.sh /w/$PYTAG/wheel$n)
    done
    W1=$(ls /w/$PYTAG/wheel1/*.whl); W2=$(ls /w/$PYTAG/wheel2/*.whl)
    [ "$(basename "$W1")" = "$(basename "$W2")" ] || { echo "the two builds produced different file names" >&2; exit 1; }
    { echo "## $PYTAG"; PATH=$PYBIN:$PATH python "$SRC/vllm-tbccl/tools/release_metadata.py" compare "$W1" "$W2"; } | tee -a "$OUT/reproducibility.txt"
    cp "$W1" "$OUT/"
    cp /w/$PYTAG/wheel1/inspect.json "$OUT/inspect-$PYTAG.json"
done
rpm -qa 'cuda*' 'libnv*' | sort >> "$OUT/build-environment.txt"
