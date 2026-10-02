#!/bin/bash
# Single-stage (PP=1) server: cuda (Linux) | metal | cpu (Mac), bf16, eager, no async scheduling. Prints "UP api=<url>" or exits 2.
#   p48_single.sh <cuda|metal|cpu> <tag> [clean|trace]      env: API_PORT (default 8124), GPU_LIN, GPU_MAC, MAXLEN
# Requires PHASE48_MODEL_LINUX / PHASE48_MODEL_MAC (never downloads).
set -u
KIND=${1:?cuda|metal|cpu}; TAG=${2:?tag}
: "${PHASE48_MODEL_LINUX:?}" "${PHASE48_MODEL_MAC:?}"
API_PORT=${API_PORT:-8124}; MAXLEN=${MAXLEN:-1024}; LIFE=${LIFE:-3000}
cd /mnt/BigChonk/projects/vllm-tbccl; mkdir -p results
COMMON="--dtype bfloat16 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --max-model-len $MAXLEN --port $API_PORT"
echo $API_PORT > results/${TAG}.port
if [ $KIND = cuda ]; then
  rm -f results/${TAG}_linux.log
  env VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_ENABLE=1 setsid timeout $LIFE .venv/bin/vllm serve $PHASE48_MODEL_LINUX $COMMON --host 192.168.3.2 --gpu-memory-utilization ${GPU_LIN:-0.25} > results/${TAG}_linux.log 2>&1 &
  echo $! > results/${TAG}_linux.pid
  for i in $(seq 1 120); do grep -q "Application startup complete" results/${TAG}_linux.log && { echo "UP api=http://192.168.3.2:$API_PORT"; exit 0; }; sleep 5; done
else
  if [ $KIND = metal ]; then BIN='~/projects/phase47/.venv/bin/vllm'; ENV="VLLM_METAL_BUILD_FROM_SOURCE=1 VLLM_TBCCL_BACKEND=metal ${EXTRA_ENV:-}"; else BIN='~/projects/vllm-tbccl/.venv/bin/vllm'; ENV="VLLM_TARGET_DEVICE=cpu"; fi
  ssh tbccl-mac "rm -f ~/projects/vllm-tbccl/results/${TAG}_mac.log"
  ssh tbccl-mac "export PATH=/opt/homebrew/bin:\$PATH; cd ~/projects/vllm-tbccl && mkdir -p results && VLLM_USE_V2_MODEL_RUNNER=0 $ENV perl -e 'alarm $LIFE; exec @ARGV' $BIN serve $PHASE48_MODEL_MAC $COMMON --host 192.168.3.1 --gpu-memory-utilization ${GPU_MAC:-0.2} > results/${TAG}_mac.log 2>&1" > /dev/null 2>&1 &
  echo $! > results/${TAG}_mac.sshpid
  for i in $(seq 1 120); do ssh tbccl-mac "grep -q 'Application startup complete' ~/projects/vllm-tbccl/results/${TAG}_mac.log" 2>/dev/null && { echo "UP api=http://192.168.3.1:$API_PORT"; exit 0; }; sleep 5; done
fi
echo "FAILED (see results/${TAG}_*.log)"; exit 2
