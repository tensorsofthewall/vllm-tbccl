#!/bin/bash
# Phase 69: single-rank vLLM 0.31.0 server (reference generation), no TBCCL involved.  p69_single.sh up <tag> | down <tag>
CMD=$1; TAG=$2; R=$(cd "$(dirname "$0")/.." && pwd); cd $R; mkdir -p results
SETSID=setsid; command -v setsid >/dev/null || SETSID="perl $(dirname "$0")/_setsid.pl"
MODEL=${MODEL:-$HOME/phase48_models/Qwen3-0.6B}; API_PORT=${API_PORT:-8168}; PY=${VENV:-$R/.venv-p69}
if [ "$CMD" = down ]; then [ -f results/${TAG}.pid ] && kill -- -$(cat results/${TAG}.pid) 2>/dev/null; exit 0; fi
env $EXTRA_ENV $SETSID $PY/bin/vllm serve $MODEL --dtype ${DTYPE:-bfloat16} --enforce-eager --no-async-scheduling --no-enable-prefix-caching --max-model-len 1024 --gpu-memory-utilization ${GU:-0.4} --host 127.0.0.1 --port $API_PORT $EXTRA_ARGS > results/${TAG}.log 2>&1 &
echo $! > results/${TAG}.pid
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:$API_PORT/v1/models > /dev/null 2>&1 && { echo "UP after ~$((i*2))s"; exit 0; }; kill -0 $(cat results/${TAG}.pid) 2>/dev/null || { echo exited; exit 2; }; sleep 2; done; echo timeout; exit 3
