#!/bin/bash
# Phase 69: PP=2 / TP=1 vLLM 0.31.0 on ONE host (two processes, loopback), Qwen3-0.6B, through vllm-tbccl (plugin only, unpatched vLLM).
#   p69_loop.sh up <tag>   |  p69_loop.sh down <tag>
# env: MODEL (default ~/phase48_models/Qwen3-0.6B), EXTRA_ENV (VAR=val ... applied to both nodes), EXTRA_ARGS (vllm serve args), API_PORT, GPU_UTIL (0.3)
CMD=$1; TAG=$2; R=$(cd "$(dirname "$0")/.." && pwd); cd $R; mkdir -p results
SETSID=setsid; command -v setsid >/dev/null || SETSID="perl $(dirname "$0")/_setsid.pl"
MODEL=${MODEL:-$HOME/phase48_models/Qwen3-0.6B}; API_PORT=${API_PORT:-8169}; GU=${GPU_UTIL:-0.3}; PY=${VENV:-$R/.venv-p69}
if [ "$CMD" = down ]; then for f in results/${TAG}_leader.pid results/${TAG}_worker.pid; do [ -f $f ] && kill -- -$(cat $f) 2>/dev/null; done; exit 0; fi
PORT=$((29700 + RANDOM % 90)); echo $PORT > results/${TAG}.port
COMMON="--dtype bfloat16 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --max-model-len 1024 --pipeline-parallel-size 2 --tensor-parallel-size 1 --nnodes 2 --master-addr 127.0.0.1 --master-port $PORT --distributed-executor-backend mp --gpu-memory-utilization $GU"
ENVS="VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=127.0.0.1 TBCCL_LOCAL_ENDPOINT=127.0.0.1:0 $EXTRA_ENV"
env $ENVS $SETSID $PY/bin/vllm serve $MODEL $COMMON --node-rank 1 --headless $EXTRA_ARGS > results/${TAG}_worker.log 2>&1 &
echo $! > results/${TAG}_worker.pid
sleep 2
env $ENVS $SETSID $PY/bin/vllm serve $MODEL $COMMON --node-rank 0 --host 127.0.0.1 --port $API_PORT $EXTRA_ARGS > results/${TAG}_leader.log 2>&1 &
echo $! > results/${TAG}_leader.pid
for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:$API_PORT/v1/models > /dev/null 2>&1 && { echo "UP after ~$((i*2))s api=http://127.0.0.1:$API_PORT"; exit 0; }; kill -0 $(cat results/${TAG}_leader.pid) 2>/dev/null || { echo "leader exited"; exit 2; }; sleep 2; done; echo "timeout"; exit 3
