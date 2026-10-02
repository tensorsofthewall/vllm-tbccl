#!/bin/bash
# Launch one node of a 2-node vLLM PP=2/TP=1 deployment over vllm-tbccl.
#   serve_pp2.sh <node_rank 0|1> <master_addr> <master_port> <local_ip> <model_path> [extra vllm args...]
# node 0 serves the OpenAI API (port 8000 by default); node 1 runs headless. Platform is chosen per host (CUDA if present).
NR=$1; MADDR=$2; MPORT=$3; LIP=$4; MODEL=$5; shift 5
export VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=$LIP TBCCL_LOCAL_ENDPOINT=$LIP:0
COMMON=(--dtype float32 --enforce-eager --no-async-scheduling --max-model-len 512 --pipeline-parallel-size 2 --tensor-parallel-size 1
        --nnodes 2 --node-rank $NR --master-addr $MADDR --master-port $MPORT --distributed-executor-backend mp)
if [ "$NR" = 0 ]; then exec vllm serve "$MODEL" "${COMMON[@]}" --host $LIP --port ${API_PORT:-8000} "$@"; else exec vllm serve "$MODEL" "${COMMON[@]}" --headless "$@"; fi
