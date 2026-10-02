#!/bin/bash
# Stage-breakdown runs (TRACE mode, in-memory events; never used for authoritative latency). For each orientation: bring the pair up in
# trace mode, send sequential requests (p16 and p128, 32 tokens), let the dump threads flush, fetch the Mac trace files, tear down.
export PHASE48_MODEL_LINUX=/mnt/win_hf_models/Qwen3-0.6B PHASE48_MODEL_MAC=/Users/ragnarok/phase48_models/Qwen3-0.6B
cd /mnt/BigChonk/projects/vllm-tbccl; mkdir -p results/trace
for ori in ${@:-A B cA}; do
  tag=p48t$ori
  case $ori in A|cA) API=http://192.168.3.2:8140; MODEL=$PHASE48_MODEL_LINUX;; B) API=http://192.168.3.1:8140; MODEL=$PHASE48_MODEL_MAC;; esac
  API_PORT=8140 GPU_LIN=0.5 scripts/p48_up.sh $ori $tag trace || { echo "$ori FAILED"; continue; }
  for pr in p16 p128; do
    .venv/bin/python examples/p48_bench.py --target $ori=$API=$MODEL --prompt $pr --tokens 32 --conc 1 --rounds 12 --warmup 2 > results/trace/${ori}_$pr.json
  done
  sleep 9
  rm -rf results/trace/$ori; mkdir -p results/trace/$ori
  cp results/${tag}_linux*.json results/trace/$ori/ 2>/dev/null
  scp -q "tbccl-mac:projects/vllm-tbccl/results/${tag}_mac*.json" results/trace/$ori/ 2>/dev/null
  scripts/p48_down.sh $tag > /dev/null 2>&1; sleep 5
done
echo TRACEDONE > results/trace/done.txt
