#!/bin/bash
# Phase 70 local (loopback) peer-failure test of a real vLLM PP=2 engine: kill one worker process mid-generation and measure how the engine and the client react.
#   p70_failure_local.sh <tag> <worker: PP0|PP1>     env as p69_loop.sh (VENV, EXTRA_ENV, MODEL, GPU_UTIL, API_PORT) ; never kills by name pattern, only the recorded worker PID
# Prints: healthy check, seconds from the kill until the in-flight client request ends (and how), whether the API still answers, seconds until every
# process of the deployment has exited after a SIGTERM to the leader's process group (bounded by 60 s), leftover process count.
TAG=$1; W=$2; R=$(cd "$(dirname "$0")/.." && pwd); cd $R; API_PORT=${API_PORT:-8169}; MODEL=${MODEL:-$HOME/phase48_models/Qwen3-0.6B}
PY=${VENV:-$R/.venv-p70-cuda}/bin/python
bash scripts/p69_loop.sh up $TAG || { echo "up failed"; bash scripts/p69_loop.sh down $TAG; exit 2; }
HR=$(curl -s -m 60 http://127.0.0.1:$API_PORT/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"1, 2, 3, 4, 5, 6, 7, 8,\",\"max_tokens\":8,\"temperature\":0}")
echo "healthy: $(echo "$HR" | $PY -c 'import json,sys; d=json.load(sys.stdin); print(repr(d["choices"][0]["text"]) if "choices" in d else "ERROR "+json.dumps(d)[:300])' 2>&1 | tail -1)"
WPID=$(ps -axo pid,command | awk -v w="VLLM::Worker_$W" '$2 == w {print $1}' | head -1); echo "killing $W pid=$WPID"
( curl -s -m 60 -o /dev/null -w "in-flight request ended: http=%{http_code} after %{time_total}s\n" http://127.0.0.1:$API_PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Once upon a time\",\"max_tokens\":900,\"temperature\":0,\"ignore_eos\":true}" ) &
CP=$!; sleep 2; T0=$(date +%s.%N); kill -9 $WPID
wait $CP; echo "kill-to-client-end: $(awk -v a=$(date +%s.%N) -v b=$T0 'BEGIN{printf "%.2f", a-b}') s"
sleep 3
echo "api after failure: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$API_PORT/v1/models) / completion: $(curl -s -m 8 -o /dev/null -w '%{http_code}' http://127.0.0.1:$API_PORT/v1/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":2,\"temperature\":0}")"
T1=$(date +%s.%N); bash scripts/p69_loop.sh down $TAG
for i in $(seq 1 120); do n=$(ps -axo command | grep -E "^VLLM::|vllm serve" | grep -v grep | wc -l | tr -d ' '); [ "$n" = 0 ] && break; sleep 0.5; done
echo "teardown: $(awk -v a=$(date +%s.%N) -v b=$T1 'BEGIN{printf "%.1f", a-b}') s, leftover processes: $n"
