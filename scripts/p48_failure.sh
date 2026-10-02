#!/bin/bash
# Full-engine silent-peer test. p48_failure.sh <A|B>
#   A: Linux CUDA leader + Mac Metal stage; the Mac worker is frozen (SIGSTOP) mid-service, then the leader is interrupted (SIGINT).
#   B: Mac Metal leader + Linux CUDA stage; the Linux worker is frozen, then the Mac leader is interrupted.
# Measures: healthy request first; then request in flight against the frozen peer; then seconds from SIGINT until every process of the leader's
# deployment on the leader host has exited (bounded by 60 s). The frozen worker is then killed by the harness (vLLM headless nodes do not self-exit).
# No recovery, retry or restart is attempted.
ORI=$1; export PHASE48_MODEL_LINUX=/mnt/win_hf_models/Qwen3-0.6B PHASE48_MODEL_MAC=/Users/ragnarok/phase48_models/Qwen3-0.6B
cd /mnt/BigChonk/projects/vllm-tbccl; tag=p48f$ORI
API_PORT=8160 GPU_LIN=0.5 scripts/p48_up.sh $ORI $tag clean || exit 2
if [ $ORI = A ]; then API=http://192.168.3.2:8160; MODEL=$PHASE48_MODEL_LINUX; REF=metal; else API=http://192.168.3.1:8160; MODEL=$PHASE48_MODEL_MAC; REF=cuda; fi
echo "healthy: $(.venv/bin/python examples/pp_client.py --url $API --model $MODEL --ref docs/data/phase48/${REF}_ref.json --max-tokens 4 | tail -1)"
if [ $ORI = A ]; then
  WPIDS=$(ssh tbccl-mac "pgrep -f 'VLLM::Worker'"); echo "freezing Mac workers: $WPIDS"; ssh tbccl-mac "kill -STOP $WPIDS"
else
  WPIDS=$(pgrep -f 'VLLM::Worker'); echo "freezing Linux workers: $WPIDS"; kill -STOP $WPIDS
fi
# CURL_M = in-flight client timeout in seconds (default 25; 0 = no in-flight request). uvicorn's graceful shutdown waits for open client
# connections, so a request that never gets a response keeps the API server (and thus "teardown") alive until the client gives up.
CURL_M=${CURL_M:-25}
[ "$CURL_M" != 0 ] && ( curl -s -m $CURL_M -o /dev/null -w "in-flight request ended: http=%{http_code} after %{time_total}s\n" $API/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" ) &
sleep 3
T0=$(date +%s.%N)
if [ $ORI = A ]; then
  kill -INT -- -"$(cat results/${tag}_linux.pid)"
  while pgrep -f "vllm serve /mnt/win_hf_models|VLLM::" > /dev/null; do sleep 0.1; [ "$(awk -v a=$(date +%s.%N) -v b=$T0 'BEGIN{print (a-b>60)?1:0}')" = 1 ] && { echo "leader teardown NOT bounded (60 s)"; break; }; done
else
  ssh tbccl-mac "pkill -INT -f '[v]llm serve .*--port 8160'"
  while ssh tbccl-mac "pgrep -f 'vllm serve .*--port 8160|VLLM::EngineCore' > /dev/null"; do sleep 0.2; [ "$(awk -v a=$(date +%s.%N) -v b=$T0 'BEGIN{print (a-b>60)?1:0}')" = 1 ] && { echo "leader teardown NOT bounded (60 s)"; break; }; done
fi
echo "leader teardown after SIGINT: $(awk -v a=$(date +%s.%N) -v b=$T0 'BEGIN{printf "%.2f", a-b}') s"
wait
if [ $ORI = A ]; then ssh tbccl-mac "kill -CONT $WPIDS; kill -9 $WPIDS" 2>/dev/null; else kill -CONT $WPIDS; kill -9 $WPIDS 2>/dev/null; fi
scripts/p48_down.sh $tag > /dev/null 2>&1
echo "leftovers: linux=$(pgrep -f 'VLLM::' | wc -l) mac=$(ssh tbccl-mac "pgrep -f 'VLLM::|vllm serve' | wc -l")"
