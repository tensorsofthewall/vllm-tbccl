#!/bin/bash
# Bring up a 2-node vLLM PP=2/TP=1 deployment over TB4: Linux CUDA <-> Mac (Metal or CPU), bf16, eager, no async scheduling.
#   p48_up.sh <A|B|cA|cB> <tag> [clean|trace]
#     A  = Linux CUDA leader (API) + Mac Metal stage      B  = Mac Metal leader (API) + Linux CUDA stage
#     cA = Linux CUDA leader        + Mac CPU stage        cB = Mac CPU leader        + Linux CUDA stage
#   clean = no tracing, no checksums (authoritative timing); trace = in-memory trace flushed by a daemon thread (stage breakdown only)
# Requires PHASE48_MODEL_LINUX / PHASE48_MODEL_MAC (local paths; nothing is ever downloaded). Startup is retried up to 3 times; every
# attempt and its failure reason go to results/<tag>_attempts.txt. Prints "UP attempts=<n> api=<url>" or exits 2.
# Env: EXTRA_ENV (space-separated VAR=value passed to BOTH hosts, e.g. VLLM_PP_LAYER_PARTITION=20,8), API_PORT (default 8123), GPU_LIN/GPU_MAC (gpu-memory-utilization), MAXLEN. Several deployments can run side by side with distinct tags/API_PORTs.
# Teardown: scripts/p48_down.sh <tag>. Only the process group started here is signalled.
set -u
ORI=${1:?orientation A|B|cA|cB}; TAG=${2:?tag}; MODE=${3:-clean}
: "${PHASE48_MODEL_LINUX:?set PHASE48_MODEL_LINUX}" "${PHASE48_MODEL_MAC:?set PHASE48_MODEL_MAC}"
[ -d "$PHASE48_MODEL_LINUX" ] || { echo "missing $PHASE48_MODEL_LINUX"; exit 2; }
ssh tbccl-mac "[ -d '$PHASE48_MODEL_MAC' ]" || { echo "missing $PHASE48_MODEL_MAC on the Mac"; exit 2; }
case $ORI in A|cA) LN=0;; B|cB) LN=1;; *) echo "bad orientation"; exit 2;; esac
case $ORI in A|B) KIND=metal;; *) KIND=cpu;; esac
MN=$((1-LN))
if [ $LN = 0 ]; then LEADER_IP=192.168.3.2; else LEADER_IP=192.168.3.1; fi
MAXLEN=${MAXLEN:-1024}; GPU_LIN=${GPU_LIN:-0.5}; GPU_MAC=${GPU_MAC:-0.2}; API_PORT=${API_PORT:-8123}
cd /mnt/BigChonk/projects/vllm-tbccl; mkdir -p results; ATT=results/${TAG}_attempts.txt; : > "$ATT"

TRACE_L=""; TRACE_M=""
if [ "$MODE" = trace ]; then
  TRACE_L="VLLM_TBCCL_TRACE=1 VLLM_TBCCL_TRACE_STEPS=1 VLLM_TBCCL_TRACE_FILE=results/${TAG}_linux"
  TRACE_M="VLLM_TBCCL_TRACE=1 VLLM_TBCCL_TRACE_STEPS=1 VLLM_TBCCL_TRACE_FILE=results/${TAG}_mac"
fi
if [ $KIND = metal ]; then
  MBIN='~/projects/phase47/.venv/bin/vllm'; MENV="VLLM_METAL_BUILD_FROM_SOURCE=1 VLLM_TBCCL_BACKEND=metal"
else
  MBIN='~/projects/vllm-tbccl/.venv/bin/vllm'; MENV=""
fi
LIFE=${LIFE:-3000}

launch() {
  local PORT=$((29600 + RANDOM % 90))
  echo $PORT > results/${TAG}.port
  if [ $LN = 0 ]; then MADDR=192.168.3.2; else MADDR=192.168.3.1; fi
  local COMMON="--dtype bfloat16 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --max-model-len $MAXLEN --pipeline-parallel-size 2 --tensor-parallel-size 1 --nnodes 2 --master-addr $MADDR --master-port $PORT --distributed-executor-backend mp"
  local lx="--node-rank $LN --gpu-memory-utilization $GPU_LIN" mx="--node-rank $MN --gpu-memory-utilization $GPU_MAC"
  if [ $LN = 0 ]; then lx="$lx --host 192.168.3.2 --port $API_PORT"; mx="$mx --headless"; else mx="$mx --host 192.168.3.1 --port $API_PORT"; lx="$lx --headless"; fi
  rm -f results/${TAG}_linux* results/${TAG}_mac*; ssh tbccl-mac "rm -f ~/projects/vllm-tbccl/results/${TAG}_mac*" 2>/dev/null
  ssh tbccl-mac "export PATH=/opt/homebrew/bin:\$PATH; cd ~/projects/vllm-tbccl && mkdir -p results && VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=192.168.3.1 TBCCL_LOCAL_ENDPOINT=192.168.3.1:0 $MENV $TRACE_M ${EXTRA_ENV:-} perl -e 'alarm $LIFE; exec @ARGV' $MBIN serve $PHASE48_MODEL_MAC $COMMON $mx > results/${TAG}_mac.log 2>&1" > /dev/null 2>&1 &
  echo $! > results/${TAG}_mac.sshpid
  env VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=192.168.3.2 TBCCL_LOCAL_ENDPOINT=192.168.3.2:0 $TRACE_L ${EXTRA_ENV:-} \
    setsid timeout $LIFE .venv/bin/vllm serve $PHASE48_MODEL_LINUX $COMMON $lx > results/${TAG}_linux.log 2>&1 &
  echo $! > results/${TAG}_linux.pid
}

fail_reason() {
  local why=""
  grep -q -E "kIOGPUCommandBufferCallbackErrorTimeout|GPU watchdog|Impacting Interactivity" results/${TAG}_linux.log 2>/dev/null && why="gpu-watchdog(linux)"
  ssh tbccl-mac "grep -q -E 'kIOGPUCommandBufferCallbackErrorTimeout|Impacting Interactivity' ~/projects/vllm-tbccl/results/${TAG}_mac.log" 2>/dev/null && why="gpu-watchdog(mac)"
  [ -z "$why" ] && why=$(grep -h -m1 -E "Error|Traceback|failed" results/${TAG}_linux.log 2>/dev/null | cut -c1-160)
  [ -z "$why" ] && why=$(ssh tbccl-mac "grep -h -m1 -E 'Error|Traceback|failed' ~/projects/vllm-tbccl/results/${TAG}_mac.log" 2>/dev/null | cut -c1-160)
  echo "${why:-unknown/timeout}"
}

for a in 1 2 3; do
  scripts/p48_down.sh "$TAG" > /dev/null 2>&1; sleep 4
  launch; sleep 3
  t0=$(date +%s)
  for i in $(seq 1 120); do
    if grep -q "Application startup complete" results/${TAG}_linux.log 2>/dev/null || ssh tbccl-mac "grep -q 'Application startup complete' ~/projects/vllm-tbccl/results/${TAG}_mac.log" 2>/dev/null; then
      echo "attempt $a: UP after $(( $(date +%s) - t0 ))s" >> "$ATT"; echo "UP attempts=$a api=http://$LEADER_IP:$API_PORT"; exit 0
    fi
    grep -q "Engine core initialization failed" results/${TAG}_linux.log 2>/dev/null && break
    ssh tbccl-mac "grep -q 'Engine core initialization failed' ~/projects/vllm-tbccl/results/${TAG}_mac.log" 2>/dev/null && break
    sleep 5
  done
  echo "attempt $a: FAILED $(fail_reason)" >> "$ATT"
done
echo "FAILED after 3 attempts (see $ATT)"; exit 2
