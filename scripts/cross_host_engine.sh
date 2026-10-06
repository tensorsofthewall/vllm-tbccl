#!/bin/bash
# PP=2 / TP=1 vLLM 0.31.0 across Linux (CUDA) and Mac (CPU), unpatched vLLM + vllm-tbccl plugin.
#   cross_host_engine.sh up <tag> <lan|tb> <macfirst|linuxfirst>      cross_host_engine.sh down <tag>
# lan: 192.168.0.x (only macfirst works: the Linux firewall blocks inbound LAN connections); tb: the Thunderbolt link 192.168.3.x (physical traffic: approval + AER gate).
# env: EXTRA_ENV (applied to BOTH hosts), EXTRA_ARGS (vllm serve args, both hosts), API_PORT, GPU_LIN, GPU_MAC, MAXLEN, LIFE, DTYPE
CMD=$1; TAG=$2; R=$(cd "$(dirname "$0")/.." && pwd); cd $R; mkdir -p results
if [ "$CMD" = down ]; then
  [ -f results/${TAG}_linux.pid ] && kill -- -$(cat results/${TAG}_linux.pid) 2>/dev/null
  # macOS: the recorded pid is the exec'd vllm serve (not a process-group leader): terminate its process tree explicitly, never by name
  ssh -o BatchMode=yes tbccl-mac "kt() { for c in \$(pgrep -P \$1); do kt \$c; done; kill \$1 2>/dev/null; }; [ -f /tmp/p69_${TAG}.pid ] && kt \$(cat /tmp/p69_${TAG}.pid); true" </dev/null
  exit 0
fi
NET=$3; ORDER=$4; API_PORT=${API_PORT:-8170}
if [ $NET = tb ]; then LIP=192.168.3.2; MIP=192.168.3.1; else LIP=192.168.0.121; MIP=192.168.0.115; fi
if [ $ORDER = macfirst ]; then MN=0; LN=1; MADDR=$MIP; API=$MIP; else MN=1; LN=0; MADDR=$LIP; API=$LIP; fi
PORT=$((29600 + RANDOM % 90)); echo $PORT > results/${TAG}.port
MODEL_L=${MODEL_L:-$HOME/models/Qwen3-0.6B}; MODEL_M=${MODEL_M:-'~/models/Qwen3-0.6B'}
COMMON="--served-model-name pp-model --dtype ${DTYPE:-bfloat16} --enforce-eager --no-async-scheduling --no-enable-prefix-caching --max-model-len ${MAXLEN:-1024} --pipeline-parallel-size 2 --tensor-parallel-size 1 --nnodes 2 --master-addr $MADDR --master-port $PORT --distributed-executor-backend mp"
lx="--node-rank $LN --gpu-memory-utilization ${GPU_LIN:-0.5}"; mx="--node-rank $MN --gpu-memory-utilization ${GPU_MAC:-0.2}"
if [ $MN = 0 ]; then mx="$mx --host $MIP --port $API_PORT"; lx="$lx --headless"; else lx="$lx --host $LIP --port $API_PORT"; mx="$mx --headless"; fi
LIFE=${LIFE:-1500}
rm -f results/${TAG}_*; 
ssh -o BatchMode=yes tbccl-mac "export PATH=/opt/homebrew/bin:\$PATH; cd ~/projects/vllm-tbccl && mkdir -p results && echo \$\$ > /tmp/vllm_tbccl_${TAG}.pid && VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=$MIP TBCCL_LOCAL_ENDPOINT=$MIP:0 ${EXTRA_ENV:-} exec perl -e 'alarm $LIFE; exec @ARGV' .venv-vllm031/bin/vllm serve $MODEL_M $COMMON $mx ${EXTRA_ARGS:-} > results/${TAG}_mac.log 2>&1" </dev/null > /dev/null 2>&1 &
env VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=$LIP TBCCL_LOCAL_ENDPOINT=$LIP:0 ${EXTRA_ENV:-} setsid timeout $LIFE .venv-vllm031/bin/vllm serve $MODEL_L $COMMON $lx ${EXTRA_ARGS:-} > results/${TAG}_linux.log 2>&1 &
echo $! > results/${TAG}_linux.pid
for i in $(seq 1 ${WAIT:-150}); do
  curl -s -m 3 http://$API:$API_PORT/v1/models > /dev/null 2>&1 && { echo "UP after ~$((i*4))s api=http://$API:$API_PORT"; exit 0; }
  sleep 4
done; echo "not up"; exit 3
