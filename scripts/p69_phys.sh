#!/bin/bash
# the vLLM 0.31 alignment work physical session over the Thunderbolt link: real vLLM 0.31.0 serving engine, PP=2/TP=1, Linux CUDA stage + Mac CPU stage, unpatched vLLM + vllm-tbccl.
#   p69_phys.sh <A|B> <tag> [trace]      A = Linux first (API on Linux), B = Mac first (API on the Mac)
# Phases (AER read after each): short, medium, sequential, concurrent(12), cancel, shutdown. Stops at the first failed phase or when the operator-set AER gate trips:
#   stop on any new nonfatal/fatal or interface change, or more than ${MAX_NEW_TIMEOUT:-0} new correctable Timeouts since the session start.
ORI=$1; TAG=$2; TRACE=$3; R=$(cd "$(dirname "$0")/.." && pwd); cd $R; D=docs/data/phase69/raw/$TAG; mkdir -p $D
PY=${VENV:-$R/.venv-p69/bin/python}; MODEL=$HOME/phase48_models/Qwen3-0.6B
[ $ORI = A ] && { ORDER=linuxfirst; API=http://192.168.3.2:8170; } || { ORDER=macfirst; API=http://192.168.3.1:8170; }
EXTRA="VLLM_USE_V2_MODEL_RUNNER=0"; [ "$TRACE" = trace ] && EXTRA="$EXTRA VLLM_TBCCL_TRACE=1 VLLM_TBCCL_TRACE_FILE=results/${TAG}_trace"
aer_val() { sed -n "s/.*$1=\([0-9]*\).*/\1/p" <<< "$2"; }
START=$(scripts/p69_aer.sh); echo "start: $START" | tee $D/aer.txt
gate() {  # $1 = label
  local now=$(scripts/p69_aer.sh); echo "after $1: $now" | tee -a $D/aer.txt
  [ "$(aer_val nonfatal "$now")" != "$(aer_val nonfatal "$START")" ] || [ "$(aer_val fatal "$now")" != "$(aer_val fatal "$START")" ] && { echo "GATE: nonfatal/fatal changed"; return 1; }
  [ $(( $(aer_val Timeout "$now") - $(aer_val Timeout "$START") )) -gt ${MAX_NEW_TIMEOUT:-0} ] && { echo "GATE: new correctable Timeouts beyond ${MAX_NEW_TIMEOUT:-0}"; return 1; }
  grep -q "thunderbolt0 up" <<< "$now" || { echo "GATE: interface not up"; return 1; }
  return 0
}
EXTRA_ENV="$EXTRA" WAIT=${WAIT:-100} scripts/p69_cross.sh up $TAG tb $ORDER 2>&1 | tee $D/up.txt | tail -1
grep -q "^UP" $D/up.txt || { echo "engine did not come up"; scripts/p69_cross.sh down $TAG; gate up; exit 2; }
gate up || { scripts/p69_cross.sh down $TAG; exit 3; }
for ph in short medium sequential concurrent cancel; do
  $PY tools/p69_client.py --api $API --model p69-model --phase $ph --ref docs/data/phase69/ref_linux_cuda.json --out $D/$ph.json 2>&1 | tail -1 | cut -c1-400 | tee -a $D/phases.txt
  rc=${PIPESTATUS[0]}
  gate $ph || { scripts/p69_cross.sh down $TAG; exit 3; }
  [ $rc = 0 ] || { echo "PHASE $ph FAILED"; scripts/p69_cross.sh down $TAG; exit 4; }
done
t0=$(date +%s.%N); scripts/p69_cross.sh down $TAG
for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 1500 ] && break; sleep 1; done
echo "shutdown: GPU memory returned after $(python3 -c "import time;print(round(time.time()-$t0,1))") s" | tee -a $D/phases.txt
ssh -o BatchMode=yes tbccl-mac "ps -axo pid,command | grep -c '[v]llm serve'" </dev/null | sed 's/^/Mac vllm processes left: /' | tee -a $D/phases.txt
gate shutdown
cp results/${TAG}_linux.log $D/linux.log 2>/dev/null; ssh -o BatchMode=yes tbccl-mac "cat ~/projects/vllm-tbccl/results/${TAG}_mac.log" </dev/null > $D/mac.log 2>/dev/null
[ "$TRACE" = trace ] && { cp results/${TAG}_trace* $D/ 2>/dev/null; scp -q "tbccl-mac:projects/vllm-tbccl/results/${TAG}_trace*" $D/ 2>/dev/null; }
echo "session $TAG done"
