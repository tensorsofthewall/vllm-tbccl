#!/bin/bash
# Phase 70: N create -> check -> destroy cycles of a cross-host deployment.   p70_cycles.sh <linuxfirst|macfirst> <tag-prefix> [n]
ORDER=$1; PFX=$2; N=${3:-3}; cd "$(dirname "$0")/.."
API=$([ $ORDER = linuxfirst ] && echo 192.168.3.2 || echo 192.168.3.1); P=.venv-p70-cuda/bin/python
for c in $(seq 1 $N); do
  T=$PFX$c; echo "== cycle $c ($ORDER)"; bash scripts/p69_aer.sh
  t0=$(date +%s); WAIT=60 bash scripts/p70_cross.sh up $T tb $ORDER > results/out_$T.txt 2>&1; tail -1 results/out_$T.txt; echo "startup+probe $(( $(date +%s) - t0 ))s"
  for ph in short concurrent cancel; do $P tools/p69_client.py --api http://$API:8170 --model p70-model --phase $ph --ref docs/data/phase70/ref_linux_cuda.json --out docs/data/phase70/${PFX}_${ph}_c$c.json 2>&1 | tail -1 | cut -c1-200; done
  bash scripts/p70_cross.sh down $T; sleep 6
  echo "leftover linux: $(ps -eo cmd | grep -E 'VLLM::|vllm serve' | grep -v grep | wc -l)  mac: $(ssh tbccl-mac 'ps -axo command | grep -E "vllm serve|VLLM::" | grep -v grep | wc -l' </dev/null)"
  bash scripts/p69_aer.sh
done
echo CYCLES_DONE
