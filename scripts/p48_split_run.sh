#!/bin/bash
# Layer-split experiment (VLLM_PP_LAYER_PARTITION, honoured identically by upstream vLLM and vllm-metal; no code change):
#   p48_split_run.sh <out.jsonl> <ori>:<rank0layers>,<rank1layers> [...]     e.g.  A:14,14 A:20,8 A:14,14 B:8,20
# For each entry (in the order given, so controls can bracket the variants): bring the pair up CLEAN, check 4 generated tokens against the
# logit-producing stage's reference, run the benchmark (C=1 p16/p128/p512 + C=4 p16, 32 tokens), tear down. The target label is <ori>_<split>_<n>.
export PHASE48_MODEL_LINUX=/mnt/win_hf_models/Qwen3-0.6B PHASE48_MODEL_MAC=/Users/ragnarok/phase48_models/Qwen3-0.6B
cd /mnt/BigChonk/projects/vllm-tbccl; OUT=$1; shift; n=0
D=docs/data/phase48
for spec in "$@"; do
  n=$((n+1)); ori=${spec%%:*}; part=${spec#*:}; label=${ori}_${part/,/-}_$n; tag=p48sp$n
  case $ori in A) API=http://192.168.3.2:8150; MODEL=$PHASE48_MODEL_LINUX; REF=metal;; cA) API=http://192.168.3.2:8150; MODEL=$PHASE48_MODEL_LINUX; REF=cpu;; B) API=http://192.168.3.1:8150; MODEL=$PHASE48_MODEL_MAC; REF=cuda;; cB) API=http://192.168.3.1:8150; MODEL=$PHASE48_MODEL_MAC; REF=cuda;; esac
  EXTRA_ENV="VLLM_PP_LAYER_PARTITION=$part" API_PORT=8150 GPU_LIN=0.5 scripts/p48_up.sh $ori $tag clean > results/up_$tag.txt || { echo "{\"label\":\"$label\",\"error\":\"up failed\"}" >> $OUT; continue; }
  echo "{\"label\":\"$label\",\"check4\":\"$(.venv/bin/python examples/pp_client.py --url $API --model $MODEL --ref $D/${REF}_ref.json --max-tokens 4 | tail -1)\"}" >> $OUT
  .venv/bin/python examples/pp_client.py --url $API --model $MODEL --ref $D/${REF}_ref.json --max-tokens 32 --out $D/split_${label}_ref.json > /dev/null
  for pr in p16 p128 p512; do .venv/bin/python examples/p48_bench.py --target $label=$API=$MODEL --prompt $pr --tokens 32 --conc 1 --rounds 9 --warmup 2 >> $OUT; done
  .venv/bin/python examples/p48_bench.py --target $label=$API=$MODEL --prompt p16 --tokens 32 --conc 4 --rounds 9 --warmup 2 >> $OUT
  scripts/p48_down.sh $tag > /dev/null 2>&1; sleep 5
done
echo DONE >> $OUT
