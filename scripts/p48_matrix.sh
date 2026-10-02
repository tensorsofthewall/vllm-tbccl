#!/bin/bash
# Interleaved benchmark matrix over the given targets: p48_matrix.sh <out.jsonl> name=url=model [name=url=model ...]
# prompts p16/p128/p512 x concurrency 1/4/12, 32 decode tokens, plus 4-token requests at C=1; rounds rotate over targets (A/B/A/B).
OUT=$1; shift
T=(); for t in "$@"; do T+=(--target "$t"); done
cd /mnt/BigChonk/projects/vllm-tbccl
for pr in p16 p128 p512; do
  for c in 1 4 12; do
    .venv/bin/python examples/p48_bench.py "${T[@]}" --prompt $pr --tokens 32 --conc $c --rounds 7 --warmup 2 --raw ${OUT%.jsonl}_raw.jsonl >> $OUT
  done
  .venv/bin/python examples/p48_bench.py "${T[@]}" --prompt $pr --tokens 4 --conc 1 --rounds 7 --warmup 2 --raw ${OUT%.jsonl}_raw.jsonl >> $OUT
done
echo DONE >> $OUT
