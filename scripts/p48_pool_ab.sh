#!/bin/bash
# Receive-buffer pool A/B: p48_pool_ab.sh <out.jsonl> <v1> <v2> ...   (each v is 0 or 1 -> VLLM_TBCCL_RECV_POOL; e.g. 0 1 0 1)
# Orientation A (CUDA stage0 -> Metal stage1), default 14/14 split, clean mode, restart per entry. Pool-on entries also get a concurrency-12 and a
# staggered-16 correctness run against the Metal reference.
export PHASE48_MODEL_LINUX=/mnt/win_hf_models/Qwen3-0.6B PHASE48_MODEL_MAC=/Users/ragnarok/phase48_models/Qwen3-0.6B
cd /mnt/BigChonk/projects/vllm-tbccl; OUT=$1; shift; n=0; D=docs/data/phase48; API=http://192.168.3.2:8150; L=$PHASE48_MODEL_LINUX
for v in "$@"; do
  n=$((n+1)); label=A_pool${v}_$n; tag=p48pool$n
  EXTRA_ENV="VLLM_TBCCL_RECV_POOL=$v" API_PORT=8150 GPU_LIN=0.5 scripts/p48_up.sh A $tag clean > results/up_$tag.txt || { echo "{\"label\":\"$label\",\"error\":\"up failed\"}" >> $OUT; continue; }
  echo "{\"label\":\"$label\",\"check4\":\"$(.venv/bin/python examples/pp_client.py --url $API --model $L --ref $D/metal_ref.json --max-tokens 4 | tail -1)\"}" >> $OUT
  if [ "$v" = 1 ]; then
    echo "{\"label\":\"$label\",\"conc12\":\"$(.venv/bin/python examples/pp_concurrent.py --url $API --model $L --ref $D/metal_ref.json --conc 12 --only 0,1,2 | tail -1)\"}" >> $OUT
    echo "{\"label\":\"$label\",\"stagger16\":\"$(.venv/bin/python examples/pp_concurrent.py --url $API --model $L --ref $D/metal_ref.json --conc 16 --stagger 0.4 --random-len --only 0,1,2,3,4 | tail -1)\"}" >> $OUT
  fi
  for pr in p16 p128; do .venv/bin/python examples/p48_bench.py --target $label=$API=$L --prompt $pr --tokens 32 --conc 1 --rounds 11 --warmup 2 >> $OUT; done
  .venv/bin/python examples/p48_bench.py --target $label=$API=$L --prompt p16 --tokens 32 --conc 4 --rounds 9 --warmup 2 >> $OUT
  scripts/p48_down.sh $tag > /dev/null 2>&1; sleep 5
done
echo DONE >> $OUT
