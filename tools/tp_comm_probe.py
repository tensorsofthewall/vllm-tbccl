"""TP=2 communication probe: latency of the exact all-reduce / all-gather the audited Qwen3-0.6B TP=2 forward issues, over TBCCL.

Not a TP implementation. The audited per-forward collectives are bf16 SUM all-reduces of [T, 1024]
(1 embedding + 2 per layer x 28 layers = 57) and one logits all-gather of [rows, 75968] -> [rows, 151936]. TBCCL has no bf16 reduction
datatype yet, so this probe uses the SAME shapes in float32 (2x the bytes) over the existing, tested reduction path, and also the byte-exact
bf16 payload through all_gather (byte-generic) -- enough to bound the communication cost of TP before any datatype work.

Two ranks, plain torch.distributed + torch_tbccl, env rendezvous (MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE):
  rank 0 (Linux): --device cuda     rank 1 (Mac): --device cpu
Prints one JSON line per case: median/p25/p75/min microseconds.
"""
import argparse
import json
import os
import statistics as st
import time

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401  (registers the 'tbccl' backend)

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--hidden", type=int, default=1024)
p.add_argument("--tokens", default="1,4,12,128,512")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--vocab-shard", type=int, default=75968)
a = p.parse_args()
rank = int(os.environ["RANK"])
dist.init_process_group("tbccl", rank=rank, world_size=2)
dev = torch.device(a.device)


def timed(fn):
    for _ in range(a.warmup):
        fn()
    xs = []
    for _ in range(a.iters):
        dist.barrier()
        t = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1e6)
    xs.sort()
    return {"median_us": round(st.median(xs), 1), "p25_us": round(xs[len(xs) // 4], 1), "p75_us": round(xs[3 * len(xs) // 4], 1), "min_us": round(xs[0], 1)}


for T in [int(x) for x in a.tokens.split(",")]:
    x = torch.ones(T, a.hidden, dtype=torch.float32, device=dev)
    r = timed(lambda: dist.all_reduce(x))
    if rank == 0:
        print(json.dumps({"op": "all_reduce_sum", "dtype": "float32", "shape": [T, a.hidden], "bytes": x.nbytes, **r}), flush=True)
    g = [torch.empty(T, a.hidden, dtype=torch.bfloat16, device=dev) for _ in range(2)]
    xb = torch.ones(T, a.hidden, dtype=torch.bfloat16, device=dev)
    r = timed(lambda: dist.all_gather(g, xb))
    if rank == 0:
        print(json.dumps({"op": "all_gather", "dtype": "bfloat16", "shape": [T, a.hidden], "bytes": xb.nbytes, **r}), flush=True)

for rows in (1, 12):
    lg = torch.ones(rows, a.vocab_shard, dtype=torch.bfloat16, device=dev)
    out = [torch.empty_like(lg) for _ in range(2)]
    r = timed(lambda: dist.all_gather(out, lg))
    if rank == 0:
        print(json.dumps({"op": "logits_all_gather", "dtype": "bfloat16", "shape": [rows, a.vocab_shard], "bytes": lg.nbytes, **r}), flush=True)
dist.barrier()
dist.destroy_process_group()
