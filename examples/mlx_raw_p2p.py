"""Raw P2P between a CUDA torch tensor (Linux) and an MLX Metal array (Mac) through the MLX->torch zero-copy CPU alias and
ProcessGroupTBCCL. Rank roles: the rank whose platform has CUDA uses torch CUDA tensors; the other rank uses MLX arrays.

Run one process per host with the usual MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE and TBCCL_LOCAL_ENDPOINT=<ip>:0.
Checks (all exact): Linux->Mac recv into an MLX array then a Metal op consumes it; Mac->Linux MLX-computed array sent from its alias.
"""
import argparse
import datetime
import json
import os
import sys
import time

import torch
import torch.distributed as dist

from vllm_tbccl._backend import register_backend

register_backend()  # the bundled "tbccl" c10d backend

p = argparse.ArgumentParser()
p.add_argument("--sizes", default="2304,25344,1048576", help="bytes")
p.add_argument("--dtypes", default="float32,float16")
a = p.parse_args()

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=60))
rank = dist.get_rank()
is_cuda = torch.cuda.is_available() and os.environ.get("RAW_P2P_SIDE", "cuda") == "cuda"
TD = {"float32": torch.float32, "float16": torch.float16}
ok = True
rows = []

if not is_cuda:
    import mlx.core as mx
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
    MD = {"float32": mx.float32, "float16": mx.float16}


def ref(n, dt, seed):
    return ((torch.arange(n, dtype=torch.int64) * 31 + seed) % 997).to(TD[dt])


for dt in a.dtypes.split(","):
    for nbytes in (int(s) for s in a.sizes.split(",")):
        n = nbytes // (4 if dt == "float32" else 2)
        # --- Linux CUDA -> Mac MLX ------------------------------------------------------------------------------
        if is_cuda:
            x = ref(n, dt, 1).cuda()
            t0 = time.monotonic_ns(); dist.send(x, dst=1 - rank); t1 = time.monotonic_ns()
            rows.append({"dir": "cuda->mlx", "dtype": dt, "bytes": nbytes, "send_ms": (t1 - t0) / 1e6})
        else:
            m = mx.zeros((n,), dtype=MD[dt]); mx.eval(m)
            alias = mlx_to_torch(m, device="cpu")
            t0 = time.monotonic_ns(); dist.recv(alias, src=1 - rank); t1 = time.monotonic_ns()
            out = (m * 2 + 1)                       # Metal consumes the SAME array (no copy, no sync beyond eval)
            mx.eval(out)
            got = mlx_to_torch(out, device="cpu")
            exp = ref(n, dt, 1).to(torch.float32) * 2 + 1
            good = torch.equal(got.to(torch.float32), exp) if dt == "float32" else bool((got.to(torch.float32) - exp).abs().max() == 0)
            ok &= good
            rows.append({"dir": "cuda->mlx", "dtype": dt, "bytes": nbytes, "recv_ms": (t1 - t0) / 1e6, "exact": good})
        # --- Mac MLX -> Linux CUDA --------------------------------------------------------------------------------
        if is_cuda:
            y = torch.zeros(n, dtype=TD[dt], device="cuda")
            t0 = time.monotonic_ns(); dist.recv(y, src=1 - rank); t1 = time.monotonic_ns()
            good = torch.equal(y.cpu(), ref(n, dt, 7))
            ok &= good
            rows.append({"dir": "mlx->cuda", "dtype": dt, "bytes": nbytes, "recv_ms": (t1 - t0) / 1e6, "exact": good})
        else:
            idx = mx.arange(n, dtype=mx.int32)
            v = (((idx * 31 + 7) % 997)).astype(MD[dt])   # produced by a Metal computation
            mx.eval(v)
            alias = mlx_to_torch(v, device="cpu")
            t0 = time.monotonic_ns(); dist.send(alias, dst=1 - rank); t1 = time.monotonic_ns()
            rows.append({"dir": "mlx->cuda", "dtype": dt, "bytes": nbytes, "send_ms": (t1 - t0) / 1e6})

# async send with references retained, then dropped after wait
if not is_cuda:
    m = mx.arange(4096, dtype=mx.int32).astype(mx.float32); mx.eval(m)
    w = dist.isend(mlx_to_torch(m, device="cpu"), dst=1 - rank)
    del m                      # user-facing reference dropped; Work retains the alias tensor (and thus the MLX buffer)
    w.wait()
else:
    z = torch.zeros(4096, device="cuda"); dist.recv(z, src=1 - rank)
    ok &= torch.equal(z.cpu(), torch.arange(4096, dtype=torch.float32))

print(f"rank {rank} ({'cuda' if is_cuda else 'mlx'}): {'ALL EXACT' if ok else 'MISMATCH'}", flush=True)
for r in rows:
    print("  " + json.dumps(r), flush=True)
dist.destroy_process_group()
sys.exit(0 if ok else 1)
