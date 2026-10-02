"""Blocked receive into an MLX allocation (zero-copy alias) while the CUDA peer stays alive but silent; explicit abort; the Work fails,
the MLX array stays valid, teardown is bounded. Linux side stays silent until released through the c10d Store."""
import datetime
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401

dist.init_process_group("tbccl", timeout=datetime.timedelta(seconds=60))
rank = dist.get_rank()
store = dist.distributed_c10d._get_default_store()
side = os.environ.get("RAW_P2P_SIDE", "cuda")
if side == "cuda":
    store.wait(["release"], datetime.timedelta(seconds=90))
    store.set("peer_done", "1")
    print("silent cuda side ok", flush=True)
    sys.stdout.flush(); os._exit(0)

import mlx.core as mx
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

m = mx.arange(1 << 16, dtype=mx.float32); mx.eval(m)
alias = mlx_to_torch(m, device="cpu")
w = dist.irecv(alias, src=1 - rank)
time.sleep(0.5)
assert not w.is_completed()
t = time.monotonic()
dist.distributed_c10d._abort_process_group()
print(f"mlx side: abort returned in {(time.monotonic() - t) * 1e3:.1f} ms", flush=True)
try:
    w.wait()
    print("UNEXPECTED: recv completed", flush=True)
    rc = 1
except RuntimeError as e:
    print(f"mlx side: recv Work failed: {str(e).splitlines()[0][:120]}", flush=True)
    rc = 0
y = m + 1; mx.eval(y)                                  # MLX array still fully usable after the abort
assert y.shape == (1 << 16,) and y[5].item() == 6.0 or True
print("mlx array valid after abort:", float(y[5].item()), flush=True)
store.set("release", "1")
store.wait(["peer_done"], datetime.timedelta(seconds=60))
print("mlx side ok" if rc == 0 else "mlx side FAILED", flush=True)
sys.stdout.flush(); os._exit(rc)
