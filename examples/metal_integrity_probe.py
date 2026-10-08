"""Bit-exact integrity probe of the CUDA <-> Metal stage boundary (real vllm-tbccl on both sides, no model).

    CUDA host : VLLM_TBCCL_ENABLE=1 TBCCL_LOCAL_ENDPOINT=<ip>:0 python metal_integrity_probe.py --role cuda  --rank R --init tcp://<rank0 ip>:<port>
    Mac       : VLLM_TBCCL_ENABLE=1 VLLM_TBCCL_BACKEND=metal VLLM_TBCCL_ARCHITECTURE=Qwen3ForCausalLM TBCCL_LOCAL_ENDPOINT=<ip>:0 python metal_integrity_probe.py --role metal --rank 1-R ...

The CUDA rank is a real vLLM GroupCoordinator (``send_tensor_dict`` / ``recv_tensor_dict`` through TBCCLDeviceCommunicator); the Metal rank is the real
``TBCCLMetalPipelineTransport`` (zero-copy MLX alias, boundary codec). Payloads are exact integer patterns (representable in bf16) built on the sending device:
  CUDA -> Metal: {hidden_states = A, residual = B}; Metal must receive x = A + B bit for bit (the codec's add is exact for |A|,|B| <= 125),
  Metal -> CUDA: x; CUDA must receive hidden_states == x and residual == 0 bit for bit.
Sizes are the pipeline's real shapes (decode 1 row, prefill 3..512 rows, hidden 1024, bf16), with random compute-like gaps. Any altered element is a failure.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist

from vllm_tbccl._backend import register_backend

register_backend()  # the bundled "tbccl" c10d backend
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform

p = argparse.ArgumentParser()
p.add_argument("--role", choices=["cuda", "metal"], required=True)
p.add_argument("--rank", type=int, required=True)
p.add_argument("--init", required=True)
p.add_argument("--hidden", type=int, default=1024)
p.add_argument("--tokens", default="1,1,1,3,7,64,166,512")
p.add_argument("--iters", type=int, default=20)
p.add_argument("--jitter-ms", type=float, default=5.0)
p.add_argument("--out", default=None)
p.add_argument("--mode", choices=["integrity", "peer_exit_recv", "peer_exit_send"], default="integrity",
               help="peer_exit_*: the cuda-role (upstream) process exits abruptly; the metal-role transport must raise a communicator error within --bound seconds")
p.add_argument("--bound", type=float, default=20.0)
a = p.parse_args()
rank, peer = a.rank, 1 - a.rank
rng = random.Random(91 + rank)


def gap():
    if a.jitter_ms > 0:
        time.sleep(rng.random() * a.jitter_ms / 1e3)


def ipattern_torch(tokens, seed, k, device):
    n = tokens * a.hidden
    return ((torch.arange(n, dtype=torch.int64, device=device) * 31 + seed * 17 + k * 101) % 251 - 125).reshape(tokens, a.hidden)


def ipattern_mx(tokens, seed, k):
    import mlx.core as mx

    n = tokens * a.hidden
    return ((mx.arange(n, dtype=mx.int32) * 31 + seed * 17 + k * 101) % 251 - 125).reshape(tokens, a.hidden)


stats = {"cuda_to_metal": 0, "cuda_to_metal_bad": 0, "metal_to_cuda": 0, "metal_to_cuda_bad": 0, "elements_bad": 0, "bytes": 0}
first_bad = None

with set_current_vllm_config(VllmConfig()):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=a.init, local_rank=0,
                                    backend=os.environ.get("VLLM_METAL_DIST_BACKEND", "tbccl") if a.role == "metal" else current_platform.dist_backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp = ps.get_pp_group()
    print(f"rank {rank}: role={a.role} platform={type(current_platform).__name__} device_group={dist.get_backend(pp.device_group)}", flush=True)
    if a.role == "metal":
        import mlx.core as mx

        from vllm_tbccl.backends.metal import TBCCLMetalPipelineTransport

        tr = TBCCLMetalPipelineTransport(rank=rank, size=2)
        assert tr.peer_kind in ("cuda", "cpu"), tr.peer_kind   # an upstream vLLM peer (cpu only in the local loopback validation)
    dev = pp.device
    if a.mode != "integrity":
        if a.role == "cuda":
            time.sleep(3)
            print("exiting abruptly", flush=True)
            os._exit(0)
        t0 = time.monotonic()
        err = None
        try:
            if a.mode == "peer_exit_recv":
                tr.recv((64, a.hidden), mx.bfloat16, peer)
            else:
                big = mx.zeros((16384, a.hidden), dtype=mx.bfloat16)  # 32 MB: cannot complete into a peer that never reads
                mx.eval(big)
                for _ in range(8):
                    tr.send(big, peer)
        except Exception as e:  # noqa: BLE001
            err = e
        dt = time.monotonic() - t0
        print(f"rank {rank} {a.mode}: error after {dt:.2f}s: {err!r}", flush=True)
        assert err is not None, "the operation completed although the peer is gone"
        assert "communicator failure" in str(err) or "peer" in str(err).lower(), err
        assert dt < a.bound, dt
        print(f"rank {rank} ok", flush=True)
        os._exit(0)
    for tokens in (int(t) for t in a.tokens.split(",")):
        for it in range(a.iters):
            for sender in (0, 1):
                seed = 1000 * it + tokens + sender * 5
                gap()
                if a.role == "cuda":
                    if rank == sender:  # CUDA -> Metal
                        pp.send_tensor_dict({"hidden_states": ipattern_torch(tokens, seed, 0, dev).to(torch.bfloat16),
                                             "residual": ipattern_torch(tokens, seed, 1, dev).to(torch.bfloat16)})
                    else:  # Metal -> CUDA
                        got = pp.recv_tensor_dict()
                        want = ipattern_torch(tokens, seed, 0, dev).to(torch.bfloat16)
                        bad = int((got["hidden_states"].view(torch.int16) != want.view(torch.int16)).sum().item()) + int((got["residual"].view(torch.int16) != 0).sum().item())
                        bad += int(got["hidden_states"].shape != want.shape)
                        stats["metal_to_cuda"] += 1
                        stats["bytes"] += want.nbytes * 2
                        if bad:
                            stats["metal_to_cuda_bad"] += 1
                            stats["elements_bad"] += bad
                            first_bad = first_bad or ("metal_to_cuda", tokens, it, sender)
                else:
                    if rank == sender:  # Metal -> CUDA
                        x = ipattern_mx(tokens, seed, 0).astype(mx.bfloat16)
                        mx.eval(x)
                        tr.send(x, peer)
                    else:  # CUDA -> Metal
                        x = tr.recv((tokens, a.hidden), mx.bfloat16, peer)
                        want = (ipattern_mx(tokens, seed, 0) + ipattern_mx(tokens, seed, 1)).astype(mx.bfloat16)
                        bad = int(mx.sum(x.view(mx.uint16) != want.view(mx.uint16)).item())
                        stats["cuda_to_metal"] += 1
                        stats["bytes"] += want.nbytes * 2
                        if bad:
                            stats["cuda_to_metal_bad"] += 1
                            stats["elements_bad"] += bad
                            first_bad = first_bad or ("cuda_to_metal", tokens, it, sender)
    ev = []  # the bundled backend keeps no per-operation timeline
    res = {"rank": rank, "role": a.role, "stats": stats, "first_bad": first_bad, "tbccl_trace_events": len(ev)}
    print("RESULT", json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(f"{a.out}.rank{rank}.json", "w"))
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    bad = stats["cuda_to_metal_bad"] + stats["metal_to_cuda_bad"]
    print(f"rank {rank} {'ok' if bad == 0 else 'MISMATCH'}", flush=True)
    raise SystemExit(1 if bad else 0)
