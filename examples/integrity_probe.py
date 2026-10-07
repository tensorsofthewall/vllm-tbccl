"""Bit-exact integrity probe of the exact payloads vLLM's pipeline moves (GroupCoordinator level, real vllm-tbccl communicator, no model).

    rank 0: VLLM_TBCCL_ENABLE=1 VLLM_TBCCL_PLATFORM=cuda|cpu TBCCL_LOCAL_ENDPOINT=<ip>:0 python integrity_probe.py --init tcp://<rank0 ip>:<port> --rank 0
    rank 1: ... --rank 1

Patterns taken from the vLLM 0.31.0 runtime: IntermediateTensors dicts {hidden_states, residual} [num_tokens, hidden] in bf16 / fp16 / fp32 (decode = 1 token, prefill sizes, the 166-token
reference prompt), sent with `send_tensor_dict` / `isend_tensor_dict` and received with `recv_tensor_dict` (generic path on a CUDA rank, the communicator's path on a CPU rank, the
sender's device label differing from the receiver's), plus non-tensor entries, the sampled-token `broadcast` (int64, 8 B), `broadcast_object`, an int32 control all-reduce on the
control group, and small/empty/scalar tensors. Payloads are exact integer patterns (representable in bf16/fp16) produced on the sending device and compared on the receiving device
bit for bit (integer view), so any element altered in transit is caught. Random compute-like gaps separate the operations.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
p.add_argument("--hidden", type=int, default=1024)
p.add_argument("--tokens", default="1,1,1,3,7,64,166,512")
p.add_argument("--iters", type=int, default=20)
p.add_argument("--jitter-ms", type=float, default=5.0)
p.add_argument("--out", default=None)
a = p.parse_args()
rank, peer = a.rank, 1 - a.rank
rng = random.Random(77 + rank)

VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def pattern(tokens, dtype, seed, k, device):
    n = tokens * a.hidden
    v = ((torch.arange(n, dtype=torch.int64, device=device) * 31 + seed * 17 + k * 101) % 251 - 125)  # integers: exact in bf16 / fp16 / fp32
    return v.to(dtype).reshape(tokens, a.hidden)


def same(x, y):
    return int((x.view(VIEW[x.dtype]) != y.view(VIEW[y.dtype])).sum().item()) == 0 and x.shape == y.shape


def gap():
    if a.jitter_ms > 0:
        time.sleep(rng.random() * a.jitter_ms / 1e3)


with set_current_vllm_config(VllmConfig()):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=a.init, local_rank=0, backend=current_platform.dist_backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp = ps.get_pp_group()
    dev = pp.device
    print(f"rank {rank}: platform={type(current_platform).__name__} device={dev} cpu_group={dist.get_backend(pp.cpu_group)} device_group={dist.get_backend(pp.device_group)} "
          f"custom_send_recv={pp.use_cpu_custom_send_recv}", flush=True)
    assert dist.get_backend(pp.cpu_group) == "tbccl" and dist.get_backend(pp.device_group) == "tbccl"
    stats = {"tensor_dict": 0, "tensor_dict_bad": 0, "isend": 0, "broadcast": 0, "broadcast_bad": 0, "object_bad": 0, "allreduce_bad": 0, "bytes": 0}
    first_bad = None
    iters = 0
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens in (int(t) for t in a.tokens.split(",")):
            for it in range(a.iters):
                for sender in (0, 1):
                    seed = 1000 * it + tokens + (7 if dtype == torch.float16 else 13 if dtype == torch.float32 else 0)
                    ref_dev = {"hidden_states": pattern(tokens, dtype, seed + sender * 5, 0, dev), "residual": pattern(tokens, dtype, seed + sender * 5, 1, dev)}
                    gap()
                    if rank == sender:
                        d = dict(ref_dev) | {"note": f"it{it}", "num_tokens": tokens}
                        if it % 2 and sender == 0:
                            for h in pp.isend_tensor_dict(d, dst=1):
                                h.wait()
                            stats["isend"] += 1
                        else:
                            pp.send_tensor_dict(d)
                    else:
                        got = pp.recv_tensor_dict()
                        want = {"hidden_states": pattern(tokens, dtype, seed + sender * 5, 0, dev), "residual": pattern(tokens, dtype, seed + sender * 5, 1, dev)}
                        ok = got["note"] == f"it{it}" and got["num_tokens"] == tokens and all(got[k].device.type == dev.type and same(got[k], want[k]) for k in want)
                        stats["tensor_dict"] += 1
                        stats["bytes"] += sum(want[k].nbytes for k in want)
                        if not ok:
                            stats["tensor_dict_bad"] += 1
                            first_bad = first_bad or (str(dtype), tokens, it, sender)
                iters += 1
    # sampled-token broadcast (what the V2 runner does after sampling), int64 8 B and a 24 B / 16 B small vector, from each rank
    for src in (0, 1):
        for it in range(50):
            gap()
            t = torch.tensor([it * 7919 + src], dtype=torch.int64, device=dev) if rank == src else torch.zeros(1, dtype=torch.int64, device=dev)
            t = pp.broadcast(t, src=src)
            stats["broadcast"] += 1
            if int(t.item()) != it * 7919 + src:
                stats["broadcast_bad"] += 1
    obj = pp.broadcast_object({"hello": list(range(50)), "s": "x" * 300} if rank == 0 else None, src=0)
    stats["object_bad"] += 0 if obj == {"hello": list(range(50)), "s": "x" * 300} else 1
    c = torch.tensor([rank + 1, 10 * (rank + 1)], dtype=torch.int32)
    dist.all_reduce(c, group=pp.cpu_group)
    stats["allreduce_bad"] += 0 if c.tolist() == [3, 30] else 1
    # empty tensor + scalar tensors inside a dict
    if rank == 0:
        pp.send_tensor_dict({"empty": torch.empty(0, a.hidden, dtype=torch.bfloat16, device=dev), "scalar": torch.tensor(5, dtype=torch.int64, device=dev), "flag": True})
    else:
        g = pp.recv_tensor_dict()
        if not (g["empty"].numel() == 0 and int(g["scalar"].item()) == 5 and g["flag"] is True):
            stats["tensor_dict_bad"] += 1
    ev = torch_tbccl.trace_events() if hasattr(torch_tbccl, "trace_events") else []
    res = {"rank": rank, "device": str(dev), "stats": stats, "first_bad": first_bad, "tbccl_trace_events": len(ev)}
    print("RESULT", json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(f"{a.out}.rank{rank}.json", "w"))
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    bad = stats["tensor_dict_bad"] + stats["broadcast_bad"] + stats["object_bad"] + stats["allreduce_bad"]
    print(f"rank {rank} {'ok' if bad == 0 else 'MISMATCH'}", flush=True)
    raise SystemExit(1 if bad else 0)
