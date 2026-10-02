"""Local two-process test of TBCCLMetalPipelineTransport (run on the Mac; needs mlx + vllm-metal).

rank 0 plays an UPSTREAM vLLM stage (CPU platform + TBCCLDeviceCommunicator, vLLM's tensor-dict wire); rank 1 is the Metal stage using the
transport. Checks, exactly: upstream -> metal (hidden_states + residual summed, rows permuted into the Metal packed order) and
metal -> upstream (hidden_states = x in upstream row order, residual = 0). Also a raw Metal<->Metal exchange when --mode raw (both ranks Metal).
"""
import argparse
import os
import sys
import types

import torch

import torch_tbccl  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
p.add_argument("--mode", default="upstream", choices=["upstream", "raw"])
a = p.parse_args()
rank = a.rank
H = 576

with set_current_vllm_config(VllmConfig()):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=a.init, local_rank=0, backend="tbccl")
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp = ps.get_pp_group()
    metal_side = rank == 1 or a.mode == "raw"
    if metal_side:
        import mlx.core as mx
        from vllm_tbccl.backends.metal import TBCCLMetalPipelineTransport

        tr = TBCCLMetalPipelineTransport(rank=rank, size=2, peer_ips=None)
        print(f"rank {rank}: peer_kind={tr.peer_kind}", flush=True)

    # request layout used by the test: upstream order = [new: A(3 tok), B(2 tok)] + [cached: C(1), D(1)]  -> 7 rows
    sched = types.SimpleNamespace(
        scheduled_new_reqs=[types.SimpleNamespace(req_id="A"), types.SimpleNamespace(req_id="B")],
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=["C", "D"]),
        num_scheduled_tokens={"A": 3, "B": 2, "C": 1, "D": 1})
    metal_packed = [("C", 1), ("D", 1), ("A", 3), ("B", 2)]            # vllm-metal: [decode rows][prefill rows]
    n = 7

    def rows(seed):   # row r of the upstream order carries the value (seed + r) in every column (exact in bf16 for small ints)
        return (torch.arange(n, dtype=torch.float32).unsqueeze(1) + seed).expand(n, H).contiguous().to(torch.bfloat16)

    if a.mode == "upstream":
        if rank == 0:       # upstream stage: send {hidden_states, residual}, then receive the Metal stage's reply
            hs, rs = rows(10), rows(100)
            pp.send_tensor_dict({"hidden_states": hs, "residual": rs}, dst=1)
            got = pp.recv_tensor_dict(src=1)
            assert set(got) == {"hidden_states", "residual"}
            # Metal sent x (in Metal order -> mapped back to upstream order) + zeros; upstream sum must equal x_up_order
            x_up = (got["hidden_states"].float() + got["residual"].float())
            expected = (rows(10).float() + rows(100).float()) * 2          # the Metal side doubles x before replying
            assert torch.equal(x_up, expected), (x_up[:, 0], expected[:, 0])
            print("rank 0 ok", flush=True)
        else:
            tr.begin_step(metal_packed, sched)
            x = tr.recv((n, H), mx.bfloat16, 0)                           # Metal packed order
            mx.eval(x)
            xt = torch.from_dlpack(x) if False else None
            from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
            got = mlx_to_torch(x, device="cpu").float()
            up_val = (torch.arange(n, dtype=torch.float32) + 10) + (torch.arange(n, dtype=torch.float32) + 100)   # per upstream row
            order = [0 + i for i in (5, 6)] + [0, 1, 2, 3, 4]               # metal order of upstream rows: C(5),D(6),A(0..2),B(3..4)
            order = [5, 6, 0, 1, 2, 3, 4]
            assert torch.equal(got[:, 0], up_val[order]), (got[:, 0], up_val[order])
            tr.send(x * 2, 0)                                              # reply in Metal order; transport maps back to upstream order
            print("rank 1 ok", flush=True)
    else:                   # raw Metal <-> Metal
        if rank == 0:
            x = mx.arange(n * H, dtype=mx.float32).reshape(n, H).astype(mx.bfloat16)
            tr.send(x, 1)
            y = tr.recv((n, H), mx.bfloat16, 1)
            mx.eval(y)
            assert mx.array_equal(y, x * 2).item()
            print("rank 0 ok", flush=True)
        else:
            y = tr.recv((n, H), mx.bfloat16, 0)
            tr.send(y * 2, 0)
            print("rank 1 ok", flush=True)
    sys.stdout.flush()
    os._exit(0)
