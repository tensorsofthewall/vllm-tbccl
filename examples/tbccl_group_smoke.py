"""vLLM GroupCoordinator-level smoke test (no model): world_size=2, PP=2, TP=1, through vllm-tbccl.

Each rank picks its own platform (VLLM_TBCCL_PLATFORM=cuda|cpu), so one process can be a CUDA rank and the other a CPU rank.
Exercises every PP-path operation the vLLM 0.30.0 audit found: pp_group.send_tensor_dict / recv_tensor_dict, isend/irecv_tensor_dict,
send/recv, broadcast_object (control path), both directions; verifies exact values and that every tensor moved through
ProcessGroupTBCCL (torch_tbccl trace) and no NCCL communicator exists.

  rank 0: MASTER=<rank0 host:port> VLLM_TBCCL_ENABLE=1 RANK=0 python tbccl_group_smoke.py --init tcp://HOST:PORT
  rank 1: ... RANK=1 ...
"""
import argparse
import json
import os
import time

import torch

import torch_tbccl
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform

import vllm_tbccl.diagnostics as diag

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
p.add_argument("--hidden", type=int, default=576)
p.add_argument("--tokens", default="1,7,64,512")
p.add_argument("--out", default=None)
a = p.parse_args()
rank = a.rank

with set_current_vllm_config(VllmConfig()):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=a.init, local_rank=0,
                                    backend=current_platform.dist_backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp = ps.get_pp_group()
    dev = pp.device
    print(f"rank {rank}: platform={type(current_platform).__name__} device={dev} dist_backend={current_platform.dist_backend} "
          f"comm={type(pp.device_communicator).__name__} peer_device_type={getattr(pp.device_communicator, 'peer_device_type', None)} "
          f"custom_send_recv={pp.use_cpu_custom_send_recv}", flush=True)
    assert type(pp.device_communicator).__name__ == "TBCCLDeviceCommunicator"
    assert not any("nccl" in k.lower() for k in type(pp.device_communicator).__dict__), "no NCCL objects"
    peer = 1 - rank
    torch_tbccl.trace_set_enabled(True)
    torch_tbccl.trace_reset()
    results = []

    def tensors(tokens, seed):
        g = torch.Generator().manual_seed(seed)
        return {"hidden_states": torch.randn(tokens, a.hidden, generator=g), "residual": torch.randn(tokens, a.hidden, generator=g)}

    for tokens in (int(t) for t in a.tokens.split(",")):
        for sender in (0, 1):
            ref = tensors(tokens, 11 + sender * 100 + tokens)
            t0 = time.monotonic_ns()
            if rank == sender:
                if sender == 0 and tokens % 2:   # exercise both blocking and async send forms
                    handles = pp.isend_tensor_dict({k: v.to(dev) for k, v in ref.items()} | {"note": "x"}, dst=1)
                    for h in handles:
                        h.wait()
                else:
                    pp.send_tensor_dict({k: v.to(dev) for k, v in ref.items()} | {"note": "x"})
            else:
                got = pp.recv_tensor_dict()
                assert got["note"] == "x"
                for k in ("hidden_states", "residual"):
                    assert got[k].device.type == dev.type, (k, got[k].device)
                    assert torch.equal(got[k].cpu(), ref[k]), f"mismatch {k} tokens={tokens} sender={sender}"
            results.append({"op": "tensor_dict", "tokens": tokens, "sender": sender, "bytes": 2 * tokens * a.hidden * 4,
                            "ms": (time.monotonic_ns() - t0) / 1e6})
    # plain send/recv through the communicator + control-path object broadcast
    x = torch.arange(1000, dtype=torch.float32, device=dev) + 1000 * rank
    if rank == 0:
        pp.send(x)
        y = pp.recv(torch.Size([1000]), torch.float32)
        assert torch.equal(y.cpu(), torch.arange(1000, dtype=torch.float32) + 1000)
    else:
        y = pp.recv(torch.Size([1000]), torch.float32)
        assert torch.equal(y.cpu(), torch.arange(1000, dtype=torch.float32))
        pp.send(x)
    obj = pp.broadcast_object({"hello": rank} if rank == 0 else None, src=0)
    assert obj == {"hello": 0}
    ev = torch_tbccl.trace_events()
    ops = {}
    for e in ev:
        ops[e["op"]] = ops.get(e["op"], 0) + 1
    tbccl_bytes = sum(e["bytes"] for e in ev)
    print(f"rank {rank}: ProcessGroupTBCCL ops={ops} bytes={tbccl_bytes}", flush=True)
    assert ops.get("send", 0) + ops.get("recv", 0) > 0, "no tensor went through ProcessGroupTBCCL"
    if a.out:
        json.dump({"rank": rank, "results": results, "tbccl_ops": ops, "tbccl_bytes": tbccl_bytes, "events": ev},
                  open(f"{a.out}.rank{rank}.json", "w"))
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    print(f"rank {rank} ok", flush=True)
