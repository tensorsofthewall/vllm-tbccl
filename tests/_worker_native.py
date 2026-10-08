"""Worker for tests/test_native_backend.py: one rank of a local group over vllm-tbccl's own "tbccl" backend (no vLLM, no torch-tbccl)."""
import argparse
import datetime
import gc
import importlib.util
import sys
import time

import torch
import torch.distributed as dist

from vllm_tbccl._backend import register_backend


def init(rank, world, port, timeout_s=60):
    dist.init_process_group("tbccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=timeout_s))


def case_basic(rank, world):
    t = torch.full((1024,), float(rank + 1))
    dist.all_reduce(t)
    assert torch.all(t == sum(range(1, world + 1))), t[:4]
    b = torch.arange(10, dtype=torch.int64) if rank == 0 else torch.zeros(10, dtype=torch.int64)
    dist.broadcast(b, src=0)
    assert b.tolist() == list(range(10))
    outs = [torch.empty(5, dtype=torch.float32) for _ in range(world)]
    dist.all_gather(outs, torch.full((5,), float(rank)))
    assert [o[0].item() for o in outs] == [float(r) for r in range(world)]
    dist.barrier()
    assert dist.get_backend() == "tbccl"


def case_p2p(rank, world):
    # byte-generic dtypes, repeated, both directions
    for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.uint8, torch.bool):
        n = 4096
        if rank == 0:
            src = (torch.arange(n) % 7).to(dtype)
            dist.send(src, dst=1)
            back = torch.empty(n, dtype=dtype)
            dist.recv(back, src=1)
            assert torch.equal(back, src), dtype
        else:
            buf = torch.empty(n, dtype=dtype)
            dist.recv(buf, src=0)
            dist.send(buf, dst=0)
    # zero-element tensors complete without touching the transport
    z = torch.empty(0)
    (dist.send(z, dst=1) if rank == 0 else dist.recv(z, src=0))


def case_lifetime(rank, world):
    # the Work and the tensor reference are dropped immediately; TBCCL must still complete the transfer into/out of the (kept alive) storage
    if rank == 0:
        for i in range(8):
            w = dist.isend(torch.full((1 << 16,), float(i)), dst=1)
            del w
            gc.collect()
        dist.barrier()
    else:
        outs = []
        for i in range(8):
            t = torch.empty(1 << 16)
            dist.irecv(t, src=0).wait()
            outs.append(t)
        dist.barrier()
        assert [o[0].item() for o in outs] == [float(i) for i in range(8)]


def case_error(rank, world):
    # a posted recv on a silent peer fails (not hangs) once the group is aborted from this side, and later operations raise
    dist.barrier()
    if rank == 0:
        t = torch.empty(16)
        work = dist.irecv(t, src=1)
        time.sleep(0.5)
        pg = dist.distributed_c10d._get_default_group()
        pg._get_backend(torch.device("cpu")).abort()
        try:
            work.wait()
        except Exception as e:  # noqa: BLE001
            assert "vllm-tbccl" in str(e), e
        else:
            raise AssertionError("recv on an aborted group completed")
        try:
            dist.send(torch.zeros(4), dst=1)
        except Exception as e:  # noqa: BLE001
            assert "vllm-tbccl" in str(e), e
        else:
            raise AssertionError("send on an aborted group succeeded")
    else:
        time.sleep(3)  # stay silent
    print(f"rank {rank} error case ok", flush=True)


def case_reinit(rank, world):
    for i in range(3):
        t = torch.full((8,), float(rank + i))
        dist.all_reduce(t)
        assert t[0].item() == sum(r + i for r in range(world))
        dist.barrier()
        if i < 2:
            dist.destroy_process_group()
            init(rank, world, PORTS[1 + i])


def case_subgroup(rank, world):
    # vLLM-style: a world group plus a 2-rank subgroup, all on the same backend
    g = dist.new_group(ranks=[0, 1], backend="tbccl")
    if rank in (0, 1):
        t = torch.full((16,), float(rank + 1))
        dist.all_reduce(t, group=g)
        assert t[0].item() == 3.0
    dist.barrier()


def case_independence(rank, world):
    assert importlib.util.find_spec("torch_tbccl") is None or "torch_tbccl" not in sys.modules
    assert "torch_tbccl" not in sys.modules


CASES = {k[5:]: v for k, v in globals().items() if k.startswith("case_")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--ports", required=True, help="comma-separated free TCPStore ports, one per initialization")
    ARGS = ap.parse_args()
    PORTS = [int(p) for p in ARGS.ports.split(",")]
    register_backend()
    init(ARGS.rank, ARGS.world, PORTS[0])
    CASES[ARGS.case](ARGS.rank, ARGS.world)
    if ARGS.case != "error":
        dist.barrier()
    dist.destroy_process_group()
    print(f"rank {ARGS.rank} {ARGS.case} ok", flush=True)
