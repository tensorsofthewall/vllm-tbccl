"""Phase 69: GroupCoordinator-level lifecycle / failure probe through vllm-tbccl (two local processes, no model).

  --mode cycles          repeat (init groups -> exchange -> destroy) N times in the same processes: no leaked threads / file descriptors / RSS growth
  --mode peer_exit_recv  rank 1 exits right after a healthy exchange; rank 0 blocks in recv_tensor_dict(): it must fail with an error within --bound s
  --mode peer_exit_send  rank 1 exits without receiving; rank 0 sends a large payload (isend_tensor_dict) and waits the handles: error within --bound s, no hang
  --mode destroy_idle    healthy exchange, then destroy with nothing pending: returns promptly, nothing left behind
"""
import argparse
import os
import sys
import threading
import time

import psutil
import torch
import torch.distributed as dist

import torch_tbccl  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
p.add_argument("--mode", required=True)
p.add_argument("--cycles", type=int, default=6)
p.add_argument("--bound", type=float, default=60.0)
a = p.parse_args()
rank = a.rank
host, port = a.init[len("tcp://"):].rsplit(":", 1)


def init(cycle=0):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=f"tcp://{host}:{int(port) + cycle * 3}", local_rank=0, backend=current_platform.dist_backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    return ps.get_pp_group()


def exchange(pp, n=64):
    x = {"hidden_states": torch.full((n, 128), float(rank + 1), dtype=torch.bfloat16).to(pp.device)}
    if rank == 0:
        pp.send_tensor_dict(x)
        g = pp.recv_tensor_dict()
        assert float(g["hidden_states"].float().cpu()[0, 0]) == 2.0
    else:
        g = pp.recv_tensor_dict()
        assert float(g["hidden_states"].float().cpu()[0, 0]) == 1.0
        pp.send_tensor_dict(x)


with set_current_vllm_config(VllmConfig()):
    proc = psutil.Process()
    if a.mode == "cycles":
        pp = init(0)
        exchange(pp)
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        time.sleep(0.5)
        base = (proc.num_fds(), threading.active_count(), proc.num_threads(), proc.memory_info().rss)
        for c in range(1, a.cycles + 1):
            pp = init(c)
            exchange(pp)
            ps.destroy_model_parallel()
            ps.destroy_distributed_environment()
        time.sleep(1.0)
        now = (proc.num_fds(), threading.active_count(), proc.num_threads(), proc.memory_info().rss)
        d = (now[0] - base[0], now[1] - base[1], now[2] - base[2], (now[3] - base[3]) / 1e6)
        print(f"rank {rank} cycles={a.cycles} fd_delta={d[0]} py_thread_delta={d[1]} os_thread_delta={d[2]} rss_delta_mb={d[3]:.1f}", flush=True)
        assert d[0] <= 2 and d[1] <= 1 and d[2] <= 2 and d[3] < 150, d
        print(f"rank {rank} ok", flush=True)
        raise SystemExit(0)
    pp = init(0)
    exchange(pp)
    if a.mode == "destroy_idle":
        t0 = time.monotonic()
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        dt = time.monotonic() - t0
        print(f"rank {rank} destroy {dt:.2f}s", flush=True)
        assert dt < 10
        print(f"rank {rank} ok", flush=True)
        raise SystemExit(0)
    if rank == 1:
        time.sleep(0.5)
        print("rank 1 exiting abruptly", flush=True)
        os._exit(0)
    t0 = time.monotonic()
    err = None
    try:
        if a.mode == "peer_exit_recv":
            pp.recv_tensor_dict()
        else:  # peer_exit_send: a payload far larger than the socket buffers, so the send cannot complete without the peer
            big = {"hidden_states": torch.zeros(64 * 1024 * 1024 // 2, dtype=torch.bfloat16).to(pp.device)}
            for h in pp.isend_tensor_dict(big, dst=1):
                h.wait()
    except Exception as e:  # noqa: BLE001
        err = e
    dt = time.monotonic() - t0
    print(f"rank 0 {a.mode}: raised={type(err).__name__ if err else None} after {dt:.2f}s: {str(err)[:160]!r}", flush=True)
    assert err is not None, "the operation completed although the peer is gone"
    assert dt < a.bound, dt
    print("rank 0 ok", flush=True)
    sys.stdout.flush()
    os._exit(0)
