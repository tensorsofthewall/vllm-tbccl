"""Silent-peer test through vLLM's real PP GroupCoordinator over vllm-tbccl (no model).

Rank 1 completes one healthy exchange, then stays alive and silent. Rank 0 blocks in pp_group.recv_tensor_dict(); a control thread
aborts the c10d process groups after --abort-after s. Expect: recv raises promptly, teardown is bounded, both processes exit.
"""
import argparse
import datetime
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state as ps
from vllm.platforms import current_platform

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
p.add_argument("--abort-after", type=float, default=1.0)
a = p.parse_args()
rank = a.rank

with set_current_vllm_config(VllmConfig()):
    ps.init_distributed_environment(world_size=2, rank=rank, distributed_init_method=a.init, local_rank=0,
                                    backend=current_platform.dist_backend)
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    pp = ps.get_pp_group()
    store = dist.distributed_c10d._get_default_store()
    x = {"h": torch.full((64, 16), float(rank + 1)).to(pp.device)}
    if rank == 0:
        pp.send_tensor_dict(x)
        got = pp.recv_tensor_dict()
        assert float(got["h"].cpu()[0, 0]) == 2.0
    else:
        got = pp.recv_tensor_dict()
        assert float(got["h"].cpu()[0, 0]) == 1.0
        pp.send_tensor_dict(x)
    print(f"rank {rank}: healthy exchange done", flush=True)
    if rank == 1:
        store.wait(["release"], datetime.timedelta(seconds=120))  # alive and silent
        store.set("r1_done", "1")
        print("rank 1 ok", flush=True)
        sys.stdout.flush()
        os._exit(0)

    def control():
        time.sleep(a.abort_after)
        t = time.monotonic()
        dist.distributed_c10d._abort_process_group()
        print(f"rank 0: abort returned in {(time.monotonic() - t) * 1e3:.1f} ms", flush=True)

    th = threading.Thread(target=control)
    th.start()
    t0 = time.monotonic()
    try:
        pp.recv_tensor_dict()
        print("rank 0: recv returned (UNEXPECTED)", flush=True)
        rc = 1
    except Exception as e:  # noqa: BLE001
        print(f"rank 0: blocked recv raised after {time.monotonic() - t0:.2f}s: {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
        rc = 0
    th.join()
    store.set("release", "1")
    store.wait(["r1_done"], datetime.timedelta(seconds=120))
    print("rank 0 ok" if rc == 0 else "rank 0 FAILED", flush=True)
    sys.stdout.flush()
    os._exit(rc)
