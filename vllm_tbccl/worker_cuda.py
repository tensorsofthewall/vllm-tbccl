"""CUDA worker for heterogeneous pipelines (see ``worker.py``): reports KV-cache layouts compatible with a CPU pipeline peer. Imports only the CUDA worker module (the CPU
worker module has import-time side effects that must not run in a CUDA process)."""
from vllm.v1.worker.gpu_worker import Worker

from .worker import _HeteroLayouts


class TbcclGpuWorker(_HeteroLayouts, Worker):
    _tbccl_kind = "cuda"
