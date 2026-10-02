"""Peer-kind discovery through the c10d default Store (non-blocking publish, bounded read): no collective is needed, so ranks that build
their groups at different times (a CUDA/CPU communicator at group construction, a Metal transport later) can never deadlock each other."""
import datetime

import torch.distributed as dist


def _store():
    return dist.distributed_c10d._get_default_store()


def publish_kind(global_rank: int, kind: str) -> None:
    _store().set(f"vllm_tbccl/kind/{global_rank}", kind)


def peer_kind(peer_global_rank: int, timeout_s: float = 300.0) -> str:
    key = f"vllm_tbccl/kind/{peer_global_rank}"
    _store().wait([key], datetime.timedelta(seconds=timeout_s))
    return _store().get(key).decode()
