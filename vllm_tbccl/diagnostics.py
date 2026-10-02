"""Opt-in (VLLM_TBCCL_TRACE=1) per-operation record of what vLLM actually sends through the TBCCL device group.

Durations are same-process monotonic clocks only; never compare stamps across hosts. Dump with dump() or at exit when
VLLM_TBCCL_TRACE_FILE is set (one JSON file per process, suffixed with the pid).
"""
import atexit
import json
import os
import threading
import time

_events = []
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("VLLM_TBCCL_TRACE", "0") not in ("", "0")


def record(op: str, nbytes: int, shape, dtype, t_entry_ns: int, t_done_ns: int, **extra) -> None:
    if not enabled():
        return
    with _lock:
        _events.append({"op": op, "bytes": int(nbytes), "shape": list(shape), "dtype": str(dtype),
                        "entry_ns": t_entry_ns, "done_ns": t_done_ns, **extra})


def now_ns() -> int:
    return time.monotonic_ns()


def events() -> list:
    with _lock:
        return list(_events)


def dump(path: str | None = None) -> None:
    path = path or os.environ.get("VLLM_TBCCL_TRACE_FILE")
    if not path:
        return
    with open(f"{path}.{os.getpid()}.json", "w") as fh:
        json.dump(events(), fh)


atexit.register(dump)
