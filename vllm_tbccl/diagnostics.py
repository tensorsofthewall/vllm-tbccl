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
        flush = len(_events) % 16 == 0
    if flush:  # worker processes are often terminated without running atexit; keep the file current
        dump()


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


def start_pg_trace(period_s: float = 1.0) -> None:
    """Record every ProcessGroupTBCCL operation (torch_tbccl trace) and keep <VLLM_TBCCL_TRACE_FILE>.pg.<pid>.json current
    from a daemon thread (no I/O on the communication path). Covers vLLM's generic send/recv path as well as ours."""
    path = os.environ.get("VLLM_TBCCL_TRACE_FILE")
    if not enabled() or not path:
        return
    import torch_tbccl

    torch_tbccl.trace_set_enabled(True)

    def loop():
        last = -1
        while True:
            time.sleep(period_s)
            ev = torch_tbccl.trace_events()
            if len(ev) != last:
                last = len(ev)
                with open(f"{path}.pg.{os.getpid()}.json.tmp", "w") as fh:
                    json.dump(ev, fh)
                os.replace(f"{path}.pg.{os.getpid()}.json.tmp", f"{path}.pg.{os.getpid()}.json")

    threading.Thread(target=loop, daemon=True, name="vllm-tbccl-trace").start()
