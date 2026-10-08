"""Opt-in (VLLM_TBCCL_TRACE=1) per-operation record of what vLLM actually sends through the TBCCL device group.

Durations are same-process monotonic clocks only; never compare stamps across hosts. Events are only appended in memory on the
communication path; a daemon thread writes the file (one JSON per process, suffixed with the pid) every VLLM_TBCCL_TRACE_PERIOD_S
(default 5) seconds when VLLM_TBCCL_TRACE_FILE is set, and once more at exit. There is no per-event I/O, and payload checksums are
only taken by the separate VLLM_TBCCL_CHECKSUM debug mode.
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
    _ensure_dumper()


def now_ns() -> int:
    return time.monotonic_ns()


def events() -> list:
    with _lock:
        return list(_events)


def dump(path: str | None = None) -> None:
    path = path or (os.path.abspath(os.environ["VLLM_TBCCL_TRACE_FILE"]) if os.environ.get("VLLM_TBCCL_TRACE_FILE") else None)
    if not path:
        return
    with open(f"{path}.{os.getpid()}.json", "w") as fh:
        json.dump(events(), fh)


atexit.register(dump)

_dumper = None


def _ensure_dumper() -> None:
    """Worker processes are often terminated without running atexit, so a daemon thread keeps the file reasonably current."""
    global _dumper
    if _dumper is not None or not os.environ.get("VLLM_TBCCL_TRACE_FILE"):
        return
    _dumper = True
    period = float(os.environ.get("VLLM_TBCCL_TRACE_PERIOD_S", "5"))

    def loop():
        last = -1
        while True:
            time.sleep(period)
            n = len(_events)
            if n != last:
                last = n
                dump()

    threading.Thread(target=loop, daemon=True, name="vllm-tbccl-trace-dump").start()


_started = False


def start_pg_trace(period_s: float = 1.0) -> None:
    """Record every ProcessGroupTBCCL operation and keep <VLLM_TBCCL_TRACE_FILE>.pg.<pid>.json current
    from a daemon thread (no I/O on the communication path). Covers vLLM's generic send/recv path as well as ours."""
    global _started
    path = (os.path.abspath(os.environ["VLLM_TBCCL_TRACE_FILE"]) if os.environ.get("VLLM_TBCCL_TRACE_FILE") else None)
    if _started or not enabled() or not path:
        return
    _started = True
    from ._backend import mode

    if mode() != "torch-tbccl":
        return  # the native backend keeps no per-operation timeline; it comes from the optional torch-tbccl compatibility mode
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
