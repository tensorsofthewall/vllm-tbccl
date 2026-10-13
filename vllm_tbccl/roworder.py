"""Explicit pipeline row order between a Metal stage and an upstream vLLM stage.

A pipeline activation holds one row per scheduled token. Each stage orders those rows differently: upstream vLLM follows its persistent input
batch (slots are reused and compacted, chunked-prefill continuations keep their slot), vllm-metal packs ``[decode rows][prefill rows]``. The
two orders cannot be derived from the scheduler output both stages receive, so the SENDER states the actual order of the rows it sends, as
``[(request_id, n_rows), ...]`` (size proportional to the number of active requests, never to the activation), and the RECEIVER computes the
permutation from the sender's order to the order its own stage consumes. Any disagreement about the active requests is an error: no permutation
is ever guessed.

This module holds the pure helpers (no torch/vllm imports) and the small per-process state the upstream worker and communicator share.
"""
import json
import os
from collections.abc import Sequence

import numpy as np

# Key of the non-tensor entry carried in vLLM's tensor-dict metadata list (only on a pipeline whose peer is a Metal stage).
ROW_ORDER_KEY = "__vllm_tbccl_row_order__"

RowOrder = Sequence[tuple[str, int]]


class RowOrderError(RuntimeError):
    pass


def normalize(order, what: str = "row order") -> list[tuple[str, int]]:
    """Validate ``[(request_id, n_rows), ...]``: string ids, unique, positive integer row counts."""
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    try:
        items = [(rid, n) for rid, n in order]
    except (TypeError, ValueError) as e:
        raise RowOrderError(f"vllm-tbccl: malformed {what}: {order!r}") from e
    for rid, n in items:
        if not isinstance(rid, str) or isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
            raise RowOrderError(f"vllm-tbccl: malformed {what} entry {(rid, n)!r} (need (request id, positive row count))")
        if rid in seen:
            raise RowOrderError(f"vllm-tbccl: duplicate request {rid!r} in the {what}")
        seen.add(rid)
        out.append((rid, int(n)))
    return out


def total_rows(order: RowOrder) -> int:
    return sum(n for _, n in order)


def row_permutation(source: RowOrder, dest: RowOrder) -> np.ndarray | None:
    """Row gather that turns activation rows in ``source`` order into ``dest`` order: ``out = rows[perm]`` (``out[m] = rows[perm[m]]``).

    Returns None for identical orders (nothing to move). Raises RowOrderError unless both sides name exactly the same requests with the same
    row counts; nothing is dropped, inserted or guessed.
    """
    src = normalize(source, "source row order")
    dst = normalize(dest, "destination row order")
    if total_rows(src) != total_rows(dst):
        raise RowOrderError(f"vllm-tbccl: the stages disagree on the number of rows ({total_rows(src)} sent, {total_rows(dst)} expected)")
    s, d = dict(src), dict(dst)
    if s.keys() != d.keys():
        raise RowOrderError(
            "vllm-tbccl: the stages disagree on the active requests: "
            f"only in the sent order {sorted(s.keys() - d.keys())}, only in the expected order {sorted(d.keys() - s.keys())}")
    for rid in s:
        if s[rid] != d[rid]:
            raise RowOrderError(f"vllm-tbccl: the stages disagree on the rows of request {rid!r} ({s[rid]} sent, {d[rid]} expected)")
    if src == dst:
        return None
    offset, start = {}, 0
    for rid, n in src:
        offset[rid] = start
        start += n
    return np.concatenate([np.arange(offset[rid], offset[rid] + n) for rid, n in dst]).astype(np.int32)


_trace_path = None


def trace(event: str, **fields) -> None:
    """Diagnostic only (VLLM_TBCCL_TRACE_ROWS=<file>): one JSON line per row-order event. Off by default."""
    global _trace_path
    if _trace_path is None:
        _trace_path = os.environ.get("VLLM_TBCCL_TRACE_ROWS", "")
    if not _trace_path:
        return
    with open(f"{_trace_path}.{os.getpid()}", "a") as f:
        f.write(json.dumps({"event": event, **fields}) + "\n")


class UpstreamRowState:
    """Per-process row-order state of an upstream (CUDA/CPU) stage; shared by the worker and the device communicator.

    ``begin_step`` is called with the scheduler output before the step; ``local_order`` reads the runner's persistent input batch (valid once the
    runner has updated its states, i.e. while the model runs and when intermediate tensors are consumed). A peer's order received with
    the activation waits in ``peer_order`` until ``take_receive_permutation`` consumes it.
    """

    def __init__(self) -> None:
        self.runner = None
        self.scheduler_output = None
        self.peer_order: list[tuple[str, int]] | None = None

    def begin_step(self, runner, scheduler_output) -> None:
        if self.peer_order is not None:
            raise RowOrderError("vllm-tbccl: a received row order was never consumed by the previous step")
        self.runner, self.scheduler_output = runner, scheduler_output

    def local_order(self) -> list[tuple[str, int]]:
        if self.runner is None or self.scheduler_output is None:
            raise RowOrderError(
                "vllm-tbccl: the worker did not register the step's row order; a Metal pipeline peer needs the vllm-tbccl workers "
                "(parallel_config.worker_cls is replaced?)")
        batch = self.runner.input_batch
        counts = self.scheduler_output.num_scheduled_tokens
        ids = list(batch.req_ids[: batch.num_reqs])
        missing = [r for r in ids if r not in counts]
        if missing:
            raise RowOrderError(f"vllm-tbccl: requests {missing} are in the input batch but were not scheduled this step")
        return normalize([(r, int(counts[r])) for r in ids], "local row order")

    def take_receive_permutation(self) -> np.ndarray | None:
        """Permutation of the received rows into this stage's input-batch order (None = identity)."""
        peer, self.peer_order = self.peer_order, None
        if peer is None:
            return None
        local = self.local_order()
        perm = row_permutation(peer, local)
        trace("recv_permute", peer_order=peer, local_order=local, perm=None if perm is None else perm.tolist())
        return perm


STATE = UpstreamRowState()
