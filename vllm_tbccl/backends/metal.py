"""TBCCLMetalPipelineTransport: vllm-metal's pipeline-parallel activation transport over torch.distributed / ProcessGroupTBCCL.

    mx.array (evaluated)  ->  zero-copy CPU torch alias (DLPack, vllm-metal's own bridge)  ->  dist.send/recv on the PP device group
                          ->  ProcessGroupTBCCL  ->  libtbccl

No NumPy staging, no bytes copy, no mx.distributed ring. The wire format is exactly the ring transport's: one row-major
``(n_tokens, hidden)`` array per boundary in the stage compute dtype. Both calls complete before returning (the pipeline boundary needs
the activation anyway); ``recv`` returns the very MLX array whose storage TBCCL wrote into. Imports of mlx / vllm_metal are lazy so
importing vllm_tbccl on Linux never needs them.
"""
import os

import torch
import torch.distributed as dist

from .. import diagnostics as diag
from .. import roworder
from ..peer import peer_kind, publish_kind
from ..wire import recv_object, send_object
from .codecs import codec_for


def _csum(t) -> int | None:
    """Opt-in (VLLM_TBCCL_CHECKSUM=1) byte-sum of a CPU alias, to prove sender/receiver bytes are identical (diagnostic only)."""
    if os.environ.get("VLLM_TBCCL_CHECKSUM", "0") in ("", "0"):
        return None
    return int(t.view(torch.uint8).to(torch.int64).sum().item())


class TBCCLMetalPipelineTransport:
    def __init__(self, rank: int, size: int, peer_ips=None) -> None:
        from vllm.distributed import get_pp_group

        pp = get_pp_group()
        if pp.world_size != size or pp.rank_in_group != rank:
            raise RuntimeError(
                f"vllm-tbccl: PP group (rank {pp.rank_in_group}/{pp.world_size}) does not match pipeline stages ({rank}/{size})")
        backend = dist.get_backend(pp.device_group)
        if backend != "tbccl":
            raise RuntimeError(
                f"vllm-tbccl: the PP device group must use the 'tbccl' backend (got {backend!r}); set VLLM_METAL_DIST_BACKEND=tbccl")
        self.rank, self.size = rank, size
        self._group = pp.device_group
        self._cpu_group = pp.cpu_group
        self._ranks = pp.ranks
        # Peer discovery through the c10d Store (no collective): an upstream vLLM peer publishes its device type when its
        # TBCCLDeviceCommunicator is built; a vllm-metal peer's transport publishes "metal".
        publish_kind(pp.ranks[rank], "metal")
        self.peer_kind = peer_kind(pp.ranks[1 - rank])
        self._zeros = {}
        install_step_timing()
        self._packed = None        # this step's actual row order [(request_id, n_rows), ...] (set by begin_step)
        self._codec = None
        # Receive-buffer pool (VLLM_TBCCL_RECV_POOL=1, off by default): decode-sized upstream receives reuse one MLX buffer per
        # (tensor, shape, dtype) instead of paying mx.zeros + mx.eval per tensor per step (~1.3 ms measured in vLLM
        # pipeline-parallel study).
        self._pool_on = os.environ.get("VLLM_TBCCL_RECV_POOL", "0") not in ("", "0")
        self._pool_max_rows = int(os.environ.get("VLLM_TBCCL_RECV_POOL_MAX_ROWS", "64"))
        self._pool = {}

    def _recv_buffer(self, key, shape, dtype):
        """A zero-initialized MLX buffer to receive into. Pooled buffers are only valid because _recv_upstream evaluates the
        add that consumes both of them before it returns, so a buffer is never read after the next receive overwrites it."""
        import mlx.core as mx

        if self._pool_on and shape[0] <= self._pool_max_rows:
            buf = self._pool.get((key, shape, str(dtype)))
            if buf is None:
                buf = mx.zeros(shape, dtype=dtype)
                mx.eval(buf)
                self._pool[(key, shape, str(dtype))] = buf
            return buf
        buf = mx.zeros(shape, dtype=dtype)
        mx.eval(buf)
        return buf

    @property
    def codec(self):
        if self._codec is None:
            arch = os.environ.get("VLLM_TBCCL_ARCHITECTURE")
            if not arch:
                raise RuntimeError(
                    "vllm-tbccl: the model architecture is unknown to the Metal pipeline transport; start with `vllm serve <local model dir>` "
                    "or set VLLM_TBCCL_ARCHITECTURE (e.g. Qwen3ForCausalLM)")
            self._codec = codec_for([arch])
        return self._codec

    def _alias(self, x):
        from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

        t = mlx_to_torch(x, device="cpu")
        if not t.is_contiguous():
            raise ValueError("vllm-tbccl: pipeline activations must be row-major contiguous (no hidden copy is made)")
        return t

    def send(self, x, dst: int):
        if self.peer_kind != "metal":
            return self._send_upstream(x, dst)
        return self._send_raw(x, dst)

    def recv(self, shape, dtype, src: int):
        if self.peer_kind != "metal":
            return self._recv_upstream(shape, dtype, src)
        return self._recv_raw(shape, dtype, src)

    def _send_raw(self, x, dst: int):
        import mlx.core as mx

        t_entry = diag.now_ns()
        mx.eval(x)                                   # the MLX producer must be finished: DLPack export needs an evaluated array
        t_eval = diag.now_ns()
        alias = self._alias(x)
        t_alias = diag.now_ns()
        cs = _csum(alias)
        dist.send(alias, dst=self._ranks[dst], group=self._group)   # blocking: x/alias stay referenced until TBCCL is done
        t_done = diag.now_ns()
        diag.record("metal_send", alias.nbytes, alias.shape, alias.dtype, t_entry, t_done, csum=cs,
                    eval_us=(t_eval - t_entry) / 1e3, alias_us=(t_alias - t_eval) / 1e3, pg_us=(t_done - t_alias) / 1e3)
        return mx.array(0)                           # completed handle: the caller's mx.eval is a no-op

    def _recv_raw(self, shape, dtype, src: int):
        import mlx.core as mx

        t_entry = diag.now_ns()
        out = mx.zeros(shape, dtype=dtype)
        mx.eval(out)                                 # real, evaluated allocation to receive into
        t_alloc = diag.now_ns()
        alias = self._alias(out)
        t_alias = diag.now_ns()
        dist.recv(alias, src=self._ranks[src], group=self._group)
        t_done = diag.now_ns()
        cs = _csum(alias)
        diag.record("metal_recv", alias.nbytes, alias.shape, alias.dtype, t_entry, t_done, csum=cs,
                    alloc_us=(t_alloc - t_entry) / 1e3, alias_us=(t_alias - t_alloc) / 1e3, pg_us=(t_done - t_alias) / 1e3)
        return out                                   # the SAME array whose storage was written; the next Metal stage consumes it

    def close(self) -> None:
        return None

    # ---- boundary codec for an UPSTREAM vLLM peer (CUDA or CPU); the per-architecture algebra lives in codecs.py ---------------
    # Upstream non-last stage output: IntermediateTensors{hidden_states, residual} where the true residual stream is
    # x = hidden_states + residual (the next layer's fused add+RMSNorm computes exactly that sum). vllm-metal's stage boundary is x itself.
    #   upstream -> metal:  x = hidden_states + residual            (one bf16 add on Metal; identical to what the next upstream layer does)
    #   metal -> upstream:  hidden_states = x, residual = 0         (the upstream layer then computes x + 0 = x, then normalizes)
    # Wire format toward/from the upstream peer is vLLM's own tensor-dict protocol, so the upstream side runs unmodified.

    def _torch_dtype(self, mdt):
        from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE

        return MLX_TO_TORCH_DTYPE[mdt]

    def _send_upstream(self, x, dst: int):
        import mlx.core as mx
        from vllm.distributed.parallel_state import TensorMetadata

        t_entry = diag.now_ns()
        order = self._step_order()               # the rows go out in this stage's packed order; the receiver permutes
        mx.eval(x)
        t_eval = diag.now_ns()
        shape = tuple(x.shape)
        tdt = self._torch_dtype(x.dtype)
        key = (shape, str(x.dtype))
        zeros = self._zeros.get(key)
        if zeros is None:
            zeros = mx.zeros(shape, dtype=x.dtype)
            mx.eval(zeros)
            self._zeros[key] = zeros
        parts = self.codec.from_metal(x, zeros)
        meta = [(k, TensorMetadata(self.peer_kind, tdt, torch.Size(shape))) for k in self.codec.keys]
        meta.append((roworder.ROW_ORDER_KEY, order))
        roworder.trace("send_order", order=order)
        g = self._ranks[dst]
        send_object(meta, g, self._cpu_group)
        xa, za = (self._alias(parts[k]) for k in self.codec.keys)
        t_alias = diag.now_ns()
        cs = _csum(xa)
        dist.send(xa, dst=g, group=self._group)
        dist.send(za, dst=g, group=self._group)
        t_done = diag.now_ns()
        diag.record("metal_send_upstream", xa.nbytes + za.nbytes, shape, tdt, t_entry, t_done, csum=cs,
                    eval_us=(t_eval - t_entry) / 1e3, alias_us=(t_alias - t_eval) / 1e3, pg_us=(t_done - t_alias) / 1e3)
        return mx.array(0)

    def _recv_upstream(self, shape, dtype, src: int):
        import mlx.core as mx

        t_entry = diag.now_ns()
        g = self._ranks[src]
        meta = recv_object(g, self._cpu_group)
        source = [v for k, v in meta if k == roworder.ROW_ORDER_KEY]
        if len(source) != 1:
            raise roworder.RowOrderError(
                "vllm-tbccl: the upstream pipeline peer did not send its row order (a vllm-tbccl without the row-order protocol); "
                "both stages must run the same vllm-tbccl version")
        source = roworder.normalize(source[0], "received row order")
        meta = [(k, v) for k, v in meta if k != roworder.ROW_ORDER_KEY]
        names = [k for k, _ in meta]
        if sorted(names) != sorted(self.codec.keys):
            raise NotImplementedError(
                f"vllm-tbccl: the {self.codec.name} boundary codec expects IntermediateTensors {list(self.codec.keys)}; peer sent {names}")
        tdt = self._torch_dtype(dtype)
        got = {}
        alloc_ns = 0
        t_meta = diag.now_ns()
        for key, md in meta:
            if tuple(md.size) != tuple(shape) or md.dtype != tdt:
                raise RuntimeError(
                    f"vllm-tbccl: boundary mismatch for {key}: peer sent {tuple(md.size)} {md.dtype}, stage expects {tuple(shape)} {tdt}")
            t_a = diag.now_ns()
            m = self._recv_buffer(key, tuple(shape), dtype)
            alias = self._alias(m)
            alloc_ns += diag.now_ns() - t_a
            dist.recv(alias, src=g, group=self._group)
            got[key] = m
        t_recv = diag.now_ns()
        cs = _csum(self._alias(got["hidden_states"]))
        x = self.codec.to_metal(got)
        dest = self._step_order()
        perm = roworder.row_permutation(source, dest)
        roworder.trace("recv_permute", peer_order=source, local_order=dest, perm=None if perm is None else perm.tolist())
        if perm is not None:
            x = self._take(x, perm)             # peer's row order -> this stage's packed order
        mx.eval(x)
        t_done = diag.now_ns()
        diag.record("metal_recv_upstream", 2 * x.nbytes, shape, tdt, t_entry, t_done, csum=cs,
                    meta_us=(t_meta - t_entry) / 1e3, pg_us=(t_recv - t_meta) / 1e3, alloc_us=alloc_ns / 1e3, add_us=(t_done - t_recv) / 1e3)
        return x

    # ---- request-row order: upstream vLLM vs vllm-metal ---------------------------------------------------------------------
    # See roworder.py. This stage's rows are [decode rows][prefill rows] (vllm-metal's packing); begin_step() records that order from the
    # runner. The upstream peer's order is whatever its persistent input batch holds and travels with its activation; it is never derived
    # from the scheduler output.

    def begin_step(self, packed, scheduler_output) -> None:
        self._packed = roworder.normalize(packed, "Metal packed row order") if self.peer_kind != "metal" else None

    def _step_order(self):
        """This step's order, consumed by the one boundary crossing of the step (a stale order is never reused)."""
        order, self._packed = self._packed, None
        if order is None:
            raise roworder.RowOrderError("vllm-tbccl: a pipeline activation crossed the stage boundary before begin_step() set this step's row order")
        return order

    def _take(self, x, idx):
        import mlx.core as mx

        return mx.take(x, mx.array(idx), axis=0)


def install_step_timing() -> None:
    """Diagnostic only (VLLM_TBCCL_TRACE_STEPS=1): time MetalModelRunner's forward-build (includes the blocking PP recv) and sampling
    (waits for the GPU) phases per step, same-process clock. Not part of the integration's behavior."""
    if os.environ.get("VLLM_TBCCL_TRACE_STEPS", "0") in ("", "0"):
        return
    from vllm_metal.v1.model_runner import MetalModelRunner

    if getattr(MetalModelRunner, "_tbccl_timed", False):
        return
    for name, label in (("_start_paged_forward", "step_forward"), ("_sample_tokens", "step_sample")):
        orig = getattr(MetalModelRunner, name)

        def make(orig=orig, label=label):
            def wrapped(self, *a, **k):
                t0 = diag.now_ns()
                try:
                    return orig(self, *a, **k)
                finally:
                    diag.record(label, 0, (0,), "n/a", t0, diag.now_ns())

            return wrapped

        setattr(MetalModelRunner, name, make())
    MetalModelRunner._tbccl_timed = True
