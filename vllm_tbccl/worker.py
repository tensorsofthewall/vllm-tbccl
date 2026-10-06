"""Shared logic of the thin worker subclasses (``worker_cuda.py`` / ``worker_cpu.py``) for heterogeneous (CUDA + CPU) pipeline stages, installed through vLLM's public ``parallel_config.worker_cls`` extension point by the
TBCCL platforms (see ``platform.py``). They replace the two compatibility patches that vLLM 0.30.0 needed (Phase 46) without modifying vLLM:

* KV-cache layout agreement. ``resolve_kv_cache_layout`` (engine core) asserts that every worker reports the *same* list of supported layouts. A CUDA FLASH_ATTN worker
  declares no restriction (the default preference list) while the CPU attention backend supports only ``LBHNC``, so a CUDA + CPU pair disagrees. A TBCCL worker whose
  pipeline peer is a different device kind reports its own list restricted to what that peer kind supports (``VLLM_TBCCL_KV_LAYOUTS=A,B`` overrides explicitly).
* PP receive buffer on CPU. ``GPUModelRunner.sync_and_gather_intermediate_tensors`` asserts the persistent ``intermediate_tensors`` buffer, normally created by the
  warm-up dummy run, which a CPU worker in eager mode never runs; the CPU subclass creates it where the runner would have (after the model is loaded).

Nothing here touches scheduling, placement, KV-cache policy, model code or sampling.
"""
import os

from vllm.distributed import get_pp_group

# Layouts a device kind can consume, for the kinds that restrict them (None = unrestricted). Sources: vLLM 0.31.0 CPU_ATTN.supported_kv_cache_layouts;
# vllm-metal MetalWorker.get_supported_kv_cache_layouts() == [KV_CACHE_LAYOUT] == ["LBNHC"] (vllm_metal/attention/caches/placement.py).
KIND_LAYOUTS = {"cpu": ("LBHNC",), "metal": ("LBNHC",)}


def restrict_layouts(own: list[str], peer_kind: str | None, own_kind: str, override: str | None = None) -> list[str]:
    """Layouts to report to the engine core: ``own`` (most preferred first) limited to what the peer's device kind supports."""
    if override:
        wanted = [x.strip() for x in override.split(",") if x.strip()]
        out = [x for x in wanted if x in own]
        if not out:
            raise ValueError(f"VLLM_TBCCL_KV_LAYOUTS={override!r} is not supported by this worker (supports {own})")
        return out
    if peer_kind is None or peer_kind == own_kind or peer_kind not in KIND_LAYOUTS:
        return own
    out = [x for x in own if x in KIND_LAYOUTS[peer_kind]]
    if not out:
        raise ValueError(f"no KV cache layout is supported by both this {own_kind} worker ({own}) and its {peer_kind} pipeline peer ({KIND_LAYOUTS[peer_kind]})")
    return out


class _HeteroLayouts:
    _tbccl_kind = "cuda"

    def get_supported_kv_cache_layouts(self) -> list[str]:
        own = super().get_supported_kv_cache_layouts()  # type: ignore[misc]
        override = os.environ.get("VLLM_TBCCL_KV_LAYOUTS")
        peer = None
        pp = get_pp_group()
        if pp.world_size == 2:
            peer = getattr(pp.device_communicator, "peer_device_type", None)
        return restrict_layouts(own, peer, self._tbccl_kind, override)
