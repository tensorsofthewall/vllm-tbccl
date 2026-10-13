"""Shared logic of the thin worker subclasses (``worker_cuda.py`` / ``worker_cpu.py``) for heterogeneous (CUDA + CPU) pipeline stages, installed through vLLM's public ``parallel_config.worker_cls`` extension point by the
TBCCL platforms (see ``platform.py``). They replace the two compatibility patches that vLLM 0.30.0 needed (the vLLM pipeline-parallel integration work) without modifying vLLM:

* KV-cache layout agreement. ``resolve_kv_cache_layout`` (engine core) asserts that every worker reports the *same* list of supported layouts. A CUDA FLASH_ATTN worker
  declares no restriction (the default preference list) while the CPU attention backend supports only ``LBHNC``, so a CUDA + CPU pair disagrees. A TBCCL worker whose
  pipeline peer is a different device kind reports its own list restricted to what that peer kind supports (``VLLM_TBCCL_KV_LAYOUTS=A,B`` overrides explicitly).
* PP receive buffer on CPU. ``GPUModelRunner.sync_and_gather_intermediate_tensors`` asserts the persistent ``intermediate_tensors`` buffer, normally created by the
  warm-up dummy run, which a CPU worker in eager mode never runs; the CPU subclass creates it where the runner would have (after the model is loaded).

Nothing here touches scheduling, placement, KV-cache policy, model code or sampling.
"""
import os

import torch
from vllm.distributed import get_pp_group

from . import roworder

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

    def execute_model(self, scheduler_output):
        # Row order toward a Metal pipeline peer (roworder.py): record the step so the communicator can state this stage's actual row order
        # and the received rows can be put into this stage's input-batch order.
        roworder.STATE.begin_step(self.model_runner, scheduler_output)
        install_row_order_hook(self.model_runner)
        install_group_row_order_hooks(get_pp_group())
        return super().execute_model(scheduler_output)  # type: ignore[misc]


def install_row_order_hook(runner) -> None:
    """Apply the received row permutation where the runner gathers the received intermediate tensors: that is after the runner has updated its
    persistent input batch (the order the rows must end up in), which the communicator cannot know when it receives."""
    if getattr(runner, "_tbccl_rows_hooked", False):
        return
    orig = runner.sync_and_gather_intermediate_tensors

    def sync_and_gather(num_tokens, intermediate_tensors, sync_self):
        perm = roworder.STATE.take_receive_permutation()
        if perm is not None and intermediate_tensors is not None:
            index = torch.as_tensor(perm, dtype=torch.long)
            for key in list(intermediate_tensors.tensors):
                t = intermediate_tensors.tensors[key]
                intermediate_tensors.tensors[key] = t.index_select(0, index.to(t.device))
        return orig(num_tokens, intermediate_tensors, sync_self)

    runner.sync_and_gather_intermediate_tensors = sync_and_gather
    runner._tbccl_rows_hooked = True


def install_group_row_order_hooks(pp) -> None:
    """The CUDA platform sends and receives tensor dicts through vLLM's generic GroupCoordinator path (a CPU worker goes through the device
    communicator instead, which carries the row order itself). For a Metal peer, add this stage's row order to the dict it sends (vLLM carries
    non-tensor entries in the metadata list, so the peer's wire format is unchanged) and take the peer's order off the dict it receives."""
    if getattr(pp, "_tbccl_rows_hooked", False) or pp.world_size != 2:
        return
    pp._tbccl_rows_hooked = True
    if getattr(pp, "use_cpu_custom_send_recv", False):
        return
    if getattr(pp.device_communicator, "peer_device_type", None) != "metal":
        return
    orig_isend, orig_irecv = pp.isend_tensor_dict, pp.irecv_tensor_dict

    def isend_tensor_dict(tensor_dict, *args, **kwargs):
        order = roworder.STATE.local_order()
        roworder.trace("send_order", order=order)
        return orig_isend({**tensor_dict, roworder.ROW_ORDER_KEY: order}, *args, **kwargs)

    def irecv_tensor_dict(*args, **kwargs):
        tensor_dict, handles, postprocess = orig_irecv(*args, **kwargs)
        if tensor_dict is None:
            return tensor_dict, handles, postprocess
        order = tensor_dict.pop(roworder.ROW_ORDER_KEY, None)
        if order is None:
            raise roworder.RowOrderError(
                "vllm-tbccl: the Metal pipeline peer did not send its row order (a vllm-tbccl without the row-order protocol); "
                "both stages must run the same vllm-tbccl version")
        roworder.STATE.peer_order = roworder.normalize(order, "received row order")
        return tensor_dict, handles, postprocess

    pp.isend_tensor_dict, pp.irecv_tensor_dict = isend_tensor_dict, irecv_tensor_dict
