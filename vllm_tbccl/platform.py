"""vLLM platform plugin: the CUDA / CPU platform the host would pick anyway, with exactly two behavioral differences:
the distributed backend is "tbccl" and the device communicator class is TBCCLDeviceCommunicator.

Selected through vLLM's public `vllm.platform_plugins` entry point; active only when VLLM_TBCCL_ENABLE=1 (so installing the
package changes nothing by default). Nothing from the base platforms (workers, attention backends, kernels) is copied.
VLLM_TBCCL_PLATFORM=cpu|cuda forces the base platform (default: cuda when a CUDA device exists, else cpu).
"""
import os
import sys

# vLLM versions this package was validated against (see docs/phase69_results.md). Other versions are used at the user's risk and warn once.
SUPPORTED_VLLM = ("0.31.0",)


def _check_vllm_version() -> None:
    import warnings

    import vllm

    if vllm.__version__.split("+")[0] not in SUPPORTED_VLLM:
        warnings.warn(f"vllm-tbccl was validated with vLLM {SUPPORTED_VLLM}; found {vllm.__version__}", RuntimeWarning, stacklevel=2)


def install_cpu_group_backend() -> None:
    """vLLM (0.31.0) creates every GroupCoordinator control group with the hard-coded backend "gloo" (``new_group(ranks, backend="gloo")`` in
    ``vllm.distributed.parallel_state``), and gloo does not rendezvous between the Linux and macOS PyTorch builds. There is no public vLLM setting for
    that backend, so this wrapper (installed from the platform plugin, no vLLM source modified) substitutes ``VLLM_CPU_GROUP_BACKEND`` (default
    "tbccl", i.e. torch-tbccl's ProcessGroup) for exactly those calls: the caller must be vllm.distributed.parallel_state and the requested backend
    must be "gloo". Every other ``new_group`` call, in vLLM or elsewhere, is untouched. ``VLLM_CPU_GROUP_BACKEND=gloo`` disables it."""
    import torch.distributed as dist

    backend = os.environ.get("VLLM_CPU_GROUP_BACKEND", "tbccl")
    if backend == "gloo" or getattr(dist.new_group, "_vllm_tbccl_wrapped", False):
        return
    real = dist.new_group

    def new_group(*args, **kwargs):
        if kwargs.get("backend") == "gloo" and sys._getframe(1).f_globals.get("__name__") == "vllm.distributed.parallel_state":
            kwargs["backend"] = backend
        return real(*args, **kwargs)

    new_group._vllm_tbccl_wrapped = True
    new_group.__wrapped__ = real
    dist.new_group = new_group


def _base_kind() -> str:
    forced = os.environ.get("VLLM_TBCCL_PLATFORM")
    if forced in ("cpu", "cuda"):
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def tbccl_platform_plugin() -> str | None:
    if os.environ.get("VLLM_TBCCL_ENABLE", "0") in ("", "0"):
        return None
    from . import backends

    if backends.select_backend() == "metal":
        # vllm-metal owns MetalPlatform. Two out-of-tree platform plugins cannot both be active, so DECLINE here and only attach as a
        # transport provider through vllm-metal's generic hooks.
        import torch_tbccl  # noqa: F401

        backends.configure_metal_environment()
        from . import diagnostics

        diagnostics.start_pg_trace()
        return None
    import torch_tbccl  # noqa: F401  (registers the "tbccl" c10d backend)

    _check_vllm_version()
    # Control groups: gloo does not rendezvous between Linux and macOS PyTorch builds and vLLM hard-codes it (see install_cpu_group_backend).
    os.environ.setdefault("VLLM_CPU_GROUP_BACKEND", "tbccl")
    install_cpu_group_backend()
    from . import diagnostics

    diagnostics.start_pg_trace()

    return "vllm_tbccl.platform.TbcclCudaPlatform" if _base_kind() == "cuda" else "vllm_tbccl.platform.TbcclCpuPlatform"


_COMM = "vllm_tbccl.communicator.TBCCLDeviceCommunicator"


def _use_tbccl_worker(vllm_config, default_cls: str, tbccl_cls: str) -> None:
    """Swap the platform's default worker for its thin TBCCL subclass (heterogeneous KV-layout agreement, CPU PP buffer); a user-chosen worker class is left alone."""
    pc = vllm_config.parallel_config
    if pc.worker_cls == default_cls and os.environ.get("VLLM_TBCCL_WORKER", "1") not in ("", "0"):
        pc.worker_cls = tbccl_cls


def __getattr__(name):
    # Define the platform classes lazily: importing vLLM's platform modules at plugin-discovery time would itself try to
    # resolve current_platform.
    if name == "TbcclCudaPlatform":
        from vllm.platforms.cuda import CudaPlatform

        class TbcclCudaPlatform(CudaPlatform):
            dist_backend = "tbccl"

            @classmethod
            def check_and_update_config(cls, vllm_config) -> None:
                super().check_and_update_config(vllm_config)
                _use_tbccl_worker(vllm_config, "vllm.v1.worker.gpu_worker.Worker", "vllm_tbccl.worker_cuda.TbcclGpuWorker")

            @classmethod
            def get_device_communicator_cls(cls) -> str:
                return _COMM

        TbcclCudaPlatform.__module__ = __name__
        globals()[name] = TbcclCudaPlatform
        return TbcclCudaPlatform
    if name == "TbcclCpuPlatform":
        from vllm.platforms.cpu import CpuPlatform

        class TbcclCpuPlatform(CpuPlatform):
            dist_backend = "tbccl"

            @classmethod
            def check_and_update_config(cls, vllm_config) -> None:
                super().check_and_update_config(vllm_config)
                _use_tbccl_worker(vllm_config, "vllm.v1.worker.cpu_worker.CPUWorker", "vllm_tbccl.worker_cpu.TbcclCpuWorker")

            @classmethod
            def get_device_communicator_cls(cls) -> str:
                return _COMM

        TbcclCpuPlatform.__module__ = __name__
        globals()[name] = TbcclCpuPlatform
        return TbcclCpuPlatform
    raise AttributeError(name)
