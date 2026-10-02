"""vLLM platform plugin: the CUDA / CPU platform the host would pick anyway, with exactly two behavioral differences:
the distributed backend is "tbccl" and the device communicator class is TBCCLDeviceCommunicator.

Selected through vLLM's public `vllm.platform_plugins` entry point; active only when VLLM_TBCCL_ENABLE=1 (so installing the
package changes nothing by default). Nothing from the base platforms (workers, attention backends, kernels) is copied.
VLLM_TBCCL_PLATFORM=cpu|cuda forces the base platform (default: cuda when a CUDA device exists, else cpu).
"""
import os


def _base_kind() -> str:
    forced = os.environ.get("VLLM_TBCCL_PLATFORM")
    if forced in ("cpu", "cuda"):
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def tbccl_platform_plugin() -> str | None:
    if os.environ.get("VLLM_TBCCL_ENABLE", "0") in ("", "0"):
        return None
    import torch_tbccl  # noqa: F401  (registers the "tbccl" c10d backend)
    from . import diagnostics

    diagnostics.start_pg_trace()

    return "vllm_tbccl.platform.TbcclCudaPlatform" if _base_kind() == "cuda" else "vllm_tbccl.platform.TbcclCpuPlatform"


_COMM = "vllm_tbccl.communicator.TBCCLDeviceCommunicator"


def __getattr__(name):
    # Define the platform classes lazily: importing vLLM's platform modules at plugin-discovery time would itself try to
    # resolve current_platform.
    if name == "TbcclCudaPlatform":
        from vllm.platforms.cuda import CudaPlatform

        class TbcclCudaPlatform(CudaPlatform):
            dist_backend = "tbccl"

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
            def get_device_communicator_cls(cls) -> str:
                return _COMM

        TbcclCpuPlatform.__module__ = __name__
        globals()[name] = TbcclCpuPlatform
        return TbcclCpuPlatform
    raise AttributeError(name)
