"""Backend selection for vllm-tbccl.

* ``torch``: upstream vLLM CPU/CUDA. vllm-tbccl owns the platform (platform plugin) and supplies the device communicator.
* ``metal``: vllm-metal owns the platform. vllm-tbccl DECLINES to be a platform plugin and only attaches as a pipeline transport
  provider (``VLLM_METAL_PP_TRANSPORT_CLS``) plus the c10d control-plane backend settings.

``VLLM_TBCCL_BACKEND=auto|torch|metal`` (default auto): auto -> torch when CUDA is present or vllm-metal/MLX is unavailable, metal on
macOS when vllm_metal is installed. Importing this module never imports mlx, CUDA modules or vllm_metal.
"""
import importlib.util
import os
import sys

BACKENDS = ("auto", "torch", "metal")
METAL_TRANSPORT_CLS = "vllm_tbccl.backends.metal.TBCCLMetalPipelineTransport"


def requested() -> str:
    value = os.environ.get("VLLM_TBCCL_BACKEND", "auto").lower()
    if value not in BACKENDS:
        raise ValueError(f"VLLM_TBCCL_BACKEND must be one of {BACKENDS}, got {value!r}")
    return value


def select_backend() -> str:
    value = requested()
    if value != "auto":
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "torch"
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "darwin" and importlib.util.find_spec("vllm_metal") is not None and importlib.util.find_spec("mlx") is not None:
        return "metal"
    return "torch"


def configure_metal_environment() -> None:
    """Point vllm-metal's generic hooks at vllm-tbccl (only fills variables the user did not set)."""
    os.environ.setdefault("VLLM_CPU_GROUP_BACKEND", "tbccl")       # vLLM control groups (needs scripts/apply_vllm_patch.py)
    os.environ.setdefault("VLLM_METAL_DIST_BACKEND", "tbccl")      # vllm-metal worker's WORLD/device groups
    os.environ.setdefault("VLLM_METAL_PP_TRANSPORT_CLS", METAL_TRANSPORT_CLS)
