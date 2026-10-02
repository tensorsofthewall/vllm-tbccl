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
    _export_architecture()


def _model_dir_from_argv():
    """The model directory of a ``vllm serve <model>`` / ``--model <model>`` command line, if it is a local directory."""
    argv = sys.argv
    cand = []
    if "--model" in argv and argv.index("--model") + 1 < len(argv):
        cand.append(argv[argv.index("--model") + 1])
    if "serve" in argv and argv.index("serve") + 1 < len(argv):
        cand.append(argv[argv.index("serve") + 1])
    return next((c for c in cand if os.path.isfile(os.path.join(c, "config.json"))), None)


def _export_architecture() -> None:
    """vllm-metal's pipeline send/recv runs outside any vLLM config context, so the boundary codec learns the model architecture
    from the environment: exported here in the parent process (workers inherit it) from the local model's config.json."""
    if os.environ.get("VLLM_TBCCL_ARCHITECTURE"):
        return
    model = _model_dir_from_argv()
    if model is None:
        return
    import json

    archs = json.load(open(os.path.join(model, "config.json"))).get("architectures") or []
    if archs:
        os.environ["VLLM_TBCCL_ARCHITECTURE"] = archs[0]
