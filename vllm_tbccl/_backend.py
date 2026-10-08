"""Registration of vllm-tbccl's own "tbccl" c10d backend (native module ``vllm_tbccl._C``).

``VLLM_TBCCL_PROCESS_GROUP=native`` (default) uses it. ``VLLM_TBCCL_PROCESS_GROUP=torch-tbccl`` is an optional compatibility mode that uses the separately
installed torch-tbccl package's backend instead; vllm-tbccl never requires that package.
"""
import os
import re

import torch
import torch.distributed as dist

BACKEND_NAME = "tbccl"
SUPPORTED_C_ABI = (1,)
TESTED_WIRE_PROTOCOL = (4,)
MODES = ("native", "torch-tbccl")


def mode() -> str:
    value = os.environ.get("VLLM_TBCCL_PROCESS_GROUP", "native").lower()
    if value not in MODES:
        raise ValueError(f"VLLM_TBCCL_PROCESS_GROUP must be one of {MODES}, got {value!r}")
    return value


def _series(version: str) -> str:
    m = re.match(r"(\d+)\.(\d+)", version)
    return f"{m.group(1)}.{m.group(2)}" if m else version


def _load_native():
    from . import _C

    built, running = _series(_C.built_with_torch()), _series(torch.__version__)
    if built != running and os.environ.get("VLLM_TBCCL_ALLOW_TORCH_MISMATCH") != "1":
        raise ImportError(
            f"vllm-tbccl's native module was compiled against torch {_C.built_with_torch()} but torch {torch.__version__} is installed; the extension links "
            f"torch's C++ ABI, which is only stable within one minor series ({built} != {running}). Install the vllm-tbccl build for this torch "
            "(VLLM_TBCCL_ALLOW_TORCH_MISMATCH=1 skips this check at your own risk).")
    if _C.built_c_abi_version() not in SUPPORTED_C_ABI or _C.c_abi_version() != _C.built_c_abi_version():
        raise ImportError(
            f"vllm-tbccl supports libtbccl C ABI {list(SUPPORTED_C_ABI)}; this build was compiled against {_C.built_c_abi_version()} and links {_C.c_abi_version()}")
    if _C.wire_protocol_version() not in TESTED_WIRE_PROTOCOL:
        import warnings

        warnings.warn(f"vllm-tbccl was built against libtbccl wire protocol {_C.wire_protocol_version()}; only {list(TESTED_WIRE_PROTOCOL)} is tested", stacklevel=2)
    return _C


def register_backend() -> None:
    """Make ``backend="tbccl"`` available to torch.distributed. Idempotent; an already-registered "tbccl" (any provider) is left alone."""
    if BACKEND_NAME in dist.Backend.backend_list:
        return
    if mode() == "torch-tbccl":
        try:
            import torch_tbccl  # noqa: F401  (optional compatibility mode; registers "tbccl" itself)
        except ImportError as e:
            raise ImportError("VLLM_TBCCL_PROCESS_GROUP=torch-tbccl needs the torch-tbccl package, which is not installed") from e
        return
    native = _load_native()
    dist.Backend.register_backend(BACKEND_NAME, native.create_backend, devices=[d for d in ("cpu", "cuda") if native.compiled_features().get(d)])


def op_stats() -> dict:
    """{operation: {"count", "bytes"}} submitted to libtbccl by this process's native backend (empty in the torch-tbccl compatibility mode)."""
    return dict(_load_native().op_stats()) if mode() == "native" else {}


def reset_op_stats() -> None:
    if mode() == "native":
        _load_native().reset_op_stats()


def info() -> dict:
    """Versions of every layer of this installation (``python -m vllm_tbccl.info``)."""
    import platform

    from . import __version__

    native = _load_native()
    return {
        "vllm_tbccl": __version__,
        "location": os.path.dirname(os.path.abspath(__file__)),
        "torch_running": torch.__version__,
        "torch_built_with": native.built_with_torch(),
        "libtbccl": native.runtime_version(),
        "c_abi": native.c_abi_version(),
        "wire_protocol": native.wire_protocol_version(),
        "devices": [d for d, ok in dict(native.compiled_features()).items() if ok],
        "process_group": mode(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "backend": BACKEND_NAME,
        "registered": BACKEND_NAME in dist.Backend.backend_list,
    }
