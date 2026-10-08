"""vllm-tbccl: carry vLLM device-group communication over TBCCL.

Self-contained: a private c10d "tbccl" backend (vllm_tbccl._C) talks to a statically linked libtbccl through its C ABI. It does not use, import or
require torch-tbccl.
"""
__version__ = "0.2.0.dev0"
