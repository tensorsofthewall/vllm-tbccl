"""vllm-tbccl: carry vLLM device-group communication over torch-tbccl.

Never links libtbccl and implements no collective algorithm: everything goes through torch.distributed with the
"tbccl" ProcessGroup backend from torch-tbccl.
"""
__version__ = "0.1.0.dev0"
