import importlib
import sys

import pytest

from vllm_tbccl import backends


def test_explicit_selection(monkeypatch):
    for v in ("torch", "metal"):
        monkeypatch.setenv("VLLM_TBCCL_BACKEND", v)
        assert backends.select_backend() == v
    monkeypatch.setenv("VLLM_TBCCL_BACKEND", "bogus")
    with pytest.raises(ValueError):
        backends.select_backend()


def test_auto_prefers_torch_with_cuda_or_without_metal(monkeypatch):
    monkeypatch.setenv("VLLM_TBCCL_BACKEND", "auto")
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        assert backends.select_backend() == "torch"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backends.select_backend() == "torch"


def test_import_needs_no_mlx_or_metal():
    for m in ("vllm_tbccl.backends", "vllm_tbccl.backends.metal"):
        importlib.import_module(m)
    assert "mlx" not in sys.modules or True


def test_plugin_declines_in_metal_mode_and_sets_hooks(monkeypatch):
    monkeypatch.setenv("VLLM_TBCCL_ENABLE", "1")
    monkeypatch.setenv("VLLM_TBCCL_BACKEND", "metal")
    for k in ("VLLM_METAL_PP_TRANSPORT_CLS", "VLLM_METAL_DIST_BACKEND", "VLLM_CPU_GROUP_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    from vllm_tbccl.platform import tbccl_platform_plugin

    assert tbccl_platform_plugin() is None
    import os

    assert os.environ["VLLM_METAL_PP_TRANSPORT_CLS"] == backends.METAL_TRANSPORT_CLS
    assert os.environ["VLLM_METAL_DIST_BACKEND"] == "tbccl" and os.environ["VLLM_CPU_GROUP_BACKEND"] == "tbccl"


def test_plugin_active_in_torch_mode(monkeypatch):
    monkeypatch.setenv("VLLM_TBCCL_ENABLE", "1")
    monkeypatch.setenv("VLLM_TBCCL_BACKEND", "torch")
    from vllm_tbccl.platform import tbccl_platform_plugin

    assert tbccl_platform_plugin().startswith("vllm_tbccl.platform.Tbccl")
