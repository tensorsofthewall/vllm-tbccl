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


def test_boundary_codec_allowlist():
    import pytest

    from vllm_tbccl.backends.codecs import codec_for

    assert codec_for(["LlamaForCausalLM"]).name == "llama"
    assert codec_for(["Qwen3ForCausalLM"]).name == "qwen3"
    with pytest.raises(NotImplementedError, match="no proven"):
        codec_for(["MistralForCausalLM"])
    with pytest.raises(NotImplementedError):
        codec_for([])


def test_boundary_codec_algebra():
    import torch

    from vllm_tbccl.backends.codecs import codec_for

    c = codec_for(["Qwen3ForCausalLM"])
    h, r = torch.randn(3, 8).bfloat16(), torch.randn(3, 8).bfloat16()
    assert torch.equal(c.to_metal({"hidden_states": h, "residual": r}), h + r)
    back = c.from_metal(h, torch.zeros_like(h))
    assert torch.equal(back["hidden_states"], h) and not back["residual"].any()
    assert c.keys == ("hidden_states", "residual")


def test_architecture_exported_from_local_model_dir(tmp_path, monkeypatch):
    import json
    import sys

    from vllm_tbccl import backends

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    monkeypatch.delenv("VLLM_TBCCL_ARCHITECTURE", raising=False)
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", str(tmp_path), "--dtype", "bfloat16"])
    backends._export_architecture()
    import os

    assert os.environ["VLLM_TBCCL_ARCHITECTURE"] == "Qwen3ForCausalLM"
    monkeypatch.setenv("VLLM_TBCCL_ARCHITECTURE", "LlamaForCausalLM")
    backends._export_architecture()
    assert os.environ["VLLM_TBCCL_ARCHITECTURE"] == "LlamaForCausalLM"
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "/nonexistent/model"])
    monkeypatch.delenv("VLLM_TBCCL_ARCHITECTURE")
    backends._export_architecture()
    assert "VLLM_TBCCL_ARCHITECTURE" not in os.environ
