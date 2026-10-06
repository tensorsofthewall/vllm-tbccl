"""Phase 69: the heterogeneous-stage worker subclasses (KV-layout agreement, CPU PP buffer) that replace the old vLLM patches."""
import pytest

pytest.importorskip("vllm")
from vllm_tbccl.worker import KIND_LAYOUTS, restrict_layouts  # noqa: E402

DEFAULT = ["LBNHC", "LBHNC", "BLNHC", "BLHNC", "BHLNC", "LHBNC"]


def test_cuda_worker_with_a_cpu_peer_reports_the_common_layout():
    assert restrict_layouts(DEFAULT, "cpu", "cuda") == ["LBHNC"]


def test_cpu_worker_and_same_kind_or_unknown_peers_are_unchanged():
    assert restrict_layouts(["LBHNC"], "cuda", "cpu") == ["LBHNC"]
    assert restrict_layouts(DEFAULT, "cuda", "cuda") == DEFAULT
    assert restrict_layouts(DEFAULT, None, "cuda") == DEFAULT


def test_disjoint_support_is_an_error_naming_both_sides():
    with pytest.raises(ValueError, match="cuda worker.*cpu pipeline peer"):
        restrict_layouts(["LBNHC"], "cpu", "cuda")


def test_explicit_override_wins_and_is_validated():
    assert restrict_layouts(DEFAULT, None, "cuda", "BLNHC,LBHNC") == ["BLNHC", "LBHNC"]
    with pytest.raises(ValueError):
        restrict_layouts(["LBHNC"], None, "cpu", "LBNHC")


def test_the_engine_core_assertion_is_what_the_restriction_resolves():
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    cfg = VllmConfig()
    with pytest.raises(AssertionError, match="disagree"):
        resolve_kv_cache_layout(cfg, [DEFAULT, ["LBHNC"]], None)
    cuda, cpu = restrict_layouts(DEFAULT, "cpu", "cuda"), ["LBHNC"]
    assert cuda == cpu
    assert resolve_kv_cache_layout(VllmConfig(), [cuda, cpu], None).name == "LBHNC"


def test_known_kind_table_matches_vllm_cpu_backend():
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    assert [x.name for x in CPUAttentionBackend.supported_kv_cache_layouts()] == list(KIND_LAYOUTS["cpu"])
