import pytest

torch = pytest.importorskip("torch")
HAS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
@pytest.mark.parametrize("platforms", [("cuda", "cuda"), ("cuda", "cpu"), ("cpu", "cuda")])
def test_pp2_group_smoke(run_pair, platforms):
    results = run_pair("examples/tbccl_group_smoke.py", platforms)
    for rc, out in results:
        assert rc == 0, out
        assert "ProcessGroupTBCCL ops=" in out and " ok" in out


def test_platform_inactive_by_default():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c", "from vllm.platforms import current_platform as p; print(type(p).__name__)"],
                         capture_output=True, text=True, env={k: v for k, v in __import__('os').environ.items() if k != "VLLM_TBCCL_ENABLE"}).stdout
    assert "Tbccl" not in out


def test_unsupported_ops_raise():
    from vllm_tbccl.communicator import TBCCLDeviceCommunicator
    inst = object.__new__(TBCCLDeviceCommunicator)
    for name in ("reduce_scatter", "gather", "all_gatherv", "reduce_scatterv"):
        with pytest.raises(NotImplementedError):
            getattr(inst, name)(None)
