"""Lifecycle and failure behaviour of the vLLM pipeline groups over vllm-tbccl (GroupCoordinator level, two local processes)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
HAS_CUDA = torch.cuda.is_available()
PLATFORMS = [("cuda", "cuda")] if HAS_CUDA else [("cpu", "cpu")]


@pytest.mark.parametrize("platforms", PLATFORMS)
def test_repeated_group_create_destroy_leaks_nothing(run_pair, platforms):
    for rc, out in run_pair("examples/lifecycle_probe.py", platforms, timeout=300, args=("--mode", "cycles")):
        assert rc == 0 and " ok" in out, out[-2000:]


@pytest.mark.parametrize("platforms", PLATFORMS)
def test_destroy_with_nothing_pending_is_prompt(run_pair, platforms):
    for rc, out in run_pair("examples/lifecycle_probe.py", platforms, timeout=120, args=("--mode", "destroy_idle")):
        assert rc == 0 and " ok" in out, out[-2000:]


@pytest.mark.parametrize("mode", ["peer_exit_recv", "peer_exit_send"])
@pytest.mark.parametrize("platforms", PLATFORMS)
def test_peer_exit_surfaces_an_error_in_bounded_time(run_pair, platforms, mode):
    (rc0, out0), (rc1, out1) = run_pair("examples/lifecycle_probe.py", platforms, timeout=180, args=("--mode", mode))
    assert rc0 == 0 and "rank 0 ok" in out0, out0[-2500:]
    assert "exiting abruptly" in out1


@pytest.mark.parametrize("platforms", PLATFORMS)
def test_silent_peer_then_abort_unblocks_recv(run_pair, platforms):
    for rc, out in run_pair("examples/tbccl_silent_peer.py", platforms, timeout=180):
        assert rc == 0, out[-2500:]
