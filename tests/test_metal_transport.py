"""Mac-only: TBCCLMetalPipelineTransport against an upstream-style peer and a Metal peer (two local processes)."""
import os
import socket
import subprocess
import sys

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("vllm_metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(mode, plats):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = []
    for rank in (0, 1):
        env = dict(os.environ, RANK=str(rank), TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", VLLM_TBCCL_ENABLE="1",
                   VLLM_TBCCL_BACKEND=plats[rank], VLLM_TBCCL_PLATFORM="cpu", VLLM_CPU_GROUP_BACKEND="tbccl",
                   # torch mode must own the platform: with vllm-metal also installed, vLLM refuses two active platform plugins
                   # unless VLLM_PLUGINS selects one. Metal mode keeps both listed: vllm-tbccl declines, vllm-metal owns the platform.
                   VLLM_PLUGINS="tbccl" if plats[rank] == "torch" else "metal,tbccl")
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "examples/metal_transport_smoke.py"), "--init",
                                       f"tcp://127.0.0.1:{port}", "--mode", mode], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True))
    outs = []
    for p in procs:
        try:
            o, _ = p.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            o, _ = p.communicate()
            o = "TIMEOUT\n" + o
        outs.append((p.returncode, o))
    return outs


def test_upstream_peer_codec_and_row_order():
    # rank 0 = upstream-style CPU platform, rank 1 = Metal transport (platform plugin declines; vllm-metal owns the platform)
    outs = _run("upstream", ("torch", "metal"))
    for rc, o in outs:
        assert rc == 0, o
    assert "rank 0 ok" in outs[0][1] and "rank 1 ok" in outs[1][1] and "peer_kind=cpu" in outs[1][1]


def test_raw_metal_to_metal():
    outs = _run("raw", ("metal", "metal"))
    for rc, o in outs:
        assert rc == 0, o
    assert "peer_kind=metal" in outs[0][1] and "rank 1 ok" in outs[1][1]
