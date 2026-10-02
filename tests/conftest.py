import os
import socket
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def run_pair():
    """Run a script as two local processes (rank 0 / rank 1) with the given base platforms; return [(rc, output)]."""

    def run(script, platforms=("cuda", "cuda"), timeout=180, args=()):
        init = f"tcp://127.0.0.1:{_free_port()}"
        procs = []
        for rank, plat in enumerate(platforms):
            env = dict(os.environ, VLLM_TBCCL_ENABLE="1", VLLM_TBCCL_PLATFORM=plat, VLLM_TBCCL_TRACE="1",
                       TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", RANK=str(rank))
            if plat == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, script), "--init", init, *args], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        out = []
        for p in procs:
            try:
                o, _ = p.communicate(timeout=timeout)
                out.append((p.returncode, o))
            except subprocess.TimeoutExpired:
                p.kill()
                o, _ = p.communicate()
                out.append((-9, "TIMEOUT\n" + o))
        return out

    return run
