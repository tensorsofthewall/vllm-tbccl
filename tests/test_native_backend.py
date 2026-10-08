"""vllm-tbccl's own c10d backend, exercised through plain torch.distributed on loopback (no vLLM model, no torch-tbccl).

Needs the compiled vllm_tbccl._C (a build with TBCCL_ROOT); skipped when it is absent.
"""
import importlib.util
import os
import socket
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tests", "_worker_native.py")

pytestmark = pytest.mark.skipif(importlib.util.find_spec("vllm_tbccl._C") is None, reason="vllm_tbccl._C is not built")


def _free_ports(n=4):
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        ports = sorted(s.getsockname()[1] for s in socks)
    finally:
        for s in socks:
            s.close()
    return ",".join(map(str, ports))


def run_group(case, world, timeout=120):
    ports = _free_ports()
    env = dict(os.environ, TBCCL_LOCAL_ENDPOINT="127.0.0.1:0", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    procs = [subprocess.Popen([sys.executable, WORKER, "--case", case, "--rank", str(r), "--world", str(world), "--ports", ports], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(world)]
    out = []
    for p in procs:
        try:
            o, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            o, _ = p.communicate()
            o = "TIMEOUT\n" + o
            p.returncode = -9
        out.append((p.returncode, o))
    assert all(rc == 0 for rc, _ in out), "\n".join(f"--- rank {i} rc={rc}\n{o}" for i, (rc, o) in enumerate(out))


@pytest.mark.parametrize("world", [2, 3, 4])
def test_collectives_world(world):
    run_group("basic", world)


def test_send_recv_dtypes():
    run_group("p2p", 2)


def test_buffer_lifetime_after_dropping_work():
    run_group("lifetime", 2)


def test_abort_fails_pending_and_later_operations():
    run_group("error", 2)


def test_repeated_initialization_over_new_stores():
    run_group("reinit", 2)


def test_subgroup_on_same_backend():
    run_group("subgroup", 3)


def test_torch_tbccl_is_not_imported_by_the_backend():
    run_group("independence", 2)
