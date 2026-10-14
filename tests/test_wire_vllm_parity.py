"""vllm_tbccl.wire reproduces vLLM's GroupCoordinator.send_object/recv_object byte for byte (pinned for the vLLM versions in SUPPORTED_VLLM)."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")
import vllm  # noqa: E402
from vllm.distributed.parallel_state import GroupCoordinator  # noqa: E402

from vllm_tbccl import wire  # noqa: E402
from vllm_tbccl.platform import SUPPORTED_VLLM  # noqa: E402

OBJECTS = [{"a": 1, "b": [1, 2, 3]}, ("x", None, 3.5), list(range(10_000))]


@pytest.fixture
def channel(monkeypatch):
    sent = deque()

    def fake_send(tensor, dst, group=None):
        sent.append((tensor.clone(), dst, group))

    def fake_recv(tensor, src, group=None):
        t, dst, grp = sent.popleft()
        assert grp is group and tuple(t.shape) == tuple(tensor.shape) and t.dtype == tensor.dtype
        tensor.copy_(t)

    monkeypatch.setattr(torch.distributed, "send", fake_send)
    monkeypatch.setattr(torch.distributed, "recv", fake_recv)
    return sent


def _coordinator(group):
    return SimpleNamespace(world_size=2, rank_in_group=0, ranks=[0, 1], cpu_group=group)


def test_vllm_version_is_one_this_wire_format_was_checked_against():
    assert vllm.__version__.split("+")[0] in SUPPORTED_VLLM


@pytest.mark.parametrize("obj", OBJECTS)
def test_send_object_matches_vllm(channel, obj):
    group = object()
    GroupCoordinator.send_object(_coordinator(group), obj, 1)
    theirs = list(channel)
    channel.clear()
    wire.send_object(obj, 1, group)
    ours = list(channel)
    assert len(theirs) == len(ours) == 2
    for (t1, d1, g1), (t2, d2, g2) in zip(theirs, ours):
        assert torch.equal(t1, t2) and t1.dtype == t2.dtype and d1 == d2 and g1 is g2


@pytest.mark.parametrize("obj", OBJECTS)
def test_recv_object_reads_vllm_frames_and_back(channel, obj):
    group = object()
    GroupCoordinator.send_object(_coordinator(group), obj, 1)
    assert wire.recv_object(1, group) == obj
    wire.send_object(obj, 1, group)
    coordinator = SimpleNamespace(world_size=2, rank_in_group=1, ranks=[0, 1], cpu_group=group)
    assert GroupCoordinator.recv_object(coordinator, 0) == obj
