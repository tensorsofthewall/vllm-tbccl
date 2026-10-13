"""Row order between a Metal stage and an upstream stage: the permutation helper, the upstream-side state, the worker hook and the communicator metadata.

The observed defect: the Metal stage derived the CUDA activation row order from the scheduler output
(scheduled_new_reqs + scheduled_cached_reqs); the real order is the persistent input batch. They differ when a chunked-prefill continuation
sits in a reused slot, and rows of different requests were exchanged.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_tbccl import roworder
from vllm_tbccl.roworder import RowOrderError, UpstreamRowState, normalize, row_permutation


def rows_for(order):
    """One distinct value per (request, row): the request's rows carry its name and index."""
    return [(rid, i) for rid, n in order for i in range(n)]


def apply(rows, perm):
    return rows if perm is None else [rows[i] for i in perm]


def test_identity_is_none():
    o = [("A", 1), ("B", 2), ("C", 1)]
    assert row_permutation(o, o) is None
    assert row_permutation(o, list(o)) is None


def test_simple_reorder_maps_exact_rows():
    src, dst = [("A", 1), ("B", 1), ("C", 1)], [("C", 1), ("A", 1), ("B", 1)]
    perm = row_permutation(src, dst)
    assert perm.tolist() == [2, 0, 1]
    assert apply(rows_for(src), perm) == rows_for(dst)


def test_multi_row_requests_keep_their_rows_together_and_in_order():
    src, dst = [("A", 3), ("B", 2), ("C", 1)], [("C", 1), ("B", 2), ("A", 3)]
    perm = row_permutation(src, dst)
    assert perm.tolist() == [5, 3, 4, 0, 1, 2]
    assert apply(rows_for(src), perm) == rows_for(dst)


def test_chunked_prefill_case_from_the_defect():
    # CUDA input-batch order: the 670-token continuation holds slot 0; scheduler output lists it last (it is a cached request).
    decodes = [(f"d{i}", 1) for i in range(10)]
    cuda_actual = [("p", 670)] + decodes
    scheduler_order = decodes + [("p", 670)]
    metal_packed = decodes + [("p", 670)]          # vllm-metal: [decode rows][prefill rows]
    perm = row_permutation(cuda_actual, metal_packed)
    assert perm is not None
    assert apply(rows_for(cuda_actual), perm) == rows_for(metal_packed)
    # the old derivation (scheduler order taken as the upstream order) called this identity and mixed 670 rows with 10 decode rows
    assert row_permutation(scheduler_order, metal_packed) is None
    assert rows_for(cuda_actual) != rows_for(scheduler_order)


def legacy_scheduler_permutation(scheduler_new, scheduler_cached, counts, packed):
    """The pre-fix derivation, kept here only to demonstrate that it disagrees with the real order."""
    up_ids = list(scheduler_new) + list(scheduler_cached)
    off, up_off = 0, {}
    for rid in up_ids:
        up_off[rid] = off
        off += counts[rid]
    return np.concatenate([np.arange(up_off[r], up_off[r] + n) for r, n in packed])


class InputBatch:
    """The persistent input batch of an upstream runner: slots are reused by new requests; finished requests are condensed away."""

    def __init__(self):
        self.req_ids = []

    def add(self, rid):
        self.req_ids.append(rid)

    def remove_condense(self, rid):          # vLLM: the last request moves into the freed slot
        i = self.req_ids.index(rid)
        last = self.req_ids.pop()
        if i < len(self.req_ids):
            self.req_ids[i] = last

    @property
    def num_reqs(self):
        return len(self.req_ids)


def step_state(batch, counts):
    runner = SimpleNamespace(input_batch=batch)
    st = UpstreamRowState()
    st.begin_step(runner, SimpleNamespace(num_scheduled_tokens=counts))
    return st


def metal_packed_order(counts, prefill_ids):
    decodes = [(r, n) for r, n in counts.items() if r not in prefill_ids]
    return decodes + [(r, counts[r]) for r in counts if r in prefill_ids]


def test_slot_reuse_and_compaction_over_consecutive_steps():
    batch = InputBatch()
    for r in ("a", "b", "c", "d"):
        batch.add(r)
    plan = []          # (batch order, tokens scheduled per request, requests that prefill this step)

    def snap(counts, prefill):
        assert list(counts) and set(counts) == set(batch.req_ids)
        plan.append((list(batch.req_ids), dict(counts), set(prefill)))

    snap({"a": 1, "b": 1, "c": 64, "d": 1}, {"c"})            # c is a chunked prefill
    batch.remove_condense("a")                                # a finishes: d (the last) moves into a's slot
    batch.add("e")                                            # a new request takes the tail
    snap({"b": 1, "d": 1, "e": 32, "c": 64}, {"c", "e"})      # the scheduler lists the continuing prefill c last; the batch holds it in slot 2
    batch.remove_condense("b")                                # b finishes: e moves into b's slot
    batch.add("f")
    snap({"c": 1, "d": 1, "f": 16, "e": 1}, {"f"})
    batch.remove_condense("d")
    snap({"f": 1, "e": 1, "c": 1}, set())
    differs = 0
    for order, counts, prefill in plan:
        b = InputBatch()
        b.req_ids = list(order)
        local = step_state(b, counts).local_order()
        assert [r for r, _ in local] == order
        packed = metal_packed_order(counts, prefill)
        perm = row_permutation(local, packed)             # CUDA -> Metal
        got = apply(rows_for(local), perm)
        assert got == rows_for(packed)                    # no request receives another request's rows
        back = row_permutation(packed, local)             # Metal -> CUDA
        assert apply(rows_for(packed), back) == rows_for(local)
        differs += order != list(counts)                  # the scheduler lists requests in its own order
    assert differs >= 1, "the sequence must exercise input-batch order != scheduler order"


def test_metal_to_cuda_direction_semantics():
    packed = [("d0", 1), ("d1", 1), ("p", 4)]               # Metal rank 0 sends rows in its packed order
    cuda_batch = [("p", 4), ("d1", 1), ("d0", 1)]           # the CUDA receiver's input-batch order
    perm = row_permutation(packed, cuda_batch)
    assert apply(rows_for(packed), perm) == rows_for(cuda_batch)


@pytest.mark.parametrize("src,dst,msg", [
    ([("A", 1), ("B", 1)], [("A", 1)], "number of rows"),
    ([("A", 1), ("B", 1)], [("A", 1), ("C", 1)], "active requests"),
    ([("A", 1), ("B", 2)], [("A", 2), ("B", 1)], "rows of request"),
    ([("A", 1), ("A", 1)], [("A", 1), ("A", 1)], "duplicate"),
    ([("A", 1), ("B", 1)], [("A", 1), ("B", 1), ("A", 1)], "duplicate"),
    ([("A", 0)], [("A", 0)], "malformed"),
    ([("A", 1), ("B", 1)], [("B", 1)], "number of rows"),
])
def test_invalid_metadata_raises(src, dst, msg):
    with pytest.raises(RowOrderError, match=msg):
        row_permutation(src, dst)


def test_missing_and_extra_request_are_named():
    with pytest.raises(RowOrderError) as e:
        row_permutation([("A", 1), ("B", 1)], [("A", 1), ("C", 1)])
    assert "'B'" in str(e.value) and "'C'" in str(e.value)


def test_normalize_rejects_non_string_ids_and_garbage():
    with pytest.raises(RowOrderError):
        normalize([(1, 2)])
    with pytest.raises(RowOrderError):
        normalize(5)


def test_upstream_state_reads_the_input_batch_not_the_scheduler_order():
    b = InputBatch(); b.req_ids = ["p", "d0", "d1"]
    counts = {"d0": 1, "d1": 1, "p": 670}                   # scheduler dict order differs from the batch
    st = step_state(b, counts)
    assert st.local_order() == [("p", 670), ("d0", 1), ("d1", 1)]


def test_unscheduled_request_in_the_batch_is_an_error():
    b = InputBatch(); b.req_ids = ["a", "ghost"]
    with pytest.raises(RowOrderError, match="ghost"):
        step_state(b, {"a": 1}).local_order()


def test_state_without_a_worker_registration_fails_loudly():
    with pytest.raises(RowOrderError, match="did not register"):
        UpstreamRowState().local_order()


def test_unconsumed_peer_order_fails_the_next_step():
    st = UpstreamRowState()
    st.peer_order = [("a", 1)]
    with pytest.raises(RowOrderError, match="never consumed"):
        st.begin_step(SimpleNamespace(), SimpleNamespace())


# ---- the worker hook: received rows end up in the runner's input-batch order ----------------------------------------------------------------
def test_worker_hook_reorders_received_rows_into_the_input_batch_order(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_tbccl.worker import install_row_order_hook

    state = UpstreamRowState()
    monkeypatch.setattr(roworder, "STATE", state)
    import vllm_tbccl.worker as w
    monkeypatch.setattr(w.roworder, "STATE", state)

    seen = {}
    batch = InputBatch(); batch.req_ids = ["p", "d0", "d1"]
    runner = SimpleNamespace(input_batch=batch)
    runner.sync_and_gather_intermediate_tensors = lambda n, it, sync: seen.setdefault("tensors", it)
    install_row_order_hook(runner)
    state.begin_step(runner, SimpleNamespace(num_scheduled_tokens={"d0": 1, "d1": 1, "p": 3}))
    peer = [("d0", 1), ("d1", 1), ("p", 3)]                # Metal packed order, as sent
    state.peer_order = peer
    values = torch.tensor([[10.0], [11.0], [20.0], [21.0], [22.0]])   # d0, d1, p0, p1, p2
    it = SimpleNamespace(tensors={"hidden_states": values.clone(), "residual": values.clone() + 100})
    runner.sync_and_gather_intermediate_tensors(5, it, True)
    got = seen["tensors"].tensors
    assert got["hidden_states"][:, 0].tolist() == [20.0, 21.0, 22.0, 10.0, 11.0]    # p, d0, d1 (the input-batch order)
    assert got["residual"][:, 0].tolist() == [120.0, 121.0, 122.0, 110.0, 111.0]
    assert state.peer_order is None


def test_worker_hook_rejects_a_peer_order_that_names_other_requests(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import vllm_tbccl.worker as w

    state = UpstreamRowState()
    monkeypatch.setattr(w.roworder, "STATE", state)
    batch = InputBatch(); batch.req_ids = ["a", "b"]
    runner = SimpleNamespace(input_batch=batch)
    runner.sync_and_gather_intermediate_tensors = lambda n, it, sync: it
    w.install_row_order_hook(runner)
    state.begin_step(runner, SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1}))
    state.peer_order = [("a", 1), ("x", 1)]
    with pytest.raises(RowOrderError, match="active requests"):
        runner.sync_and_gather_intermediate_tensors(2, SimpleNamespace(tensors={}), True)


# ---- communicator: the sender states its order; a Metal peer without it is refused -----------------------------------------------------------
def _communicator(monkeypatch, peer):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm_tbccl import communicator as c

    comm = c.TBCCLDeviceCommunicator.__new__(c.TBCCLDeviceCommunicator)
    comm._peer_device_type = peer
    comm.ranks = [0, 1]
    comm.rank_in_group = 0
    comm.world_size = 2
    comm.device = __import__("torch").device("cpu")
    sent = []
    monkeypatch.setattr(comm, "_send_object", lambda obj, dst: sent.append(obj))
    monkeypatch.setattr(comm, "send", lambda t, dst=None: None)
    return comm, sent, c


def test_communicator_sends_the_local_order_to_a_metal_peer_only(monkeypatch):
    import torch

    state = UpstreamRowState()
    monkeypatch.setattr(roworder, "STATE", state)
    b = InputBatch(); b.req_ids = ["p", "d0"]
    state.begin_step(SimpleNamespace(input_batch=b), SimpleNamespace(num_scheduled_tokens={"d0": 1, "p": 3}))
    comm, sent, _ = _communicator(monkeypatch, "metal")
    comm.send_tensor_dict({"hidden_states": torch.zeros(4, 2)}, dst=1)
    meta = dict(sent[0])
    assert meta[roworder.ROW_ORDER_KEY] == [("p", 3), ("d0", 1)]
    comm2, sent2, _ = _communicator(monkeypatch, "cuda")
    comm2.send_tensor_dict({"hidden_states": torch.zeros(4, 2)}, dst=1)
    assert roworder.ROW_ORDER_KEY not in dict(sent2[0])          # upstream <-> upstream keeps vLLM's own wire format


def test_communicator_refuses_a_metal_peer_without_row_order(monkeypatch):
    pytest.importorskip("vllm")
    import torch
    from vllm.distributed.parallel_state import TensorMetadata

    comm, _, _ = _communicator(monkeypatch, "metal")
    meta = [("hidden_states", TensorMetadata("cpu", torch.bfloat16, torch.Size([0, 2])))]
    monkeypatch.setattr(comm, "_recv_object", lambda src: meta)
    with pytest.raises(RowOrderError, match="did not send its row order"):
        comm.recv_tensor_dict(1)


def test_communicator_parks_the_peer_order_for_the_worker(monkeypatch):
    pytest.importorskip("vllm")
    import torch
    from vllm.distributed.parallel_state import TensorMetadata

    state = UpstreamRowState()
    monkeypatch.setattr(roworder, "STATE", state)
    comm, _, _ = _communicator(monkeypatch, "metal")
    meta = [("hidden_states", TensorMetadata("cpu", torch.bfloat16, torch.Size([0, 2]))), (roworder.ROW_ORDER_KEY, [("a", 2), ("b", 1)])]
    monkeypatch.setattr(comm, "_recv_object", lambda src: meta)
    out = comm.recv_tensor_dict(1)
    assert set(out) == {"hidden_states"} and state.peer_order == [("a", 2), ("b", 1)]


# ---- CUDA platform: vLLM's generic isend/irecv path carries the row order as a non-tensor metadata entry ---------------------------------------
class FakeGroup:
    def __init__(self, peer="metal", custom=False):
        self.world_size = 2
        self.use_cpu_custom_send_recv = custom
        self.device_communicator = SimpleNamespace(peer_device_type=peer)
        self.sent, self.to_return = [], None

    def isend_tensor_dict(self, tensor_dict, *a, **k):
        self.sent.append(dict(tensor_dict))
        return []

    def irecv_tensor_dict(self, *a, **k):
        return dict(self.to_return), [], []


def test_cuda_generic_path_adds_the_row_order_for_a_metal_peer(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import vllm_tbccl.worker as w

    state = UpstreamRowState()
    monkeypatch.setattr(w.roworder, "STATE", state)
    b = InputBatch(); b.req_ids = ["p", "d0"]
    state.begin_step(SimpleNamespace(input_batch=b), SimpleNamespace(num_scheduled_tokens={"d0": 1, "p": 3}))
    pp = FakeGroup()
    w.install_group_row_order_hooks(pp)
    pp.isend_tensor_dict({"hidden_states": "T"}, dst=1)
    assert pp.sent[0] == {"hidden_states": "T", roworder.ROW_ORDER_KEY: [("p", 3), ("d0", 1)]}
    pp.to_return = {"hidden_states": "R", roworder.ROW_ORDER_KEY: [("d0", 1), ("p", 3)]}
    out, _, _ = pp.irecv_tensor_dict()
    assert out == {"hidden_states": "R"} and state.peer_order == [("d0", 1), ("p", 3)]
    pp.to_return = {"hidden_states": "R"}
    with pytest.raises(RowOrderError, match="did not send its row order"):
        pp.irecv_tensor_dict()


def test_generic_path_hooks_are_inert_for_cuda_cuda_and_for_the_cpu_custom_path():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import vllm_tbccl.worker as w

    for pp in (FakeGroup(peer="cuda"), FakeGroup(peer="cpu"), FakeGroup(peer="metal", custom=True)):
        before = (pp.isend_tensor_dict, pp.irecv_tensor_dict)
        w.install_group_row_order_hooks(pp)
        assert (pp.isend_tensor_dict, pp.irecv_tensor_dict) == before
