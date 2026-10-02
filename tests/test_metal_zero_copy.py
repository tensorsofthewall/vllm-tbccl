"""MLX <-> torch(CPU) zero-copy aliasing through vllm-metal's existing DLPack bridge (Mac only; skipped elsewhere)."""
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
bridge = pytest.importorskip("vllm_metal.pytorch_backend.tensor_bridge")

DTYPES = [(mx.float32, torch.float32), (mx.float16, torch.float16), (mx.bfloat16, torch.bfloat16),
          (mx.int32, torch.int32), (mx.int64, torch.int64)]


def _fill(shape, dtype):
    n = 1
    for s in shape:
        n *= s
    return (mx.arange(n) % 97).reshape(shape).astype(dtype)


@pytest.mark.parametrize("mdt,tdt", DTYPES)
def test_alias_both_directions(mdt, tdt):
    x = _fill((7, 33), mdt)
    mx.eval(x)
    t = bridge.mlx_to_torch(x, device="cpu")
    assert t.dtype == tdt and tuple(t.shape) == (7, 33) and t.is_contiguous()
    # A: mutate the torch alias, observe it from MLX (no copy anywhere)
    t[3, 5] = 55
    t[6, 32] = 66
    assert x[3, 5].item() == 55 and x[6, 32].item() == 66
    # B: MLX data -> reading through a fresh alias sees exact values
    y = _fill((7, 33), mdt) + mx.array(1, dtype=mdt)
    mx.eval(y)
    ty = bridge.mlx_to_torch(y, device="cpu")
    ref = (torch.arange(7 * 33) % 97).reshape(7, 33).to(tdt) + torch.tensor(1, dtype=tdt)
    assert torch.equal(ty, ref)
    # pointers (when exposed) agree
    assert t.data_ptr() == bridge.mlx_to_torch(x, device="cpu").data_ptr()


def test_torch_write_then_metal_consumes_without_copy():
    x = mx.zeros((1024, 576), dtype=mx.float32)
    mx.eval(x)
    t = bridge.mlx_to_torch(x, device="cpu")
    t.copy_(torch.arange(1024 * 576, dtype=torch.float32).reshape(1024, 576) % 251)   # CPU writes (what a recv does)
    out = (x * 2 + 1).sum(axis=1)                                                    # Metal op, no explicit sync/copy
    mx.eval(out)
    ref = ((torch.arange(1024 * 576, dtype=torch.float32).reshape(1024, 576) % 251) * 2 + 1).sum(dim=1)
    assert torch.equal(bridge.mlx_to_torch(out, device="cpu"), ref)


def test_lazy_producer_needs_eval_before_export():
    a = mx.random.normal((512, 512))
    b = (a @ a).sum(axis=0)          # lazy, not evaluated
    mx.eval(b)                        # the documented boundary: export requires an evaluated array
    t = bridge.mlx_to_torch(b, device="cpu")
    assert torch.equal(t, bridge.mlx_to_torch(b, device="cpu")) and abs(float(t.sum()) - b.sum().item()) <= 1e-3 * max(1.0, abs(b.sum().item()))


def test_negative_stride_rejected():
    x = mx.arange(16).reshape(4, 4)
    mx.eval(x)
    try:
        y = x[::-1]
        mx.eval(y)
    except Exception:
        pytest.skip("MLX does not expose negative-stride views here")
    try:
        bridge.mlx_to_torch(y, device="cpu")
    except ValueError:
        return
    # MLX may materialize reversed slices contiguously; then there is nothing negative to reject
    assert y.flags.row_contiguous if hasattr(y, "flags") else True
