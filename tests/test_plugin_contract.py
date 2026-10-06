"""Phase 69: plugin discovery, platform registration, supported-version tuple and the scope of the control-group backend wrapper (vLLM 0.31.0)."""
import importlib.metadata as md
import importlib.util
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")


def test_entry_point_is_registered_for_vllm_platform_plugins():
    eps = {e.name: e for e in md.entry_points(group="vllm.platform_plugins")}
    assert eps["tbccl"].value == "vllm_tbccl.platform:tbccl_platform_plugin"


def test_plugin_is_inert_unless_enabled():
    code = "import vllm_tbccl.platform as p, os; os.environ.pop('VLLM_TBCCL_ENABLE', None); print(p.tbccl_platform_plugin())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "VLLM_TBCCL_ENABLE"})
    assert out.stdout.strip().splitlines()[-1] == "None", out.stderr[-500:]


HAS_METAL = importlib.util.find_spec("vllm_metal") is not None


@pytest.mark.skipif(HAS_METAL, reason="vllm-metal installed: it owns the platform (see test_metal_platform_stays_owner_when_both_plugins_are_installed)")
def test_platform_selected_by_vllm_discovery_has_tbccl_dist_backend_and_communicator():
    code = ("from vllm.platforms import current_platform as p; print(type(p).__name__, p.dist_backend, p.get_device_communicator_cls())")
    env = dict(os.environ, VLLM_TBCCL_ENABLE="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    last = out.stdout.strip().splitlines()[-1]
    assert last.endswith("tbccl vllm_tbccl.communicator.TBCCLDeviceCommunicator"), (out.stdout[-400:], out.stderr[-400:])
    assert last.split()[0] in ("TbcclCudaPlatform", "TbcclCpuPlatform")


@pytest.mark.skipif(not HAS_METAL, reason="needs vllm-metal")
def test_metal_platform_stays_owner_when_both_plugins_are_installed():
    code = ("from vllm.platforms import current_platform as p; import os; "
            "print(type(p).__name__, os.environ.get('VLLM_METAL_PP_TRANSPORT_CLS'), os.environ.get('VLLM_METAL_DIST_BACKEND'), os.environ.get('VLLM_CPU_GROUP_BACKEND'))")
    env = dict(os.environ, VLLM_TBCCL_ENABLE="1", VLLM_TBCCL_BACKEND="metal")
    for k in ("VLLM_METAL_PP_TRANSPORT_CLS", "VLLM_METAL_DIST_BACKEND", "VLLM_CPU_GROUP_BACKEND"):
        env.pop(k, None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip().splitlines()[-1] == "MetalPlatform vllm_tbccl.backends.metal.TBCCLMetalPipelineTransport tbccl tbccl", (out.stdout[-400:], out.stderr[-400:])
    off = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={k: v for k, v in env.items() if k != "VLLM_TBCCL_ENABLE"})
    assert off.stdout.strip().splitlines()[-1] == "MetalPlatform None None None", (off.stdout[-400:], off.stderr[-400:])


def test_supported_vllm_tuple_is_declared():
    from vllm_tbccl import platform

    assert platform.SUPPORTED_VLLM == ("0.30.0", "0.31.0")


def _record(monkeypatch):
    seen = []

    def real(*a, **k):
        seen.append((a, dict(k)))
        return object()

    monkeypatch.setattr(dist, "new_group", real)
    return seen


def _call_as(module_name, fn, *a, **k):
    g = {"__name__": module_name, "dist": dist, "a": a, "k": k}
    exec("r = dist.new_group(*a, **k)", g)
    return g["r"]


def test_control_group_backend_wrapper_is_scoped_to_vllm_parallel_state_gloo_calls(monkeypatch):
    from vllm_tbccl import platform

    seen = _record(monkeypatch)
    monkeypatch.setenv("VLLM_CPU_GROUP_BACKEND", "tbccl")
    platform.install_cpu_group_backend()
    _call_as("vllm.distributed.parallel_state", None, [0, 1], backend="gloo")          # vLLM's control group: substituted
    _call_as("vllm.distributed.parallel_state", None, [0, 1], backend="tbccl")         # the device group: untouched
    _call_as("some.other.module", None, [0, 1], backend="gloo")                        # anybody else's gloo group: untouched
    _call_as("vllm.distributed.parallel_state", None, [0, 1])                          # no backend argument: untouched
    assert [k.get("backend") for _, k in seen] == ["tbccl", "tbccl", "gloo", None]
    assert seen[0][0] == ([0, 1],) and seen[2][1]["backend"] == "gloo"


def test_wrapper_is_idempotent_and_can_be_disabled(monkeypatch):
    from vllm_tbccl import platform

    seen = _record(monkeypatch)
    monkeypatch.setenv("VLLM_CPU_GROUP_BACKEND", "gloo")
    platform.install_cpu_group_backend()
    _call_as("vllm.distributed.parallel_state", None, [0], backend="gloo")
    assert seen[-1][1]["backend"] == "gloo"          # explicitly disabled
    monkeypatch.setenv("VLLM_CPU_GROUP_BACKEND", "tbccl")
    platform.install_cpu_group_backend()
    first = dist.new_group
    platform.install_cpu_group_backend()
    assert dist.new_group is first and getattr(first, "_vllm_tbccl_wrapped", False)
    n = len(seen)
    _call_as("vllm.distributed.parallel_state", None, [0], backend="gloo")
    assert len(seen) == n + 1 and seen[-1][1]["backend"] == "tbccl"   # wrapped exactly once (no double call)


def test_every_package_directory_is_listed_in_pyproject():
    """A normal (non-editable) install must contain every subpackage: an editable install hides a missing one, and vLLM swallows a plugin import error and silently falls back to NCCL."""
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[1]
    listed = set(tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    found = {".".join(p.parent.relative_to(root).parts) for p in (root / "vllm_tbccl").rglob("__init__.py")}
    assert found <= listed, f"missing from pyproject packages: {sorted(found - listed)}"
