"""Release gate: vllm-tbccl must stand alone. It never depends on, imports or links torch-tbccl (nor any other adapter)."""
import ast
import importlib.metadata as md
import importlib.util
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the package that is actually imported: the checkout in a development tree, the installed wheel when the sources are not next to the tests
PKG = list(importlib.util.find_spec("vllm_tbccl").submodule_search_locations)[0]
FORBIDDEN = {"torch_tbccl", "exo_tbccl"}
# modules allowed to reference an optional compatibility package, and only lazily (inside a function) or as a dotted string
OPTIONAL = {os.path.join(PKG, "_backend.py"), os.path.join(PKG, "diagnostics.py")}


def _imports(path):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node, a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node, node.module.split(".")[0]


def test_scan_covers_the_package():
    assert sum(1 for _ in _python_files()) >= 8, PKG


def _python_files():
    for base, _, names in os.walk(PKG):
        for n in names:
            if n.endswith(".py"):
                yield os.path.join(base, n)


def test_no_module_level_import_of_another_adapter():
    bad = []
    for path in _python_files():
        tree = ast.parse(open(path).read())
        top_level = {id(n) for n in tree.body}
        for node, mod in _imports(path):
            if mod not in FORBIDDEN:
                continue
            if path in OPTIONAL and id(node) not in top_level:
                continue  # a lazy import in an explicitly optional compatibility path
            bad.append(f"{os.path.relpath(path, PKG)}:{node.lineno} imports {mod}")
    assert not bad, bad


def test_pyproject_does_not_require_another_adapter():
    text = open(os.path.join(ROOT, "pyproject.toml")).read()
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M).group(1)
    for name in ("torch-tbccl", "torch_tbccl", "exo-tbccl", "exo_tbccl"):
        assert name not in deps, deps


def test_compatibility_manifest_has_no_adapter_requirement():
    import json

    requires = json.load(open(os.path.join(ROOT, "compatibility.json")))["requires"]
    assert "torch_tbccl" not in requires and "torch-tbccl" not in requires
    assert requires["tbccl"] == {"c_abi": [1], "wire_protocol": [4]}


def test_installed_metadata_has_no_adapter_requirement():
    try:
        reqs = md.requires("vllm-tbccl") or []
    except md.PackageNotFoundError:
        pytest.skip("vllm-tbccl is not installed")
    assert not [r for r in reqs if re.match(r"\s*(torch[-_]tbccl|exo[-_]tbccl)", r, re.I)], reqs
