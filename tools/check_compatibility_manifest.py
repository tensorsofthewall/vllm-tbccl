#!/usr/bin/env python3
"""Check that compatibility.json agrees with vllm_tbccl/platform.py and pyproject.toml.

The sources stay the single definition; this check fails when the manifest drifts from them. The TBCCL C ABI and wire protocol
versions are defined by TBCCL and enforced by torch-tbccl; this project records which ones its validation used.
The files are read statically: nothing is imported.
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "compatibility.json").read_text())

supported = None
for node in ast.parse((ROOT / "vllm_tbccl" / "platform.py").read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SUPPORTED_VLLM":
        supported = ast.literal_eval(node.value)

pyproject = (ROOT / "pyproject.toml").read_text()
checks = {
    "package.development_version": (manifest["package"]["development_version"], re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)),
    "python.requires": (manifest["python"]["requires"], re.search(r'requires-python\s*=\s*"([^"]+)"', pyproject).group(1)),
    "frameworks.vllm.supported": (tuple(manifest["frameworks"]["vllm"]["supported"]), tuple(supported) if supported else None),
}
bad = [k for k, (declared, actual) in checks.items() if declared != actual]
for k in bad:
    print(f"MISMATCH {k}: manifest {checks[k][0]!r}, sources {checks[k][1]!r}")
if bad:
    sys.exit(1)
print("compatibility.json agrees with the sources")
