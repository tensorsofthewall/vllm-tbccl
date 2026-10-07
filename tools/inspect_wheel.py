"""Inspect a built vllm-tbccl wheel WITHOUT importing it.

    python tools/inspect_wheel.py dist/vllm_tbccl-*.whl [--json OUT]

Fails (exit 1) on: a missing module or metadata; a missing licence file; a version that disagrees with vllm_tbccl/__init__.py; a platform tag other than py3-none-any; development files (tests, tools,
docs, examples, sources) or build-machine paths in any member.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

MODULE = "vllm_tbccl"
REQUIRED = ["vllm_tbccl/__init__.py"]
NATIVE = False
SYSTEM_DIRS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/local/lib")
DEV_PREFIXES = ("tests/", "tools/", "docs/", "examples/", "scripts/", "benchmarks/", ".github/")


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_tags(filename):
    parts = os.path.basename(filename)[:-4].split("-")
    return parts[-3], parts[-2], parts[-1].split(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    problems, report = [], {"wheel": os.path.basename(a.wheel)}
    z = zipfile.ZipFile(a.wheel)
    names = z.namelist()
    report["members"] = names
    for m in REQUIRED:
        if m not in names:
            problems.append(f"missing module {m}")
    meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    text = z.read(meta).decode() if meta else ""
    if not meta:
        problems.append("missing METADATA")
    report["requires_dist"] = re.findall(r"^Requires-Dist: (.+)$", text, re.M)
    report["requires_python"] = (re.search(r"^Requires-Python: (.+)$", text, re.M) or [None, None])[1]
    py_tag, abi_tag, plats = parse_tags(a.wheel)
    report["tag"] = [py_tag, abi_tag, plats]
    if (py_tag, abi_tag, plats) != ("py3", "none", ["any"]):
        problems.append(f"a pure-Python wheel must be py3-none-any, got {py_tag}-{abi_tag}-{plats}")
    if not any(re.fullmatch(r".*\.dist-info/licenses/LICENSE", n) or re.fullmatch(r".*\.dist-info/LICENSE", n) for n in names):
        problems.append("the licence file is not in the wheel")
    ver_in_name = os.path.basename(a.wheel).split("-")[1]
    ver_src = re.search(r'__version__ = "([^"]+)"', z.read("vllm_tbccl/__init__.py").decode()) if "vllm_tbccl/__init__.py" in names else None
    if not ver_src or ver_src.group(1) != ver_in_name:
        problems.append(f"wheel version {ver_in_name} != vllm_tbccl/__init__.py {ver_src.group(1) if ver_src else None}")
    if True:
        meta_ver = re.search(r"^Version: (.+)$", text, re.M)
        if not meta_ver or meta_ver.group(1) != ver_in_name:
            problems.append(f"metadata version {meta_ver.group(1) if meta_ver else None} != file name version {ver_in_name}")
    for n in names:
        if n.startswith(DEV_PREFIXES) or n.endswith((".pyc", ".o", ".a", ".c", ".cpp", ".h")):
            problems.append(f"development file in the wheel: {n}")
        if not n.endswith(".so"):
            private = re.findall(rb"/(?:home|Users|mnt|tmp)/[\w.\-]+", z.read(n))
            if private:
                problems.append(f"{n} contains private or build paths: {sorted(set(x.decode() for x in private))}")

    report["problems"] = problems
    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
