#!/usr/bin/env python3
"""Check the vllm-metal fork pin of vllm-tbccl. Stable releases pin an immutable tag of the maintained fork, never a moving branch.

    check_framework_pin.py                  offline: the pin is well formed, and the reference patch still matches the recorded provenance (header SHA and file checksum)
    check_framework_pin.py --online         additionally: the fork tag exists on GitHub and resolves to the recorded SHA; the pin agrees with the TBCCL ecosystem file
    check_framework_pin.py --patch-diff     additionally: a normalized diff of upstream base -> fork release equals the reference patch (needs git and network)

A tag that resolves to another commit is an INTEGRITY FAILURE; the script never updates the recorded SHA. The reference patch is provenance (a reconstruction of
the fork snapshot), not something users apply.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ECOSYSTEM_URL = "https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/release/ecosystem.json"
UPSTREAM = "vllm-project/vllm-metal"


def normalize_diff(text):
    """The source diff without the e-mail header, the index lines and the format-patch signature, so metadata differences do not matter."""
    i = text.find("diff --git")
    if i < 0:
        return []
    text = re.sub(r"\n-- \n[^\n]*\n*$", "\n", text[i:])
    lines = [l for l in text.splitlines() if not l.startswith("index ")]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def check_offline(manifest, root, rep):
    fw = manifest.get("frameworks", {})
    vllm, metal = fw.get("vllm", {}), fw.get("vllm_metal", {})
    supported = vllm.get("supported")
    if not isinstance(supported, list) or not supported or not all(mc.split_version(v) and not mc.split_version(v)[3] for v in supported):
        rep.add(mc.INTEGRITY, f"frameworks.vllm.supported {supported!r} must be a list of stable versions")
    for key in ("fork", "fork_tag", "fork_revision", "revision", "transport_patch", "transport_patch_sha256"):
        if not metal.get(key):
            rep.add(mc.INTEGRITY, f"frameworks.vllm_metal.{key} is missing")
    if rep.worst() != mc.OK:
        return None
    if not mc.REPO_NAME.match(metal["fork"]):
        rep.add(mc.INTEGRITY, f"fork repository {metal['fork']!r} is not owner/name")
    if re.search(r"(^|/)(main|master|HEAD)$", metal["fork_tag"]):
        rep.add(mc.INTEGRITY, f"fork_tag {metal['fork_tag']!r} is a moving ref")
    for key in ("fork_revision", "revision"):
        if not mc.FULL_SHA.match(metal[key]):
            rep.add(mc.INTEGRITY, f"{key} is not a full 40-character SHA")
    patch = root / "vllm_tbccl" / "patches" / metal["transport_patch"]
    if not patch.exists():
        rep.add(mc.INTEGRITY, f"the reference patch {metal['transport_patch']} is missing")
        return None
    data = patch.read_bytes()
    if hashlib.sha256(data).hexdigest() != metal["transport_patch_sha256"]:
        rep.add(mc.INTEGRITY, "the reference patch does not match transport_patch_sha256")
    head = data.decode().split("\n", 1)[0].split()
    if head[:1] != ["From"] or head[1:2] != [metal["fork_revision"]]:
        rep.add(mc.INTEGRITY, "the reference patch header does not name the recorded fork_revision")
    if rep.worst() == mc.OK:
        rep.ok(f"pin {metal['fork']}@{metal['fork_tag']} -> {metal['fork_revision'][:12]} is well formed; reference patch matches")
    return patch


def check_online(manifest, rep):
    metal = manifest["frameworks"]["vllm_metal"]
    try:
        got = mc.ls_remote_tag(metal["fork"], metal["fork_tag"])
    except mc.Network as e:
        rep.add(mc.WARNING, f"cannot reach GitHub for {metal['fork']}@{metal['fork_tag']} ({e})")
    else:
        if got is None:
            rep.add(mc.INTEGRITY, f"{metal['fork']}@{metal['fork_tag']} no longer exists")
        elif got != metal["fork_revision"]:
            rep.add(mc.INTEGRITY, f"{metal['fork']}@{metal['fork_tag']} resolves to {got}, recorded {metal['fork_revision']}")
        else:
            rep.ok(f"{metal['fork']}@{metal['fork_tag']} resolves to {got}")
    try:
        with urllib.request.urlopen(urllib.request.Request(ECOSYSTEM_URL, headers={"User-Agent": "tbccl-maintenance"}), timeout=30) as r:
            eco = json.load(r)["vllm_tbccl"]
    except (OSError, ValueError, KeyError) as e:
        rep.add(mc.WARNING, f"the TBCCL ecosystem file could not be read ({type(e).__name__})")
        return
    mine = {"metal_repo": metal["fork"], "metal_tag": metal["fork_tag"], "metal_sha": metal["fork_revision"], "metal_upstream_base": metal["revision"]}
    for key, value in mine.items():
        if eco.get(key) != value:
            rep.add(mc.INTEGRITY, f"{key}: ecosystem file says {eco.get(key)!r}, this repository records {value!r}")


def check_patch_diff(manifest, patch, rep):
    metal = manifest["frameworks"]["vllm_metal"]
    with tempfile.TemporaryDirectory(prefix="pin-diff-") as tmp:
        def git(*a):
            return subprocess.run(["git", "-C", tmp, *a], capture_output=True, text=True)
        git("init", "-q")
        for sha in (metal["revision"], metal["fork_revision"]):
            repo = UPSTREAM if sha == metal["revision"] else metal["fork"]
            p = git("fetch", "-q", "--depth", "1", f"https://github.com/{repo}.git", sha)
            if p.returncode:
                rep.add(mc.WARNING, f"could not fetch {sha[:12]} from {repo}: {p.stderr.strip()[:120]}")
                return
        p = git("diff", "--no-color", "--no-ext-diff", metal["revision"], metal["fork_revision"])
        if p.returncode:
            rep.add(mc.WARNING, f"git diff failed: {p.stderr.strip()[:120]}")
            return
    if normalize_diff(p.stdout) == normalize_diff(patch.read_text()):
        rep.ok(f"diff({metal['revision'][:12]}..{metal['fork_revision'][:12]}) equals the reference patch")
    else:
        rep.add(mc.INTEGRITY, "the normalized diff between upstream base and the fork release differs from the reference patch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--patch-diff", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    manifest = mc.load_json(root / "compatibility.json")
    rep = mc.Report("vllm-metal fork pin")
    patch = check_offline(manifest, root, rep)
    if patch and args.online:
        check_online(manifest, rep)
    if patch and args.patch_diff:
        check_patch_diff(manifest, patch, rep)
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
