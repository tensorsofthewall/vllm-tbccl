"""The shipped patches are discoverable from the installed package (no source checkout)."""
import subprocess
import sys

import pytest

from vllm_tbccl import patches

METAL = "vllm-metal-0001-pluggable-pp-transport.patch"


def test_the_patches_are_package_data():
    assert METAL in patches.names()
    assert patches.path(METAL).is_file() and patches.path(METAL).parent == patches.directory()


def test_unknown_patch_is_a_clear_error():
    with pytest.raises(FileNotFoundError, match="available"):
        patches.path("nope.patch")
    with pytest.raises(FileNotFoundError):
        patches.path("../pyproject.toml")


def test_command_line_prints_paths():
    out = subprocess.run([sys.executable, "-m", "vllm_tbccl.patches", "--path", METAL], capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(patches.path(METAL))
    listing = subprocess.run([sys.executable, "-m", "vllm_tbccl.patches", "--list"], capture_output=True, text=True, check=True).stdout.split()
    assert METAL in listing
    bad = subprocess.run([sys.executable, "-m", "vllm_tbccl.patches", "--path", "missing.patch"], capture_output=True, text=True)
    assert bad.returncode == 1


def test_the_metal_patch_is_a_well_formed_git_patch():
    p = subprocess.run(["git", "apply", "--stat", str(patches.path(METAL))], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
