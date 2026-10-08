"""The patch files that ship with vllm-tbccl, discoverable from an installed package.

The Metal pairing (a CUDA node and a Metal node, vLLM 0.30.0) needs ``vllm-metal-0001-pluggable-pp-transport.patch`` applied to a vllm-metal checkout::

    git -C <vllm-metal checkout> am "$(python -m vllm_tbccl.patches --path vllm-metal-0001-pluggable-pp-transport.patch)"

``python -m vllm_tbccl.patches --list`` prints the patch names, ``--path`` the directory and ``--path NAME`` the file.
"""
import pathlib

_DIR = pathlib.Path(__file__).resolve().parent


def directory() -> pathlib.Path:
    return _DIR


def names() -> list:
    return sorted(p.name for p in _DIR.glob("*.patch"))


def path(name: str) -> pathlib.Path:
    p = _DIR / name
    if p.name != name or not p.is_file():
        raise FileNotFoundError(f"no patch named {name!r}; available: {', '.join(names())}")
    return p
