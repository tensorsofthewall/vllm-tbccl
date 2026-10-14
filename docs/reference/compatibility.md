# Compatibility

| | Value |
|---|---|
| Package version | 0.2.0rc2 (release candidate; no final release has been published) |
| Python | 3.12 and 3.13 (`requires-python = ">=3.12,<3.14"`; wheels `cp312` and `cp313`). The CUDA and Metal pairing runs on Python 3.12 on the Mac (vllm-metal 0.31.0 and vLLM's macOS wheel); use the same minor on both hosts |
| vLLM | **0.31.0** for both pairings; `SUPPORTED_VLLM = ("0.31.0",)`. 0.30.0 was the validated version of 0.2.0rc1 and is historical |
| PyTorch | 2.13.x (the bundled native module is tied to one torch minor series) |
| TBCCL | C ABI 1, wire protocol 4 (bundled; statically linked) |
| vllm-metal | needed for the Metal pairing only: the fork `tensorsofthewall/vllm-metal` at tag `v0.31.0-tbccl1` (commit `fa4f934586bc27ac974a54183237594d1dc28fc5`), which is vllm-metal **0.31.0** (upstream commit `e7a0c90bec180e5dd4275c473fd7a36e709a930e`) plus the generic transport seam; a reference copy of that change ships as `vllm_tbccl/patches/vllm-metal-0001-pluggable-pp-transport.patch` (SHA-256 `af950dfc4da283efcbbd7e55dcf28d1ffe6d7d608ded034a6888ea56844b3756`; [patch workflow](../guides/patches.md)); MLX 0.32.1 |

## Validated pairings

| Pairing | Stack | Model |
|---|---|---|
| Linux CUDA and Mac CPU, Thunderbolt 4 | vLLM 0.31.0 | Qwen3-0.6B |
| Linux CUDA and Mac Metal, Thunderbolt 4, both pipeline orders | vLLM 0.31.0 and the vllm-metal fork tag `v0.31.0-tbccl1` (0.31.0 plus the transport seam) | Qwen3-0.6B |

Both are pipeline parallelism with two stages (`PP=2`, `TP=1`), validated on the physical link at TBCCL wire protocol 4 with CPython 3.12 (the CUDA and Metal pairing also once with Linux on 3.13). 0.2.0rc1 was validated with vLLM 0.30.0 and has a known row-order defect for CUDA and Metal under concurrent requests, fixed after the release candidate (see the changelog). See [validated configurations](../project/validated-configurations.md) and [validation](../validation/0.2.0.md).

Core transport compatibility (platforms, CUDA, world sizes, the C ABI and wire protocol) is defined by the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html). vllm-tbccl depends only on vLLM and PyTorch at run time (libtbccl is bundled); it does not depend on torch-tbccl or exo-tbccl.

## Wheels

| Wheel | Platform | Needs |
|---|---|---|
| `vllm_tbccl-<version>-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` and `...-cp313-cp313-...` | Linux x86-64 | vLLM from PyPI and the CUDA 13 build of PyTorch 2.13 |
| `vllm_tbccl-<version>-cp312-cp312-macosx_14_0_arm64.whl` and `...-cp313-cp313-...` | macOS 14 or newer, Apple silicon | vLLM for macOS (and vllm-metal for the Metal pairing) |
| `vllm_tbccl-<version>.tar.gz` | source distribution (builds against an installed TBCCL) | `TBCCL_ROOT` |

The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `vllm_tbccl/platform.py` and `pyproject.toml`.
