# Compatibility

| | Value |
|---|---|
| Package version | 0.2.0rc1 (release candidate; no final release has been published) |
| Python | 3.13 only (`requires-python = ">=3.13,<3.14"`; the wheels are `cp313`) |
| vLLM | 0.31.0 (CUDA and CPU pairing), 0.30.0 (CUDA and Metal pairing); `SUPPORTED_VLLM = ("0.30.0", "0.31.0")` |
| PyTorch | 2.13.x (the bundled native module is tied to one torch minor series) |
| TBCCL | C ABI 1, wire protocol 4 (bundled; statically linked) |
| vllm-metal | needed for the Metal pairing only; validated at commit `d8cb131` carrying the transport seam from `vllm_tbccl/patches/` ([patch workflow](../guides/patches.md)) |

## Validated pairings

| Pairing | Stack | Model |
|---|---|---|
| Linux CUDA and Mac CPU, Thunderbolt 4 | vLLM 0.31.0 | Qwen3-0.6B |
| Linux CUDA and Mac Metal, Thunderbolt 4, both orientations | vLLM 0.30.0 and vllm-metal with the transport patch | Qwen3-0.6B |

Both are pipeline parallelism with two stages (`PP=2`, `TP=1`). The CUDA and Metal pairing was re-validated on the physical link with the 0.2.0rc1 wheels (wire protocol 4). The physical validation of the CUDA and CPU pairing was recorded with an earlier TBCCL wire protocol; after the move to wire protocol 4 the test suites were re-run, but the physical CUDA and CPU runs were not repeated. See [validated configurations](../project/validated-configurations.md) and [validation](../validation/0.2.0.md).

Core transport compatibility (platforms, CUDA, world sizes, the C ABI and wire protocol) is defined by the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html). vllm-tbccl depends only on vLLM and PyTorch at run time (libtbccl is bundled); it does not depend on torch-tbccl or exo-tbccl.

## Wheels

| Wheel | Platform | Needs |
|---|---|---|
| `vllm_tbccl-0.2.0rc1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | Linux x86-64 | vLLM from PyPI and the CUDA 13 build of PyTorch 2.13 |
| `vllm_tbccl-0.2.0rc1-cp313-cp313-macosx_14_0_arm64.whl` | macOS 14 or newer, Apple silicon | vLLM for macOS (and vllm-metal for the Metal pairing) |
| `vllm_tbccl-0.2.0rc1.tar.gz` | source distribution (builds against an installed TBCCL) | `TBCCL_ROOT` |

The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `vllm_tbccl/platform.py` and `pyproject.toml`.
