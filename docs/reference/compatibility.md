# Compatibility

| | Value |
|---|---|
| Package version | 0.2.0.dev0 (development; no release has been published) |
| Python | 3.10 or newer (`requires-python`); validated with 3.13 |
| vLLM | 0.31.0 (CUDA and CPU pairing), 0.30.0 (CUDA and Metal pairing); `SUPPORTED_VLLM = ("0.30.0", "0.31.0")` |
| torch-tbccl | 0.2.0.dev0, with PyTorch 2.13.0 |
| TBCCL | C ABI 1, wire protocol 4 (through torch-tbccl) |
| vllm-metal | the upstream commit the patch in `vllm_tbccl/patches/` applies to; needed for the Metal pairing only |

## Validated pairings

| Pairing | Stack | Model |
|---|---|---|
| Linux CUDA and Mac CPU, Thunderbolt 4 | vLLM 0.31.0 | Qwen3-0.6B |
| Linux CUDA and Mac Metal, Thunderbolt 4, both orientations | vLLM 0.30.0 and vllm-metal with the transport patch | Qwen3-0.6B |

Both are pipeline parallelism with two stages (`PP=2`, `TP=1`). The physical-link validation of the CUDA and CPU pairing was recorded with an earlier TBCCL wire protocol; after the move to wire protocol 4 the test suites were re-run, but the physical vLLM runs were not repeated. See [validation](../validation/0.2.0.md).

The machine-readable form is `compatibility.json`; `tools/check_compatibility_manifest.py` verifies it against `vllm_tbccl/platform.py` and `pyproject.toml`.
