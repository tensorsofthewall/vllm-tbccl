# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. vllm-tbccl has had no final release. 0.2.0rc1 is a release candidate (pre-release), not production-ready.

## 0.2.0rc1 (release candidate)

First release candidate of the first planned release, 0.2.0. Expect an rc2 if a blocker is found.

### Installation

- CI-built platform wheels (Linux x86-64 manylinux_2_28 for the CUDA-13 PyTorch; macOS arm64 14.0+) and an sdist, attached to the GitHub pre-release and uploaded to TestPyPI, with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The wheel bundles TBCCL 0.6.0rc1 statically.
- vllm-tbccl is standalone and does not require `torch-tbccl`; install it into the environment that holds vLLM (0.30.0 or 0.31.0) and `torch>=2.13,<2.14`.

### Added

- A vLLM platform plugin that carries device-group communication over TBCCL through its own bundled, private `tbccl` c10d backend (`vllm_tbccl._C`, statically linked to libtbccl's C ABI). It does not use or require torch-tbccl.
- `python -m vllm_tbccl.info` reports the installed layers (torch, libtbccl version, C ABI, wire protocol, devices).
- Two-stage pipeline parallelism (`PP=2`, `TP=1`) between a CUDA node and a CPU node with vLLM 0.31.0, unmodified.
- The same between a CUDA node and a Metal node with vLLM 0.30.0 and vllm-metal plus the transport patch in `vllm_tbccl/patches/`.

### Changed

- The package is now a platform wheel (it carries a native module) instead of a pure-Python wheel, and no longer depends on torch-tbccl. `TBCCL_LOCAL_ENDPOINT` must use port 0.

### Fixed

- None.

### Compatibility

- vLLM 0.31.0 (CUDA and CPU pairing) and vLLM 0.30.0 with vllm-metal (CUDA and Metal pairing); see `compatibility.json`.
- TBCCL C ABI 1 and wire protocol 4 (bundled); PyTorch 2.13.x; CPython 3.13.

### Known limitations

- Two-rank groups and `TP=1` only.
- The Metal pairing needs the patch under `vllm_tbccl/patches/` applied to vllm-metal. The patch ships inside the installed package; `python -m vllm_tbccl.patches --path vllm-metal-0001-pluggable-pp-transport.patch` prints its location.
- Experimental: supported vLLM versions are exactly those listed in the compatibility manifest.
