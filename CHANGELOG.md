# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. vllm-tbccl has not been released: everything below is unreleased.

## Unreleased

Planned for 0.2.0. This section describes the first planned release and changes until it is published.

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
