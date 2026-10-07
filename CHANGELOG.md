# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. vllm-tbccl has not been released: everything below is unreleased.

## Unreleased

Planned for 0.2.0. This section describes the first planned release and changes until it is published.

### Added

- A vLLM platform plugin that carries device-group communication over torch-tbccl.
- Two-stage pipeline parallelism (`PP=2`, `TP=1`) between a CUDA node and a CPU node with vLLM 0.31.0, unmodified.
- The same between a CUDA node and a Metal node with vLLM 0.30.0 and vllm-metal plus the transport patch in `patches/`.

### Changed

- None.

### Fixed

- None.

### Compatibility

- vLLM 0.31.0 (CUDA and CPU pairing) and vLLM 0.30.0 with vllm-metal (CUDA and Metal pairing); see `compatibility.json`.
- Requires torch-tbccl and TBCCL wire protocol 4.

### Known limitations

- Two-rank groups and `TP=1` only.
- The Metal pairing needs the patch under `patches/` applied to vllm-metal; it is not part of the installed package.
- Experimental: supported vLLM versions are exactly those listed in the compatibility manifest.
