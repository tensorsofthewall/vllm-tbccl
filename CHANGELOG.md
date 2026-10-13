# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. vllm-tbccl has had no final release. 0.2.0rc1 is a release candidate (pre-release), not production-ready.

## Unreleased (planned 0.2.0)

### Fixed

- Fixed CUDA<->Metal pipeline row permutation under concurrent requests, slot reuse and chunked prefill. Earlier versions could derive activation row order from the scheduler output rather than the persistent input batch, causing request activations to be exchanged between rows. The sending stage now states the actual order of the rows it sends (request ids and row counts, proportional to the number of active requests) with its activation, and the receiving stage computes the permutation into its own order; stages that disagree about the active requests fail with an error instead of guessing. Both stages must run this version: a peer without the row-order protocol is refused. 0.2.0rc1 was affected; with 12 concurrent requests CUDA rank 0 to Metal rank 1, 6 to 8 completions could differ from the single-rank reference (sequential requests and CUDA<->CPU were not affected). CUDA<->CPU pipelines are unchanged.

### Changed

- CPython 3.12 is supported in addition to 3.13 (the vllm-metal 0.31.0 macOS stack is CPython 3.12). The native module is built per Python minor (cp312, cp313).

## 0.2.0rc1 (release candidate)

First release candidate of the first planned release, 0.2.0. Expect an rc2 if a blocker is found.

### Installation

- CI-built platform wheels (Linux x86-64 manylinux_2_28 for the CUDA-13 PyTorch; macOS arm64 14.0+) and an sdist, attached to the GitHub pre-release (not published on PyPI or TestPyPI), with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The wheel bundles TBCCL 0.6.0rc1 statically.
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
