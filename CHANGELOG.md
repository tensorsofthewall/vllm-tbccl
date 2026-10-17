# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. 0.2.0 is the first release; it is experimental.

## 0.2.1

Patch release. It adds the public vLLM benchmark provider and its documentation. There is no change to the vLLM platform plugin, the pipeline communication, the supported vLLM version (0.31.0), the pinned vllm-metal fork revision, PyTorch (2.13.x), the Python versions, the TBCCL C ABI (1) or the wire protocol (4), and nothing in this release is a performance improvement. The wheels are built against TBCCL 0.6.1 (a patch release of 0.6.0 without runtime change); the minimum TBCCL package stays 0.6.0.

### Installation

- The same wheels (`cp312` and `cp313`, Linux x86-64 and macOS arm64) and sdist as 0.2.0, named `vllm_tbccl-0.2.1-...`, attached to the GitHub release with SPDX SBOMs, `SHA256SUMS` and build-provenance attestations.
- The benchmark provider is **not** in the wheel. It is the `benchmarks/provider` directory of the source of this release (the source archive of the tag `v0.2.1`); the shared runner from the `tbccl` repository runs it with your vLLM environment's interpreter.

### Added

- A benchmark provider (provider protocol 1) for the shared `tbccl` benchmark suite: TTFT, TPOT, inter-token latency and end-to-end latency of streamed requests, throughput at several concurrencies, pipeline-orientation (CUDA first or Metal first) measurements, a single-device reference, strict stream accounting and a correctness gate, with `describe` and `validate` steps.
- Benchmark documentation: the workloads, how they are timed, the measured reference results with their hardware, software and limits, and a quickstart.

### Compatibility

- vLLM 0.31.0, PyTorch 2.13.x, CPython 3.12 and 3.13, TBCCL 0.6.0 or newer (C ABI 1, wire protocol 4). The vllm-metal fork pin is unchanged.

### Known limitations

- The reference benchmark results were measured on one Linux CUDA / Apple Metal machine pair with Qwen3-0.6B (bf16), PP=2, TP=1, and are labelled with the software they used.
- The vllm-metal fork is installed from its pinned revision; see the installation guide for what that build needs on a Mac.

## 0.2.0

First release. Promoted from release candidate 0.2.0rc2 without a runtime change: it carries the CUDA<->Metal row-order fix and the vllm-metal fork dependency of 0.2.0rc2. 0.2.0rc1 must not be used for CUDA and Metal pipelines (see Fixed). The CUDA and CPU pairing was not affected.

### Fixed

- Fixed CUDA<->Metal pipeline row permutation under concurrent requests, slot reuse and chunked prefill. Earlier versions could derive activation row order from the scheduler output rather than the persistent input batch, causing request activations to be exchanged between rows. The sending stage now states the actual order of the rows it sends (request ids and row counts, proportional to the number of active requests) with its activation, and the receiving stage computes the permutation into its own order; stages that disagree about the active requests fail with an error instead of guessing. Both stages must run this version: a peer without the row-order protocol is refused. 0.2.0rc1 was affected; with 12 concurrent requests CUDA rank 0 to Metal rank 1, 6 to 8 completions could differ from the single-rank reference (sequential requests and CUDA<->CPU were not affected). CUDA<->CPU pipelines are unchanged.

### Changed

- The Metal pairing depends on the maintained vllm-metal fork at an immutable tag, `tensorsofthewall/vllm-metal@v0.31.0-tbccl1` (commit `fa4f934586bc27ac974a54183237594d1dc28fc5`: upstream vllm-metal 0.31.0, `e7a0c90b`, plus the generic pipeline-transport seam). There is no patch step any more; the patch file in the package is a reference copy (SHA-256 `af950dfc4da283efcbbd7e55dcf28d1ffe6d7d608ded034a6888ea56844b3756`).
- CPython 3.12 is supported in addition to 3.13 (the vllm-metal 0.31.0 macOS stack is CPython 3.12). The native module is built per Python minor (cp312, cp313).

## 0.2.0rc2 (release candidate)

Supersedes 0.2.0rc1, which must not be used for CUDA and Metal pipelines (see Fixed). The CUDA and CPU pairing was not affected.

### Fixed

- Fixed CUDA<->Metal pipeline row permutation under concurrent requests, slot reuse and chunked prefill. Earlier versions could derive activation row order from the scheduler output rather than the persistent input batch, causing request activations to be exchanged between rows. The sending stage now states the actual order of the rows it sends (request ids and row counts, proportional to the number of active requests) with its activation, and the receiving stage computes the permutation into its own order; stages that disagree about the active requests fail with an error instead of guessing. Both stages must run this version: a peer without the row-order protocol is refused. 0.2.0rc1 was affected; with 12 concurrent requests CUDA rank 0 to Metal rank 1, 6 to 8 completions could differ from the single-rank reference (sequential requests and CUDA<->CPU were not affected). CUDA<->CPU pipelines are unchanged.

### Changed

- The Metal pairing depends on the maintained vllm-metal fork at an immutable tag, `tensorsofthewall/vllm-metal@v0.31.0-tbccl1` (commit `fa4f934586bc27ac974a54183237594d1dc28fc5`: upstream vllm-metal 0.31.0, `e7a0c90b`, plus the generic pipeline-transport seam). There is no patch step any more; the patch file in the package is a reference copy (SHA-256 `af950dfc4da283efcbbd7e55dcf28d1ffe6d7d608ded034a6888ea56844b3756`).
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
