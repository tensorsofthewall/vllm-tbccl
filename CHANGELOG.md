# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. 0.2.0 is the first release; it is experimental.

## 0.3.1

Security-maintenance patch release. There is no change to the adapter source (`csrc/`, `vllm_tbccl/`), the supported vLLM 0.31.0 line, the TBCCL C ABI (1) or the wire protocol (4); the version string is the only source difference. It is experimental.

This is **not** a fix for a vulnerability in vllm-tbccl or in 0.3.0. In October 2026 the emSign root affected by the eMudhra intermediate-key compromise was removed from the CA bundles of Python tooling (certifi 2026.10.19, pip 26.2.2, uv 0.13.4). None of the published 0.3.0 wheels contains a CA bundle or TLS code (all of them were searched). What this release changes is the environment the wheels are built in and the TBCCL they bundle: they are rebuilt in a patched manylinux image, with pip 26.2.2 and uv 0.13.5 and with the revoked root blocked in the build container's system bundle, against TBCCL 0.7.1. The 0.3.0 wheels stay as published and are not unsafe to use.

### Installation

- Two CI-built wheels per platform set, named `vllm_tbccl-0.3.1-...`, attached to the GitHub release with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. They are not published on PyPI or TestPyPI. The wheels statically link TBCCL 0.7.1.

### Security

- The documentation build pins certifi 2026.10.19; the release and documentation jobs install pip 26.2.2 from its hash-pinned wheel before their first pip use; the manylinux_2_28 image moves to a digest with pip 26.2.2, certifi 2026.10.19 and uv 0.13.4 (compilers unchanged) and the clean-install test image to Python 3.13.17; `tools/check_trust_dependencies.py` rejects affected pins and TLS-verification bypasses.
- The Metal pairing's installation steps now say to install uv 0.13.4 or later before running the fork's installer: the installer of the immutable fork tag `v0.31.0-tbccl1` (which does not change) installs uv 0.9.18 when no uv is found. The fork's `main` installs uv 0.13.5.
- Use pip 26.2.2 or later, uv 0.13.4 or later and certifi 2026.10.19 or later in the environments where you install the adapter; upgrading the `certifi` package does not patch an older pip. See `SECURITY.md`.

## 0.3.0

The first minor release after 0.2, promoted from release candidate 0.3.0rc1 with no code change. There is no change to the vLLM platform plugin, the pipeline communication, the activation metadata format, the supported vLLM version (0.31.0), the pinned vllm-metal fork revision, PyTorch (2.13.x), the Python versions (3.12 and 3.13), the TBCCL C ABI (1) or the wire protocol (4).

### Installation

- The same wheels (`cp312` and `cp313`, Linux x86-64 and macOS arm64) and sdist as 0.2.x, named `vllm_tbccl-0.3.0-...`, attached to the GitHub release with SPDX SBOMs, `SHA256SUMS` and build-provenance attestations.
- The wheels are built against TBCCL 0.7.0 and statically link it. A 0.2.x wheel keeps the TBCCL it was built with.

### Changed

- The bundled TBCCL 0.7.0 runs the two-rank `all_gather` as a concurrent exchange and offers an opt-in pipelined `all_reduce` (`TBCCL_ALLREDUCE_PIPELINE_BYTES`). The pipeline boundary of vLLM uses point-to-point transfers, so neither changes the cost of an activation transfer; the adapter needs no setting.

### Documentation

- The requirement that `VLLM_USE_V2_MODEL_RUNNER=0` be set on **every** host is stated in the limitations and troubleshooting pages: a mismatch can leave the engine hanging with no error, and the V2 model runner is not supported in the validated configuration.
- The activation-metadata exchange (an 8-byte size message and a pickled payload on the CPU group, then the tensors on the TBCCL group) costs about 190 to 210 microseconds per boundary transfer on a Thunderbolt 4 link. It is unchanged in this release; changing it would end interoperability with vLLM's generic path.

### Compatibility

- vLLM 0.31.0, PyTorch 2.13.x, CPython 3.12 and 3.13, TBCCL 0.6.0 or newer as a build-time minimum (C ABI 1, wire protocol 4). The vllm-metal fork pin is unchanged.

### Known limitations

- The measurements above come from one Linux CUDA / Apple Metal machine pair. PCIe AER reporting was unavailable on the Linux host, so no statement about AER errors is made.
- On a cross-host CUDA and CPU pair, 7 of 8 test prompts matched the single-host reference token for token in both orientations; the one difference is a near-tie in the next-token probabilities.

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
