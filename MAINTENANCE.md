# Maintenance

The shared maintenance policy of the TBCCL ecosystem (branches, tag immutability, pinning, release cadence, security and correctness fixes, upstream updates, what needs physical requalification, automation and its failure categories) is [MAINTENANCE.md in tbccl](https://github.com/tensorsofthewall/tbccl/blob/main/MAINTENANCE.md); the versioning rules are in [VERSIONING.md](https://github.com/tensorsofthewall/tbccl/blob/main/VERSIONING.md). This file records only what is specific to vllm-tbccl.

## Supported release

vllm-tbccl 0.2.0: vLLM 0.31.0, CPython 3.12 and 3.13 (the CUDA and Metal pairing needs 3.12 on the Mac), TBCCL 0.6.0 (C ABI 1, wire protocol 4), Linux x86-64 (CUDA 13) and macOS arm64. The current ecosystem status (all four projects and their pins) is in the [ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html), built from [`release/ecosystem.json`](https://github.com/tensorsofthewall/tbccl/blob/main/release/ecosystem.json). Documentation built from a release tag describes that release only and is not updated when other projects release.

## Pins

The Metal stage runs `tensorsofthewall/vllm-metal` at the immutable tag `v0.31.0-tbccl1` (`fa4f934586bc27ac974a54183237594d1dc28fc5`, upstream v0.31.0 plus the pluggable PP-transport seam). `python3 tools/check_framework_pin.py` checks the pin offline; `--online` confirms the tag, and `--patch-diff` confirms that the diff from upstream `e7a0c90bec180e5dd4275c473fd7a36e709a930e` to the fork tag still equals the reference patch kept under `vllm_tbccl/patches/`. The patch is provenance, not an install step.

## What needs physical requalification here

Runtime sources of this repository: `csrc/` and `vllm_tbccl/` other than the reference patch. A change to them, to the framework pin or to the supported TBCCL line needs physical requalification: the CUDA<->Metal pipeline-parallel campaign (both orientations, concurrent requests, parity with the single-rank reference) from the public wheels and the pinned vllm-metal tag. Documentation, tests, CI, workflows, release metadata and GitHub Release edits do not. The full matrix is in the shared policy.

## Automation in this repository

`docs` workflow (every pull request, offline): release state, tooling tests, internal links, framework pin. `maintenance` workflow (weekly and by hand): release state, stable release surface, links, documentation hosts and the documentation build, the online framework pin. `release-preflight` workflow (by hand, with the intended tag). The `guard` job of `release-artifacts` refuses a build whose tag, version, manifest, pin or TBCCL line disagree. `tools/verify_public_release.py` verifies a published release from the outside, and `tools/collect_diagnostics.py` prints the environment report requested by the issue template.

## Pin integrity versus current-ecosystem consistency

`tools/check_framework_pin.py --online` reports two separate things. Release pin integrity (the immutable fork tag still resolves to the SHA recorded in `compatibility.json`) fails hard whatever the release state. Current-ecosystem consistency (this repository against `release/ecosystem.json` on tbccl `main`) fails only for the published stable version; for an unreleased candidate it is a warning that the ecosystem file has not yet been updated, and `--ecosystem-file` or `--ecosystem-ref` compares against a candidate ecosystem instead. See the [maintenance policy](https://github.com/tensorsofthewall/tbccl/blob/main/MAINTENANCE.md).
