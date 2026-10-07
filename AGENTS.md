# AGENTS.md

Technical guidance for contributors and coding agents working in this repository. User documentation is in `README.md`; design notes are in `docs/architecture.md` and `docs/vllm_api_audit.md`. Contribution workflow is in `CONTRIBUTING.md`.

## Purpose

`vllm-tbccl` is an out-of-tree vLLM platform integration that carries vLLM's device-group communication, and with vllm-metal also pipeline activations, over `torch-tbccl` and therefore over TBCCL. It owns no transport or algorithm and never links libtbccl.

```
vLLM (device groups / PP)  ->  vllm-tbccl  ->  torch-tbccl (c10d backend "tbccl")  ->  installed libtbccl
```

## Layout

| Path | Role |
|---|---|
| `vllm_tbccl/` | Platform plugin, device communicator, backends and boundary codecs |
| `patches/` | The generic pluggable pipeline-transport patch for vllm-metal (`git am` onto a vllm-metal checkout) and a legacy vLLM hook patch |
| `tests/`, `examples/`, `tools/`, `scripts/` | Tests, probes and clients, helper tools, launch scripts for multi-host runs |

## Architecture boundaries

- Do not add transport, algorithm, staging or reduction code here. If something is missing, identify the missing torch-tbccl or TBCCL API.
- Do not patch vLLM or vllm-metal in place. The one vllm-metal change (a generic pluggable pipeline-transport seam) is carried as a patch file.
- Supported combinations are listed in `README.md` and enforced in `vllm_tbccl/platform.py` (`SUPPORTED_VLLM`). Do not extend them without validation evidence.

## Critical invariants

- Groups are 2-rank; pipeline parallelism is validated with PP=2 and TP=1.
- The control group backend and the device group backend are separate; the plugin installs scoped wrappers instead of modifying vLLM.
- Boundary codecs for the Metal path are an explicit allowlist (`vllm_tbccl/backends/codecs.py`); an unlisted architecture is rejected rather than assumed to behave like another.
- Lifetime and failure behavior come from torch-tbccl and TBCCL: a communicator failure is communicator-wide and there is no recovery.

## Build and test

```sh
uv venv .venv && uv pip install --python .venv/bin/python -e .   # needs vllm and torch-tbccl installed in the same environment
.venv/bin/python -m pytest -q tests
```

- Use one torch version across vLLM, torch-tbccl and this package; do not use `--reinstall`, which rewrites torch.
- `VLLM_TBCCL_ENABLE=1` enables the platform plugin in a process. `TBCCL_LOCAL_ENDPOINT=<host>:0` lets each communicator choose its own port pair on loopback.
- End-to-end and multi-host scripts need a local model; never download model weights. Metal tests skip on Linux. Physical-link runs are opt-in.
- Peer-failure tests are loopback only.

## Code ownership expectations

Changes to `vllm_tbccl/platform.py`, the backend codecs and the patch files need maintainer review.

## Conventions

- Match the surrounding code. Comment only non-obvious constraints.
- Do not commit absolute machine paths or model locations.
- Add explicit paths when staging files.
