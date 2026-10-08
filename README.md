# vllm-tbccl

vllm-tbccl is an out-of-tree [vLLM](https://github.com/vllm-project/vllm) platform plugin that carries vLLM's device-group communication, and with vllm-metal also its pipeline activations, over [TBCCL](https://github.com/tensorsofthewall/tbccl) through its own bundled `tbccl` c10d backend. It lets one vLLM pipeline span machines with different accelerators, for example a Linux host with an NVIDIA GPU and a Mac, joined by a direct Thunderbolt 4 link or any TCP network, with no NCCL data path. It is self-contained: the wheel carries a small native c10d backend statically linked to libtbccl's stable C ABI. It owns no transport or algorithm, and it does not use or require torch-tbccl.

> **Status:** release candidate **0.2.0rc1** (pre-release, experimental; not production-ready). The final 0.2.0 has not been released. vllm-tbccl is standalone: it bundles its TBCCL backend and does **not** require `torch-tbccl`.

## What you can use it for

- Two-stage pipeline parallelism (`PP=2`, `TP=1`, two-rank groups) between a **CUDA** node and a **CPU** node with vLLM **0.31.0**, unmodified.
- The same between a **CUDA** node and a **Metal** (Apple GPU) node with vLLM **0.30.0** and vllm-metal plus the transport patch in `vllm_tbccl/patches/`.

## Install

```sh
uv venv .venv && . .venv/bin/activate
uv pip install vllm==0.31.0           # or 0.30.0 for the Metal pairing
# build from source against an installed TBCCL prefix (a release wheel needs no TBCCL install):
TBCCL_ROOT=<installed tbccl prefix> uv pip install --no-build-isolation --no-deps -e .
```

Details, including the vllm-metal patch: [installing](docs/getting-started/install.md).

## Minimal use

```sh
export VLLM_TBCCL_ENABLE=1                 # vLLM loads the "tbccl" platform plugin
export TBCCL_LOCAL_ENDPOINT=<host>:0       # each communicator picks its own port pair
export VLLM_USE_V2_MODEL_RUNNER=0          # on both hosts of a CUDA and CPU pair
```

`examples/serve_pp2.sh` launches a two-node pipeline-parallel server and `examples/pp_client.py` queries it. Real runs need a local model; no script downloads weights.

## Supported configurations

| Pairing | vLLM | Validated model |
|---|---|---|
| Linux CUDA and Mac CPU over Thunderbolt 4 | 0.31.0 | Qwen3-0.6B |
| Linux CUDA and Mac Metal over Thunderbolt 4 | 0.30.0 + vllm-metal with the patch | Qwen3-0.6B |

The stack is PyTorch 2.13.0 and TBCCL C ABI 1 / wire protocol 4. Other versions warn once at plugin activation and nothing is claimed for them. See [compatibility](docs/reference/compatibility.md) and the draft [validation](docs/validation/0.2.0.md).

## Documentation

The documentation is in `docs/` and builds with `make docs`: [getting started](docs/getting-started/index.md), [guides](docs/guides/index.md), [concepts](docs/concepts/index.md), [reference](docs/reference/index.md). Contributing: `CONTRIBUTING.md` and `AGENTS.md`.

The hosted documentation is at https://vllm-tbccl.tensorsofthewall.com/ (development documentation built from `main` until the first release); you can also build it with `make docs`. Related projects and their documentation: `docs/related-projects.md`.

## License

vllm-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.

Security: see `SECURITY.md` and the security model in the documentation. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.
