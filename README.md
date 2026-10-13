# vllm-tbccl

[![docs](https://github.com/tensorsofthewall/vllm-tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/vllm-tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/vllm-tbccl?include_prereleases&label=pre-release)](https://github.com/tensorsofthewall/vllm-tbccl/releases)

vllm-tbccl is an out-of-tree [vLLM](https://github.com/vllm-project/vllm) platform plugin that lets one vLLM pipeline span machines with different accelerators, for example a Linux host with an NVIDIA GPU and a Mac with Apple silicon, joined by a direct Thunderbolt 4 link or any TCP network, with no NCCL data path. It carries vLLM's device-group communication (and, with [vllm-metal](https://github.com/vllm-project/vllm-metal), its pipeline activations) over [TBCCL](https://github.com/tensorsofthewall/tbccl).

> **Status:** release candidate **0.2.0rc1** (pre-release, experimental; not production-ready). The final 0.2.0 has not been released.

vllm-tbccl is **standalone**: the wheel carries its own small native c10d backend, statically linked to libtbccl's stable C ABI. It owns no transport or algorithm, and it does **not** use or require torch-tbccl (or any other TBCCL adapter).

```text
vLLM  ->  vllm-tbccl (platform plugin + bundled "tbccl" c10d backend)  ->  libtbccl  ->  TCP / Thunderbolt 4
```

## What is supported

Two-stage pipeline parallelism (`PP=2`, `TP=1`, two-rank groups) in two pairings:

| Pairing | vLLM | Validated model |
|---|---|---|
| Linux **CUDA** node and a **CPU** node (a Mac) over Thunderbolt 4 | **0.31.0**, unmodified | Qwen3-0.6B |
| Linux **CUDA** node and a Mac **Metal** (Apple GPU) node over Thunderbolt 4 | **0.30.0**, unmodified, plus [vllm-metal](https://github.com/vllm-project/vllm-metal) with the transport patch shipped in this package | Qwen3-0.6B |

The stack is Python 3.13, PyTorch 2.13.x and TBCCL C ABI 1 / wire protocol 4. Other versions warn once at plugin activation and nothing is claimed for them. Tensor parallelism above 1 and other models are not validated ([known limitations](docs/project/limitations.md)).

## Quick start

```sh
uv venv --python 3.13 .venv && . .venv/bin/activate
# Linux: the CUDA 13 build of PyTorch, then vLLM, then the wheel
uv pip install vllm==0.30.0 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0rc1/vllm_tbccl-0.2.0rc1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
python -m vllm_tbccl.info
```

On the node that serves the API (Linux CUDA, node 0) and on the headless node (the Mac, node 1) set, in each process:

```sh
export VLLM_TBCCL_ENABLE=1                 # vLLM loads the "tbccl" platform plugin
export TBCCL_LOCAL_ENDPOINT=<this host's link address>:0
export VLLM_HOST_IP=<this host's link address>
export VLLM_USE_V2_MODEL_RUNNER=0          # on both hosts of a mixed pair
```

[Installing](docs/getting-started/install.md) has every platform; the [quickstart](docs/getting-started/quickstart.md) is a complete, validated two-node PP=2 launch and client call, and [how do I know TBCCL is being used?](docs/guides/verify-active.md) shows how to check that the pipeline really runs on TBCCL. `examples/serve_pp2.sh` launches one node of a CUDA and CPU pair. Real runs need a local model; no script downloads weights.

## Supported configurations

The authoritative version table is the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html); [this repository's compatibility page](docs/reference/compatibility.md) adds the vLLM-specific detail, and the [validated configurations](docs/project/validated-configurations.md) page lists what was actually run.

## Documentation

Hosted documentation: https://vllm-tbccl.tensorsofthewall.com/ (`latest` is development documentation built from `main`; release-candidate versions appear as numbered versions; `stable` does not exist until the final release). Build it locally with `make docs`. Start with [getting started](docs/getting-started/index.md), the [patch workflow](docs/guides/patches.md), [troubleshooting](docs/guides/troubleshooting.md) and [known limitations](docs/project/limitations.md).

## Known limitations

Pipeline parallelism with two stages only; one validated model; the Metal pairing needs vLLM 0.30.0 and a patched vllm-metal; a dead worker on the headless node is **not detected by vLLM's leader** (an upstream vLLM limitation, not a TBCCL transport failure). See [known limitations](docs/project/limitations.md).

## Development

`CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base; the [development documentation](docs/development/index.md) covers building, testing and where a change belongs. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

vllm-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.
