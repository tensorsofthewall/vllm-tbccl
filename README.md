# vllm-tbccl

[![docs](https://github.com/tensorsofthewall/vllm-tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/vllm-tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/vllm-tbccl?label=release)](https://github.com/tensorsofthewall/vllm-tbccl/releases)

vllm-tbccl is an out-of-tree [vLLM](https://github.com/vllm-project/vllm) platform plugin that lets one vLLM pipeline span machines with different accelerators, for example a Linux host with an NVIDIA GPU and a Mac with Apple silicon, joined by a direct Thunderbolt 4 link or any TCP network, with no NCCL data path. It carries vLLM's device-group communication (and, with [vllm-metal](https://github.com/vllm-project/vllm-metal), its pipeline activations) over [TBCCL](https://github.com/tensorsofthewall/tbccl).

> **Status:** **0.2.0** (first release; experimental). 0.2.0rc1 had a correctness defect for CUDA and Metal pipelines under concurrent requests (row order between the stages); it is fixed since 0.2.0rc2, see the [changelog](CHANGELOG.md). Do not use 0.2.0rc1.

vllm-tbccl is **standalone**: the wheel carries its own small native c10d backend, statically linked to libtbccl's stable C ABI. It owns no transport or algorithm, and it does **not** use or require torch-tbccl (or any other TBCCL adapter).

```text
vLLM  ->  vllm-tbccl (platform plugin + bundled "tbccl" c10d backend)  ->  libtbccl  ->  TCP / Thunderbolt 4
```

## Supported configurations

Two-stage pipeline parallelism (`PP=2`, `TP=1`, two-rank groups) in two pairings. **The validated configuration is PP=2 and TP=1; nothing else is claimed.**

| Pairing | Linux node | Mac node | Needs the vllm-metal fork? |
|---|---|---|---|
| **A. CUDA + CPU** | vLLM 0.31.0, CUDA 13 PyTorch | vLLM 0.31.0 (CPU build for macOS) | no |
| **B. CUDA + Metal** | vLLM 0.31.0, CUDA 13 PyTorch | [`tensorsofthewall/vllm-metal`](https://github.com/tensorsofthewall/vllm-metal) at tag `v0.31.0-tbccl1` (vllm-metal 0.31.0 plus a generic transport seam) | **yes** |

Validated model: Qwen3-0.6B, over Thunderbolt 4. The stack is Python 3.12 (the primary validated interpreter on both hosts; 3.13 wheels exist for the CUDA and CPU pairing), PyTorch 2.13.x and TBCCL C ABI 1 / wire protocol 4. Other versions warn once at plugin activation and nothing is claimed for them. Tensor parallelism above 1 and other models are not validated ([known limitations](docs/project/limitations.md)). The authoritative version table is the [current ecosystem status](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html) (a living page) with the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html); this repository's [compatibility page](docs/reference/compatibility.md) and [validated configurations](docs/project/validated-configurations.md) record what was run for this release.

## Installation

Prebuilt wheels, no compiler, no TBCCL install. Every command names its environment (`.venv/bin/python`), so nothing is activated and it is always clear which environment holds vLLM and vllm-tbccl. vllm-tbccl goes into the **same** environment as vLLM on each node.

**Linux CUDA node (both pairings):**

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.31.0" --torch-backend cu130
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0/vllm_tbccl-0.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
.venv/bin/python -m vllm_tbccl.info
```

**Mac node, pairing A (CPU):**

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0/vllm_tbccl-0.2.0-cp312-cp312-macosx_14_0_arm64.whl
.venv/bin/python -m vllm_tbccl.info
```

**Mac node, pairing B (Metal)**: install the fork at its immutable tag with its own installer from that checkout (it needs Xcode's command-line tools and Metal toolchain; it creates `.venv-vllm-metal` inside the checkout), then add the wheel to that same environment:

```sh
git clone --branch v0.31.0-tbccl1 --depth 1 https://github.com/tensorsofthewall/vllm-metal.git
(cd vllm-metal && ./install.sh)
uv pip install --python vllm-metal/.venv-vllm-metal/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0/vllm_tbccl-0.2.0-cp312-cp312-macosx_14_0_arm64.whl
vllm-metal/.venv-vllm-metal/bin/python -m vllm_tbccl.info
```

Do **not** use the Homebrew formula or the `curl | bash` installer from the upstream vllm-metal README for this pairing: they install the unmodified upstream vllm-metal, which has no transport seam. `--no-deps` keeps the installer from replacing vLLM or PyTorch. Details, source builds (developers only) and troubleshooting: [installing](docs/getting-started/install.md).

**Optional helper.** [`install_adapter.py`](https://tbccl.tensorsofthewall.com/en/latest/getting-started/install-adapters.html) selects and verifies the vllm-tbccl wheel for an environment in which you have already installed vLLM 0.31.0: `python3 install_adapter.py install --adapter vllm --python .venv/bin/python --device cuda` (`cpu` or `mps` on a Mac). It checks the vLLM version and the SHA-256 against the release, installs with `--no-deps`, never installs or changes vLLM, and `--dry-run` prints the manual commands. The commands above keep working without it.

## Verify the installation

`vllm_tbccl.info` (above) prints the package, torch, bundled libtbccl, C ABI and wire protocol as JSON; every machine of a job must report the same wire protocol, and `registered` is `false` in a plain process because the backend registers when the plugin activates inside vLLM. Then check that vLLM selects the TBCCL platform plugin (Linux CUDA node, and the CPU pairing on the Mac):

```sh
VLLM_TBCCL_ENABLE=1 .venv/bin/python -c "from vllm.platforms import current_platform; print(type(current_platform).__name__)"
```

It prints `TbcclCudaPlatform` on a CUDA node (a Mac in the CPU pairing prints `TbcclCpuPlatform`; without `VLLM_TBCCL_ENABLE=1` it prints vLLM's own `CpuPlatform`). Without `VLLM_TBCCL_ENABLE=1` the plugin stays inactive and vLLM picks its own platform. On the Metal Mac vllm-metal owns the platform and vllm-tbccl attaches through the fork's transport seam: run the same check with `vllm-metal/.venv-vllm-metal/bin/python` and expect the vllm-metal platform. `vllm --version` does not activate platforms and shows nothing about the plugin. To confirm a running pipeline really uses TBCCL, see [how do I know TBCCL is being used?](docs/guides/verify-active.md).

## Minimal example: a PP=2 launch

On each node set, in the launching process:

```sh
export VLLM_TBCCL_ENABLE=1                 # vLLM loads the "tbccl" platform plugin
export TBCCL_LOCAL_ENDPOINT=<this host's link address>:0
export VLLM_HOST_IP=<this host's link address>
export VLLM_USE_V2_MODEL_RUNNER=0          # on both hosts of a mixed pair
```

The [quickstart](docs/getting-started/quickstart.md) is a complete, validated two-node PP=2 launch and client call (with `curl` or `examples/pp_client.py`) for both pairings. Real runs need a local model; no script downloads weights.

## Documentation

Hosted documentation: https://vllm-tbccl.tensorsofthewall.com/ (`stable` is the latest final release, numbered versions are earlier releases, and `latest` is development documentation built from `main`). Build it locally with `make docs`. Start with [getting started](docs/getting-started/index.md), the [patch workflow](docs/guides/patches.md), [troubleshooting](docs/guides/troubleshooting.md) and [known limitations](docs/project/limitations.md).

Questions, feedback and what to include in a report: [getting help](https://tbccl.tensorsofthewall.com/en/latest/project/getting-help.html). The projects collect [no telemetry](https://tbccl.tensorsofthewall.com/en/latest/project/privacy.html).

## Benchmarks

Run the vLLM pipeline benchmarks (TTFT, TPOT, ITL, throughput, correctness) on your own hardware, standalone or through the TBCCL benchmark runner: [benchmarks](docs/benchmarks/index.md). Reference measurements from one machine pair, with their limits, are in the [results](docs/benchmarks/results.md).

## Limitations

Pipeline parallelism with two stages only; one validated model; the Metal pairing needs the vllm-metal fork tag `v0.31.0-tbccl1` (vllm-metal 0.31.0 plus the transport seam); a dead worker on the headless node is **not detected by vLLM's leader** (an upstream vLLM limitation, not a TBCCL transport failure). See [known limitations](docs/project/limitations.md).

## Development

[MAINTENANCE.md](MAINTENANCE.md) records the maintenance policy and what needs physical requalification. `CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base; the [development documentation](docs/development/index.md) covers building, testing and where a change belongs. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

vllm-tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.
