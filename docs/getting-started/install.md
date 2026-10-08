# Installing vllm-tbccl

vllm-tbccl installs as a separate package into the environment that holds vLLM, and nothing else of the TBCCL family: the wheel bundles its own native `tbccl` c10d backend, statically linked to TBCCL's C ABI. It does not use or require torch-tbccl. You do not install tbccl separately.

## Supported combinations

| Pairing | vLLM | Notes |
|---|---|---|
| CUDA node and CPU node | **0.31.0** | validated on a Linux CUDA host and a Mac CPU host over Thunderbolt 4 |
| CUDA node and Metal node | **0.30.0** | needs [vllm-metal](https://github.com/vllm-project/vllm-metal) with the transport patch shipped in this package ([patch workflow](../guides/patches.md)); validated on a Linux CUDA host and a Mac Metal host over Thunderbolt 4 |

Both use pipeline parallelism with two stages (`PP=2`, `TP=1`) and two-rank groups. `SUPPORTED_VLLM` in `vllm_tbccl/platform.py` is `("0.30.0", "0.31.0")`; other vLLM versions warn once when the plugin activates and nothing is claimed for them. vllm-metal's current head does not run on vLLM 0.31.0, so the Metal pairing uses 0.30.0. See [compatibility](../reference/compatibility.md).

## Requirements

| | Linux x86-64 (CUDA node) | macOS arm64 (CPU or Metal node) |
|---|---|---|
| Python | **3.13 only** (the wheel is `cp313`) | **3.13 only** |
| PyTorch | **2.13.x, the CUDA 13 build** (`+cu130`); vLLM pins the exact version | 2.13.x |
| vLLM | 0.31.0 or 0.30.0, from PyPI | 0.31.0 (CPU) or 0.30.0 (Metal), built for macOS as vllm-metal's installer does |
| Wheel | `manylinux_2_27_x86_64` | `macosx_14_0_arm64` |

```{warning}
The Linux wheel is linked against `libtorch_cuda.so` and `libc10_cuda.so`: it needs the **CUDA 13 build of PyTorch**, which is what vLLM installs when you use the `cu130` index below. A CPU-only PyTorch for Linux does not work. Use one torch version in the whole environment, install the wheel with `--no-deps` and never use `--reinstall`, which rewrites torch.
```

## Install the release-candidate wheel

The 0.2.0rc1 wheels and sdist are attached to the GitHub pre-release [`v0.2.0rc1`](https://github.com/tensorsofthewall/vllm-tbccl/releases/tag/v0.2.0rc1). They are not published on PyPI or TestPyPI.

**Linux CUDA node:**

```sh
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install vllm==0.30.0 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match       # 0.31.0 for the CPU pairing
uv pip install --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0rc1/vllm_tbccl-0.2.0rc1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
python -m vllm_tbccl.info
```

**Mac node:** install vLLM for macOS (for the Metal pairing, vLLM 0.30.0 and vllm-metal, following the vllm-metal project's own instructions), then:

```sh
uv pip install --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.0rc1/vllm_tbccl-0.2.0rc1-cp313-cp313-macosx_14_0_arm64.whl
python -m vllm_tbccl.info
```

For the Metal pairing also apply the vllm-metal transport patch ([patch workflow](../guides/patches.md)).

## Check the installation

```sh
python -m vllm_tbccl.info
```

prints the versions of every layer as JSON: the package, torch (running and built-with), the bundled libtbccl, the **C ABI** and **wire protocol**, the devices, and `process_group` (`native` by default). `registered` is `false` in a plain Python process: the `tbccl` backend is registered when the plugin activates inside vLLM. The libtbccl version shown is the package version (`0.6.0`; the release-candidate suffix is not part of that field). Every machine of a job must report the same wire protocol. Then `VLLM_TBCCL_ENABLE=1 vllm --version` logs `Platform plugin tbccl is activated` (vLLM lists the plugin under "Available plugins" in any case; the activation line appears only when `VLLM_TBCCL_ENABLE=1`).

## Build from source

Building is needed only without a matching wheel or to change the code. Native module and TBCCL are build-time inputs:

```sh
uv venv --python 3.13 .venv && . .venv/bin/activate
uv pip install vllm==0.31.0 ...            # as above, for your pairing
TBCCL_ROOT=<installed tbccl prefix> uv pip install --no-build-isolation --no-deps -e .
```

`TBCCL_ROOT` is an installed TBCCL prefix (a [release archive](https://tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) works). A wheel needs no TBCCL install at all.

## After upgrading TBCCL

Because the wheel bundles TBCCL, upgrading means installing a newer vllm-tbccl wheel on **every** machine of the job. Ranks built against different wire protocols cannot connect; check with `python -m vllm_tbccl.info` on each.
