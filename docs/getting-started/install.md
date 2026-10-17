# Installing vllm-tbccl

vllm-tbccl installs as a separate package into the environment that holds vLLM, and nothing else of the TBCCL family: the wheel bundles its own native `tbccl` c10d backend, statically linked to TBCCL's C ABI. It does not use or require torch-tbccl. You do not install tbccl separately.

## Supported combinations

| Pairing | vLLM | Notes |
|---|---|---|
| CUDA node and CPU node | **0.31.0** | validated on a Linux CUDA host and a Mac CPU host over Thunderbolt 4 |
| CUDA node and Metal node | **0.31.0** | needs the [`tensorsofthewall/vllm-metal`](https://github.com/tensorsofthewall/vllm-metal) at the immutable tag `v0.31.0-tbccl1` (commit `fa4f934586bc27ac974a54183237594d1dc28fc5`, upstream vllm-metal 0.31.0 plus the generic transport seam) ([Mac node](#install-a-wheel)); validated on a Linux CUDA host and a Mac Metal host over Thunderbolt 4 in both pipeline orders |

Both use pipeline parallelism with two stages (`PP=2`, `TP=1`) and two-rank groups. `SUPPORTED_VLLM` in `vllm_tbccl/platform.py` is `("0.31.0",)`; other vLLM versions (including 0.30.0, which the 0.2.0rc1 validation used) warn once when the plugin activates and nothing is claimed for them. See [compatibility](../reference/compatibility.md).

## Requirements

| | Linux x86-64 (CUDA node) | macOS arm64 (CPU or Metal node) |
|---|---|---|
| Python | **3.12 or 3.13** (wheels `cp312` and `cp313`) | **3.12** for the Metal pairing (vllm-metal 0.31.0 and vLLM's macOS wheel are CPython 3.12); 3.12 or 3.13 for the CPU pairing |
| PyTorch | **2.13.x, the CUDA 13 build** (`+cu130`); vLLM pins the exact version | 2.13.x |
| vLLM | 0.31.0, from PyPI | 0.31.0, the prebuilt macOS wheel that vllm-metal's installer uses |
| Wheel | `manylinux_2_27_x86_64` | `macosx_14_0_arm64` |

```{warning}
The Linux wheel is linked against `libtorch_cuda.so` and `libc10_cuda.so`: it needs the **CUDA 13 build of PyTorch**, which is what vLLM installs when you use the `cu130` index below. A CPU-only PyTorch for Linux does not work. Use one torch version in the whole environment, install the wheel with `--no-deps` and never use `--reinstall`, which rewrites torch.
```

## Install a wheel

The 0.2.1 wheels (`cp312` and `cp313`) and sdist are attached to the GitHub release [`v0.2.1`](https://github.com/tensorsofthewall/vllm-tbccl/releases/tag/v0.2.1). They are not published on PyPI or TestPyPI.

```{warning}
0.2.0rc1 (superseded) derived the CUDA and Metal row order from the scheduler output and could exchange activations between concurrent requests; use 0.2.0rc2 or later for that pairing (see the [changelog](https://github.com/tensorsofthewall/vllm-tbccl/blob/main/CHANGELOG.md)).
```

Every command names its environment (`.venv/bin/python`), so nothing needs to be activated, and the environment that holds vLLM is also the one that must hold vllm-tbccl. Both go into the same environment on each node. Replace `0.2.0`/`cp312` for another release or interpreter (the CUDA and CPU pairing also has `cp313` wheels).

**Linux CUDA node (both pairings):**

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.31.0" --torch-backend cu130
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.1/vllm_tbccl-0.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
.venv/bin/python -m vllm_tbccl.info
```

`--torch-backend cu130` makes `uv pip` take PyTorch (2.13.0+cu130, the version vLLM 0.31.0 pins) from the CUDA 13 index. It is a preview feature of uv; if your uv rejects it, the equivalent explicit index is `uv pip install --python .venv/bin/python "vllm==0.31.0" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`.

**Mac node, CUDA and CPU pairing (no vllm-metal fork):**

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.1/vllm_tbccl-0.2.1-cp312-cp312-macosx_14_0_arm64.whl
.venv/bin/python -m vllm_tbccl.info
```

The first wheel is the prebuilt macOS CPU build of vLLM 0.31.0 that vllm-metal's installer also uses.

**Mac node, CUDA and Metal pairing:** install the vllm-metal fork at the immutable tag (vllm-metal 0.31.0 plus the generic transport seam, commit `fa4f934586bc27ac974a54183237594d1dc28fc5`; no patch step) with the installer **in that checkout**. It creates a Python 3.12 environment at `vllm-metal/.venv-vllm-metal`, installs the prebuilt vLLM 0.31.0 macOS wheel and the fork editable, and builds the fork's Metal artifacts (this needs Xcode's command-line tools and Metal toolchain). There is no prebuilt public wheel of the fork: the source checkout below is the supported method, and nothing in these instructions relies on a temporary CI build artifact. Then add vllm-tbccl to that same environment:

```sh
git clone --branch v0.31.0-tbccl1 --depth 1 https://github.com/tensorsofthewall/vllm-metal.git
(cd vllm-metal && ./install.sh)
uv pip install --python vllm-metal/.venv-vllm-metal/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.2.1/vllm_tbccl-0.2.1-cp312-cp312-macosx_14_0_arm64.whl
vllm-metal/.venv-vllm-metal/bin/python -m vllm_tbccl.info
```

```{warning}
The inherited upstream vllm-metal instructions (the Homebrew formula and the `curl ... | bash` installer from the upstream README) install the **unmodified** upstream vllm-metal, which lacks the transport seam, and `./install.sh --stable`/`--dev` outside a checkout does the same. Run `./install.sh` without arguments from the tagged checkout, as above. The fork is pinned by tag, never by branch. The patch shipped in this package is a reference copy of the same change ([patch workflow](../guides/patches.md)).
```

macOS asks, the first time a new Python interpreter listens on the network, whether to allow incoming connections; allow it, otherwise the Mac node waits silently for the pipeline peer.

## Check the installation

```sh
.venv/bin/python -m vllm_tbccl.info
```

(on a Metal Mac use `vllm-metal/.venv-vllm-metal/bin/python`) prints the versions of every layer as JSON: the package, torch (running and built-with), the bundled libtbccl, the **C ABI** and **wire protocol**, the devices, and `process_group` (`native` by default). `registered` is `false` in a plain Python process: the `tbccl` backend is registered when the plugin activates inside vLLM. The libtbccl version shown is the package version (`0.6.0`; the release-candidate suffix is not part of that field). Every machine of a job must report the same wire protocol. Then `VLLM_TBCCL_ENABLE=1 .venv/bin/python -c "from vllm.platforms import current_platform; print(type(current_platform).__name__)"` prints `TbcclCudaPlatform` on a CUDA node (verified on a clean install). `vllm --version` does not activate platforms and shows nothing about the plugin; a running `vllm serve` logs `Platform plugin tbccl is activated`.

## Build from source

For developers only: building is needed only without a matching wheel or to change the code, and is not part of installing the wheel. Native module and TBCCL are build-time inputs:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.31.0" --torch-backend cu130            # as above, for your pairing
TBCCL_ROOT=<installed tbccl prefix> uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
```

`TBCCL_ROOT` is an installed TBCCL prefix (a [release archive](https://tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) works). A wheel needs no TBCCL install at all.

## After upgrading TBCCL

Because the wheel bundles TBCCL, upgrading means installing a newer vllm-tbccl wheel on **every** machine of the job. Ranks built against different wire protocols cannot connect; check with `python -m vllm_tbccl.info` on each.
