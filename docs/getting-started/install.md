# Installing vllm-tbccl

vllm-tbccl installs as a separate package into the environment that holds vLLM, and nothing else of the TBCCL family: the wheel bundles its own native `tbccl` c10d backend, statically linked to TBCCL's C ABI. It does not use or require torch-tbccl. You do not install tbccl separately.

## Supported combinations

| Pairing | vLLM | Notes |
|---|---|---|
| CUDA node and CPU node | **0.31.0** | validated on a Linux CUDA host and a Mac CPU host over Thunderbolt 4 |
| CUDA node and Metal node | **0.31.0** | needs the [`tensorsofthewall/vllm-metal`](https://github.com/tensorsofthewall/vllm-metal) at the immutable tag `v0.31.0-tbccl1` (commit `fa4f934586bc27ac974a54183237594d1dc28fc5`, upstream vllm-metal 0.31.0 plus the generic transport seam) ([Mac node](#install-a-wheel)); validated on a Linux CUDA host and a Mac Metal host over Thunderbolt 4 in both pipeline orders |

Both use pipeline parallelism with two stages (`PP=2`, `TP=1`) and two-rank groups. `SUPPORTED_VLLM` in `vllm_tbccl/platform.py` is `("0.31.0",)`; other vLLM versions (including 0.30.0, which the 0.2.0rc1 validation used) warn once when the plugin activates and nothing is claimed for them. See [compatibility](../reference/compatibility.md).

## Which pairing do I need?

You always need a Linux CUDA node as stage 0 or 1; the question is what the Mac node runs.

| | **CUDA and CPU** | **CUDA and Metal** |
|---|---|---|
| The Mac node runs | vLLM's prebuilt macOS **CPU** wheel (0.31.0) | the **vllm-metal fork**, on the Apple GPU |
| Extra requirements | none beyond vLLM | the fork at the immutable tag `v0.31.0-tbccl1` (not upstream vllm-metal), Python 3.12, Xcode 26.2 or newer with its Metal toolchain for `./install.sh` |
| Python on the Mac | 3.12 or 3.13 | 3.12 |
| `VLLM_TBCCL_BACKEND` on the Mac | unset, or `torch` if vllm-metal is also installed there | `metal` |
| Choose it when | you want the simplest Mac install, or the Mac GPU is not needed | the Mac stage should use the GPU |
| Walk-through | [Quickstart](quickstart.md), last section | [Quickstart](quickstart.md) |

Both are validated for PP=2, TP=1 with Qwen3-0.6B ([validated configurations](../project/validated-configurations.md)). Do **not** install an unmodified upstream vllm-metal for the Metal pairing: it lacks the transport seam.

**Which machine runs which command.** The commands under "Install a wheel" are per node: the Linux block on the Linux CUDA node, one of the two Mac blocks on the Mac. Each node has its own Python environment, and in each one **the same environment holds vLLM and vllm-tbccl** (`.venv` on Linux; `.venv` or `vllm-metal/.venv-vllm-metal` on the Mac). Install the same vllm-tbccl version on both nodes.

## Requirements

| | Linux x86-64 (CUDA node) | macOS arm64 (CPU or Metal node) |
|---|---|---|
| Python | **3.12 or 3.13** (wheels `cp312` and `cp313`) | **3.12** for the Metal pairing (vllm-metal 0.31.0 and vLLM's macOS wheel are CPython 3.12); 3.12 or 3.13 for the CPU pairing |
| PyTorch | **2.13.x, the CUDA 13 build** (`+cu130`); vLLM pins the exact version | 2.13.x |
| vLLM | 0.31.0, from PyPI | 0.31.0, the prebuilt macOS wheel that vllm-metal's installer uses |
| Wheel | `manylinux_2_27_x86_64` | `macosx_14_0_arm64` |
| Tools | uv 0.13.4 or later, pip 26.2.2 or later and certifi 2026.10.19 or later (the October 2026 emSign CA removal; see the [security policy](https://tbccl.tensorsofthewall.com/en/latest/project/security-policy.html)). The wheel contains no CA bundle | same |

```{warning}
The Linux wheel is linked against `libtorch_cuda.so` and `libc10_cuda.so`: it needs the **CUDA 13 build of PyTorch**, which is what vLLM installs when you use the `cu130` index below. A CPU-only PyTorch for Linux does not work. Use one torch version in the whole environment, install the wheel with `--no-deps` and never use `--reinstall`, which rewrites torch.
```

## Install a wheel

The 0.3.1 wheels (`cp312` and `cp313`) and sdist are attached to the GitHub release [`v0.3.1`](https://github.com/tensorsofthewall/vllm-tbccl/releases/tag/v0.3.1). They are not published on PyPI or TestPyPI.

```{warning}
0.2.0rc1 (superseded) derived the CUDA and Metal row order from the scheduler output and could exchange activations between concurrent requests; use 0.2.0rc2 or later for that pairing (see the [changelog](https://github.com/tensorsofthewall/vllm-tbccl/blob/main/CHANGELOG.md)).
```

Every command names its environment (`.venv/bin/python`), so nothing needs to be activated, and the environment that holds vLLM is also the one that must hold vllm-tbccl. Both go into the same environment on each node. Replace `0.3.1`/`cp312` for another release or interpreter (the CUDA and CPU pairing also has `cp313` wheels).

### Linux CUDA node (both pairings)


```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "vllm==0.31.0" --torch-backend cu130
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.3.1/vllm_tbccl-0.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
.venv/bin/python -m vllm_tbccl.info
```

`--torch-backend cu130` makes `uv pip` take PyTorch (2.13.0+cu130, the version vLLM 0.31.0 pins) from the CUDA 13 index. It is a preview feature of uv; if your uv rejects it, the equivalent explicit index is `uv pip install --python .venv/bin/python "vllm==0.31.0" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match`.

### Mac node, CUDA and CPU pairing (no vllm-metal fork)


```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.3.1/vllm_tbccl-0.3.1-cp312-cp312-macosx_14_0_arm64.whl
.venv/bin/python -m vllm_tbccl.info
```

The first wheel is the prebuilt macOS CPU build of vLLM 0.31.0 that vllm-metal's installer also uses.

### Mac node, CUDA and Metal pairing: vllm-metal installation

Install uv 0.13.4 or later first (`brew install uv`, or see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)): the `scripts/lib.sh` of the immutable tag installs uv 0.9.18 when it finds no uv, and that version trusts the emSign root revoked in October 2026. With uv already installed the installer leaves it alone. Then install the vllm-metal fork at the immutable tag (vllm-metal 0.31.0 plus the generic transport seam, commit `fa4f934586bc27ac974a54183237594d1dc28fc5`; no patch step) with the installer **in that checkout**. It creates a Python 3.12 environment at `vllm-metal/.venv-vllm-metal`, installs the prebuilt vLLM 0.31.0 macOS wheel and the fork editable, and builds the fork's Metal artifacts (this needs a macOS SDK 26.2 or newer, that is Xcode 26.2 or newer, and its Metal toolchain; the installer downloads the toolchain component with `xcodebuild` if it is missing, and refuses an older SDK before compiling). The fork is installed editable, so the checkout must stay where it is: the compiled paged-attention extension and four `.metallib` files are built into its `vllm_metal/metal/` directory. There is no prebuilt public wheel of the fork: the source checkout below is the supported method, and nothing in these instructions relies on a temporary CI build artifact. Then add vllm-tbccl to that same environment:

```sh
git clone --branch v0.31.0-tbccl1 --depth 1 https://github.com/tensorsofthewall/vllm-metal.git
(cd vllm-metal && ./install.sh)
uv pip install --python vllm-metal/.venv-vllm-metal/bin/python --no-deps https://github.com/tensorsofthewall/vllm-tbccl/releases/download/v0.3.1/vllm_tbccl-0.3.1-cp312-cp312-macosx_14_0_arm64.whl
vllm-metal/.venv-vllm-metal/bin/python -m vllm_tbccl.info
```

This procedure was exercised from an empty directory on a Mac (macOS 26.5, Xcode SDK 26.5): `./install.sh` built `_paged_ops.cpython-312-darwin.so` and the four metallibs (`paged_attention_v2_kern`, `gdn_kern`, `paged_mla_kern`, `paged_attention_nax_kern`), and the environment held vLLM 0.31.0, vllm-metal 0.31.0, MLX 0.32.1 and PyTorch 2.13.0. To check the build of your own installation, without loading a model:

```sh
vllm-metal/.venv-vllm-metal/bin/python - <<'PY'
from vllm_metal.metal import build
print(build.output_path().exists(), all(build.metallib_path(n).exists() for n in (*build.METALLIB_NAMES, build.NAX_METALLIB_NAME)))
PY
```

It prints `True True` when the extension and all four metallibs exist. If an artifact is missing, vllm-metal stops at startup with `Prebuilt native extension not found`; rerun `./install.sh` in the checkout and read its build output.

```{warning}
The inherited upstream vllm-metal instructions (the Homebrew formula and the `curl ... | bash` installer from the upstream README) install the **unmodified** upstream vllm-metal, which lacks the transport seam, and `./install.sh --stable`/`--dev` outside a checkout does the same. Run `./install.sh` without arguments from the tagged checkout, as above. The fork is pinned by tag, never by branch. The patch shipped in this package is a reference copy of the same change ([patch workflow](../guides/patches.md)).
```

macOS asks, the first time a new Python interpreter listens on the network, whether to allow incoming connections; allow it, otherwise the Mac node waits silently for the pipeline peer.

## Check the installation

```sh
.venv/bin/python -m vllm_tbccl.info
```

(on a Metal Mac use `vllm-metal/.venv-vllm-metal/bin/python`) prints the versions of every layer as JSON: the package, torch (running and built-with), the bundled libtbccl, the **C ABI** and **wire protocol**, the devices, and `process_group` (`native` by default). `registered` is `false` in a plain Python process: the `tbccl` backend is registered when the plugin activates inside vLLM. The libtbccl version shown is the package version of the TBCCL that the wheel bundles (`0.7.1` for the 0.3.1 wheels; the release-candidate suffix is not part of that field). Every machine of a job must report the same wire protocol. Then `VLLM_TBCCL_ENABLE=1 .venv/bin/python -c "from vllm.platforms import current_platform; print(type(current_platform).__name__)"` prints `TbcclCudaPlatform` on a CUDA node (verified on a clean install). `vllm --version` does not activate platforms and shows nothing about the plugin; a running `vllm serve` logs `Platform plugin tbccl is activated`.

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
