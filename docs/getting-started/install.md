# Installing vllm-tbccl

vllm-tbccl is a vLLM platform plugin. It installs as a separate package into the environment that holds vLLM, and nothing else of the TBCCL family: the wheel bundles its own native `tbccl` c10d backend, statically linked to [TBCCL](https://github.com/tensorsofthewall/tbccl)'s C ABI. It does not use or require torch-tbccl. It owns no transport or algorithm.

## Supported combinations

| Pairing | vLLM | Notes |
|---|---|---|
| CUDA node and CPU node | **0.31.0** | validated on a Linux CUDA host and a Mac CPU host over Thunderbolt 4 |
| CUDA node and Metal node | **0.30.0** | needs [vllm-metal](https://github.com/vllm-project/vllm-metal) and the transport patch in `vllm_tbccl/patches/`; validated on a Linux CUDA host and a Mac Metal host over Thunderbolt 4 |

Both combinations use pipeline parallelism with two stages (`PP=2`, `TP=1`) and two-rank groups. The package version is 0.2.0.dev0 (development; no release published). `SUPPORTED_VLLM` in `vllm_tbccl/platform.py` is `("0.30.0", "0.31.0")`; other vLLM versions warn once when the plugin activates and nothing is claimed for them. See [Compatibility](../reference/compatibility.md).

vllm-metal's current head does not run on vLLM 0.31.0, so the Metal pairing uses 0.30.0.

## Steps

```sh
uv venv .venv && . .venv/bin/activate
uv pip install vllm==<0.31.0 or 0.30.0>        # the version for your pairing
# build the native module against an installed TBCCL prefix (a release wheel carries it prebuilt):
TBCCL_ROOT=<installed tbccl prefix> uv pip install --no-build-isolation --no-deps -e .
```

Use one torch version in the whole environment (vLLM pins the version; the validated one is PyTorch 2.13.0), and never use `--reinstall`, which rewrites torch. For the Metal pairing also install vllm-metal at the validated upstream commit with the patch applied:

```sh
git -C <vllm-metal checkout> am "$(python -m vllm_tbccl.patches --path vllm-metal-0001-pluggable-pp-transport.patch)"
```

The patch adds a generic pluggable pipeline-transport seam to vllm-metal without changing its default behavior.

## Rebuild after TBCCL changes

After installing a TBCCL with a different wire protocol version, rebuild vllm-tbccl against it; ranks built against different wire versions cannot connect. Check an installation with `python -m vllm_tbccl.info`.
