# vllm-tbccl

Out-of-tree vLLM integration that carries vLLM's **device-group** communication over
[torch-tbccl](../torch-tbccl) (`ProcessGroupTBCCL`) and therefore over TBCCL, with no NCCL data path.
It owns no transport and no algorithm and never links libtbccl.

Status: **vLLM 0.31.0, unmodified** (PP=2 / TP=1, 2-rank groups): install the plugin, set `VLLM_TBCCL_ENABLE=1`; no vLLM patch is needed (supported tuple vllm-tbccl 0.2.0.dev0 + vLLM 0.31.0 + torch-tbccl 0.2.0.dev0 + PyTorch 2.13.0 + libtbccl C ABI 1). Earlier releases used vLLM 0.30.0 with `patches/0001-generic-heterogeneous-hooks.patch` (obsolete for 0.31.0; kept as history) and remain documented in `docs/architecture.md`, `docs/vllm_api_audit.md`.
For CUDA + CPU pairs set `VLLM_USE_V2_MODEL_RUNNER=0` on both hosts.

Enable per process: `VLLM_TBCCL_ENABLE=1` (vLLM then loads the `tbccl` platform plugin: your normal CUDA or CPU
platform with `dist_backend="tbccl"` and `TBCCLDeviceCommunicator`). `TBCCL_LOCAL_ENDPOINT=<host>:0` lets every
communicator pick its own port pair. `VLLM_TBCCL_TRACE=1` records per-operation diagnostics.

Metal + CUDA: an unmodified vLLM **0.30.0** CUDA stage and a current **vllm-metal** (upstream `4e63b19` plus the generic transport seam in `patches/vllm-metal-0001-*.patch`) Metal stage run as one PP=2 pipeline over Thunderbolt;
vllm-metal's current HEAD does not run on vLLM 0.31.0, so this pairing uses 0.30.0 (`SUPPORTED_VLLM` is `("0.30.0", "0.31.0")`). No vLLM patch is needed.
Cross-host engines need the Thunderbolt link (the executor's reverse-direction queues are blocked by the LAN firewall), and a dead headless-node worker is not detected by vLLM's leader (use client timeouts).

Metal: `VLLM_TBCCL_BACKEND=metal` with vllm-metal installed carries vllm-metal's pipeline activations over TBCCL (zero-copy MLX alias) and can pair with a CUDA or CPU
upstream-vLLM stage (boundary codecs: Llama and Qwen3, an explicit allowlist in `vllm_tbccl/backends/codecs.py`; the architecture comes from the local model's `config.json`). Needs the local vllm-metal commit in `patches/vllm-metal-0001-*.patch` and `scripts/apply_vllm_patch.py`.

```sh
uv venv .venv && . .venv/bin/activate
uv pip install vllm==0.31.0           # or 0.30.0 for the Metal pairing
# build and install torch-tbccl against an installed TBCCL prefix, then:
uv pip install -e .
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

The stack is torch-tbccl 0.2.0.dev0, PyTorch 2.13.0, TBCCL C ABI 1 / wire protocol 4. Other versions warn once at plugin activation and nothing is claimed for them. See [compatibility](docs/reference/compatibility.md) and the draft [validation](docs/validation/0.2.0.md).

## Documentation

The documentation is in `docs/` and builds with `make docs`: [getting started](docs/getting-started/index.md), [guides](docs/guides/index.md), [concepts](docs/concepts/index.md), [reference](docs/reference/index.md). Contributing: `CONTRIBUTING.md` and `AGENTS.md`.

## License

No license file has been published yet.
