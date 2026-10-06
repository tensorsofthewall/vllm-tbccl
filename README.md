# vllm-tbccl

Out-of-tree vLLM integration that carries vLLM's **device-group** communication over
[torch-tbccl](../torch-tbccl) (`ProcessGroupTBCCL`) and therefore over TBCCL, with no NCCL data path.
It owns no transport and no algorithm and never links libtbccl.

Status: **vLLM 0.31.0, unmodified** (PP=2 / TP=1, 2-rank groups): install the plugin, set `VLLM_TBCCL_ENABLE=1`; no vLLM patch is needed (supported tuple vllm-tbccl 0.2.0.dev0 + vLLM 0.31.0 + torch-tbccl 0.2.0.dev0 + PyTorch 2.13.0 + libtbccl C ABI 1). Earlier releases used vLLM 0.30.0 with `patches/0001-generic-heterogeneous-hooks.patch` (obsolete for 0.31.0; kept as history) and remain documented in `docs/architecture.md`, `docs/vllm_api_audit.md`.
For CUDA + CPU pairs set `VLLM_USE_V2_MODEL_RUNNER=0` on both hosts.

Enable per process: `VLLM_TBCCL_ENABLE=1` (vLLM then loads the `tbccl` platform plugin: your normal CUDA or CPU
platform with `dist_backend="tbccl"` and `TBCCLDeviceCommunicator`). `TBCCL_LOCAL_ENDPOINT=<host>:0` lets every
communicator pick its own port pair. `VLLM_TBCCL_TRACE=1` records per-operation diagnostics.

Metal: `VLLM_TBCCL_BACKEND=metal` with vllm-metal installed carries vllm-metal's pipeline activations over TBCCL (zero-copy MLX alias) and can pair with a CUDA or CPU
upstream-vLLM stage (boundary codecs: Llama and Qwen3, an explicit allowlist in `vllm_tbccl/backends/codecs.py`; the architecture comes from the local model's `config.json`). Needs the local vllm-metal commit in `patches/vllm-metal-0001-*.patch` and `scripts/apply_vllm_patch.py`.

```sh
uv venv .venv && . .venv/bin/activate
uv pip install vllm==0.31.0           # or 0.30.0 for the Metal pairing
# build and install torch-tbccl against an installed TBCCL prefix, then:
uv pip install -e .
```

Compatibility: no change was needed for TBCCL 0.4 (N-rank algorithms) or 0.5.0 (nonblocking submission, structured errors, C ABI); the test suite still passes against the
TBCCL 0.5.0 install. This integration is 2-rank only. exo's pipeline integration is a separate project, `../exo-tbccl`.
