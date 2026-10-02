# vllm-tbccl

Out-of-tree vLLM integration that carries vLLM's **device-group** communication over
[torch-tbccl](../torch-tbccl) (`ProcessGroupTBCCL`) and therefore over TBCCL, with no NCCL data path.
It owns no transport and no algorithm and never links libtbccl.

Status (Phase 46, experimental): vLLM 0.30.0, 2-rank groups only. See `docs/architecture.md`,
`docs/vllm_api_audit.md`, `docs/phase46_results.md`.

Enable per process: `VLLM_TBCCL_ENABLE=1` (vLLM then loads the `tbccl` platform plugin: your normal CUDA or CPU
platform with `dist_backend="tbccl"` and `TBCCLDeviceCommunicator`). `TBCCL_LOCAL_ENDPOINT=<host>:0` lets every
communicator pick its own port pair. `VLLM_TBCCL_TRACE=1` records per-operation diagnostics.
