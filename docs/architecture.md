# vllm-tbccl architecture

```
vLLM model/executor (mp, nnodes=2, PP=2, TP=1)
   -> GroupCoordinator (pp group)
        generic path (CUDA rank)         CPU-platform fast path (CPU rank)
             |                                   |
             |                       vllm_tbccl.TBCCLDeviceCommunicator
             v                                   v
   torch.distributed (device_group = ProcessGroupTBCCL, cpu_group = gloo)
   -> torch-tbccl -> installed libtbccl -> TCP over Thunderbolt 4
```
* vllm-tbccl: platform plugin (dist_backend + communicator class selection), `TBCCLDeviceCommunicator`, opt-in diagnostics. No libtbccl link, no algorithms.
* torch-tbccl additions for this phase: `ProcessGroup.send/recv`, one-rank groups, `host:0` auto ports.
* gloo carries only vLLM's pickled metadata/control objects; every tensor of model data goes through TBCCL (verified from the ProcessGroupTBCCL trace).
* See `vllm_api_audit.md` for call sites and invariants, `phase46_results.md` for measurements.

## Final stack as validated (Phase 46)
Pinned vLLM 0.30.0, mp executor with `--nnodes 2 --headless`, PP=2/TP=1, `VLLM_USE_V2_MODEL_RUNNER=0`, platform plugin `tbccl` (CUDA on Linux, CPU on Mac), device group = control group = `tbccl`.
vLLM hooks (`scripts/apply_vllm_patch.py`): control-group backend env var, KV-layout intersection, lazy `intermediate_tensors`. Launch: `examples/serve_pp2.sh`, clients `examples/pp_client.py`, `pp_batch.py`.
Heterogeneous-platform blockers found and resolved are tabulated in `phase46_results.md`; unresolved: TP>1 untested (model head count), FP16/BF16 reductions unsupported by TBCCL.
