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
