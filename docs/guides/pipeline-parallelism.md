# Pipeline parallelism over TBCCL

The validated topology is two vLLM nodes, one pipeline stage each: a leader node and a headless node (`--nnodes 2 --headless`), with the multiprocessing executor, `PP=2`, `TP=1`, and no async scheduling. The pipeline's device group and vLLM's control group both use the `tbccl` backend.

```
vLLM (mp executor, nnodes=2, PP=2, TP=1)
  GroupCoordinator (pp group)
     CUDA rank: generic path        CPU rank: vllm_tbccl.TBCCLDeviceCommunicator
  torch.distributed: device group = ProcessGroupTBCCL, control group = tbccl (scoped wrapper)
  bundled tbccl c10d backend -> libtbccl (C ABI) -> TCP (Thunderbolt 4)
```

## Real hardware requirements

- Cross-host engines need a direct link such as Thunderbolt 4 between the nodes. The executor's reverse-direction queues are blocked by an ordinary LAN firewall, so a LAN cannot be used as the pipeline link.
- A dead worker on the headless node is not detected by vLLM's leader. Use client timeouts and an external liveness check.
- Peer-failure behavior is validated on loopback only.

## What the plugin changes

The plugin overrides only the distributed backend name and the device communicator class of the platform that vLLM would have chosen anyway (`CudaPlatform` or `CpuPlatform`). Workers, attention backends, kernels and allocators are inherited. Four heterogeneity problems are handled inside the package, without patching vLLM:

| Problem | Handling |
|---|---|
| control groups default to gloo, which does not connect these hosts | a scoped `new_group` wrapper maps gloo to `tbccl` only for callers in `vllm.distributed.parallel_state` |
| the two stages report different KV-cache layouts | thin worker subclasses (selected through vLLM's public `worker_cls`) agree on a layout |
| a CPU eager worker as second stage needs lazily created intermediate tensors | `TbcclCpuWorker` |
| vLLM bakes the sender's device type into tensor metadata, so a CUDA rank cannot talk to a CPU rank | `TBCCLDeviceCommunicator` relabels tensors for the receiving device |

Tensors of model data go through `ProcessGroupTBCCL`; per-operation tracing is available with `VLLM_TBCCL_TRACE=1` ([configuration](../reference/configuration.md)).

## Limits

- Two-rank groups; `PP=2`, `TP=1`. `reduce_scatter` and `gather` raise `NotImplementedError`; tensor parallelism above 1 was not exercised.
- Validated with Qwen3-0.6B; other models and larger sizes have not been validated.
