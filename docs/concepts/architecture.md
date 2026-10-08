# Architecture

```
vLLM (device groups, PP)  ->  vllm-tbccl (platform plugin + private c10d backend "tbccl")  ->  libtbccl C ABI (statically linked)
```

vllm-tbccl is an out-of-tree vLLM platform plugin. It owns no transport, no algorithm and no staging; if something is missing it belongs in TBCCL. It does not depend on torch-tbccl: the wheel bundles a minimal `c10d::Backend` (`csrc/`) that maps exactly the operations vLLM uses (send/recv, all-reduce SUM, broadcast, all-gather, 2-rank gather, barrier, abort) onto libtbccl's C ABI. The two projects are peers over libtbccl. It registers through the `vllm.platform_plugins` entry point (`tbccl = vllm_tbccl.platform:tbccl_platform_plugin`) and is active only when `VLLM_TBCCL_ENABLE=1`.

## Modules

| Module | Role |
|---|---|
| `platform.py` | platform subclass of the host's CUDA or CPU platform, scoped control-group wrapper, supported vLLM versions |
| `communicator.py` | `TBCCLDeviceCommunicator`: device communicator for pipeline groups |
| `worker.py`, `worker_cuda.py`, `worker_cpu.py` | thin worker subclasses (heterogeneous KV-layout agreement, CPU pipeline buffer) |
| `_backend.py`, `_C` | registration of the bundled `tbccl` c10d backend (native module built from `csrc/`), ABI/torch checks, `python -m vllm_tbccl.info` |
| `peer.py`, `wire.py` | peer description and tensor-dict wire handling for heterogeneous stages |
| `backends/` | the Metal pairing: transport, boundary codecs, receive-buffer pool |
| `diagnostics.py` | in-memory tracing with an off-path dump thread |

## Dual backend

`VLLM_TBCCL_BACKEND=auto|torch|metal`: `torch` means vllm-tbccl owns the CUDA or CPU platform; `metal` means vllm-metal owns the Metal platform and vllm-tbccl attaches through vllm-metal's generic hooks ([CUDA and Metal](../guides/cuda-metal.md)).

## No patches to vLLM or vllm-metal

vLLM is not modified: every adaptation uses a public extension point (the platform plugin entry point, `worker_cls`, a scoped `new_group` wrapper). The single change vllm-metal needs, a generic pluggable pipeline-transport seam with unchanged default behavior, is carried as a patch file ({doc}`../adr/0001-no-in-place-patches`).
