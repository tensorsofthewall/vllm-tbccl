# Architecture

```text
vLLM (device groups, PP)  ->  vllm-tbccl (platform plugin + bundled "tbccl" c10d backend)  ->  libtbccl C ABI (statically linked)  ->  TCP / Thunderbolt 4
```

vllm-tbccl is an out-of-tree vLLM platform plugin. It owns no transport, no algorithm and no staging; if something is missing it belongs in TBCCL. It is **standalone**: the wheel bundles a minimal `c10d::Backend` (`csrc/`) that maps exactly the operations vLLM uses (send/recv, all-reduce SUM, broadcast, all-gather, 2-rank gather, barrier, abort) onto libtbccl's C ABI. It does not depend on torch-tbccl or any other adapter; the two are peers over libtbccl. The plugin registers through the `vllm.platform_plugins` entry point (`tbccl = vllm_tbccl.platform:tbccl_platform_plugin`) and is active only when `VLLM_TBCCL_ENABLE=1`.

(An optional compatibility mode, `VLLM_TBCCL_PROCESS_GROUP=torch-tbccl`, uses a separately installed torch-tbccl's backend instead of the bundled one. It is not the default, is never required, and the default architecture above does not involve torch-tbccl.)

## Who owns what

| Layer | Owner | What it does |
|---|---|---|
| vLLM | upstream vLLM | scheduling, the multi-node executor, model runner, the device groups |
| the plugin and communicator | **vllm-tbccl** | selects the `tbccl` backend for device and control groups, relabels tensors across device types, thin worker subclasses |
| the bundled c10d backend | **vllm-tbccl** (`csrc/`) | maps c10d operations to the C ABI; no transport logic |
| vllm-metal | upstream vllm-metal | the Metal platform and model runner; one generic transport seam (patch carried here) |
| libtbccl | **tbccl** | transport, collectives, CUDA staging, failure semantics |

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
| `patches/` | patch files shipped as package data ([patch workflow](../guides/patches.md)) |

## Dual backend

`VLLM_TBCCL_BACKEND=auto|torch|metal`: `torch` means vllm-tbccl owns the CUDA or CPU platform; `metal` means vllm-metal owns the Metal platform and vllm-tbccl attaches through vllm-metal's generic hooks ([CUDA and Metal](../guides/cuda-metal.md)). `auto` chooses by environment (vllm-metal installed selects `metal`).

## Why there are patches, and why this is not a fork

vLLM is not modified: every adaptation uses a public extension point (the platform plugin entry point, `worker_cls`, a scoped `new_group` wrapper). The one change vllm-metal needs, a generic pluggable pipeline-transport seam with unchanged default behavior, cannot be made from outside, so it is carried as a patch file and is meant to be proposed upstream ({doc}`../adr/0001-no-in-place-patches`). A legacy vLLM patch also ships, for history only. Details and the apply/verify procedure: [patch workflow](../guides/patches.md).

## Data path

The activations a stage hands to the next one are CUDA tensors (staged through pinned host memory by libtbccl), CPU tensors, or, on a Mac Metal node, evaluated MLX arrays viewed as zero-copy CPU tensors in unified memory. There is no direct CUDA-to-Metal DMA. See the [TBCCL architecture](https://tbccl.tensorsofthewall.com/en/latest/concepts/architecture.html).
