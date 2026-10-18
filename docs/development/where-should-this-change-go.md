# Where should this change go?

The TBCCL ecosystem is four repositories with strict layering. Putting a change in the right place keeps the architecture clean. **If the missing thing is generic (every adapter would want it), it belongs in tbccl. If it mentions a framework, it belongs in that framework's adapter.**

| The change is about... | It belongs in |
|---|---|
| transport, collectives, algorithms, bootstrap, the wire protocol, the C ABI, CUDA staging, Metal-shared handling, work and failure semantics, `tbccl-info`, release archives | **tbccl** |
| PyTorch `ProcessGroup` / `Work` behavior, c10d backend registration, dtype and `ReduceOp` mapping, the MPS dispatch registration, DDP behavior, the torch-tbccl wheel | **torch-tbccl** |
| the vLLM platform plugin, the device communicator, worker subclasses, the bundled c10d backend, the vllm-metal transport seam and its patch, the vllm-tbccl wheel | **vllm-tbccl** |
| the exo `MlxTbccl` instance type, the MLX/DLPack bridge, `TbcclPipelineComm`, typed errors, the exo-tbccl wheel | **exo-tbccl** |
| exo discovery, topology, placement, the `PipelineComm` seam, the runner byte exchange | the exo integration (the exo fork / upstream exo), not exo-tbccl |
| vLLM scheduling, the multi-node executor, worker-death detection | upstream **vLLM** |
| the vllm-metal model runner | upstream **vllm-metal** (vllm-tbccl carries one generic patch for the pipeline-transport seam) |
| `torch.distributed` internals, the MPS backend, Metal assertions | upstream **PyTorch** |
| MLX array behavior and device backends | upstream **MLX** |

## Layering rules

```text
libtbccl must not depend on any framework.
Adapters may depend on libtbccl.
Adapters must not require other adapters.
```

- A transport, algorithm, chunk-scheduling or staging change never goes into an adapter. If an adapter seems to need one, identify the missing generic libtbccl API instead.
- An adapter is not patched around a libtbccl defect: reproduce it outside the framework, fix it in tbccl as its own change, then update the adapter.
- A framework concept (a tensor type, a model name, a process-group class) never goes into tbccl.

## Examples

| Request | Where | Why |
|---|---|---|
| "all_reduce is slow for 8 MiB tensors" | **tbccl** | algorithm selection and chunking are runtime behavior |
| "vLLM hangs when the headless worker dies" | upstream **vLLM** | the multi-node executor has no liveness check; there is no TBCCL frame in the stuck stacks |
| "support another model architecture on the Metal node" | this adapter (a boundary codec) | codecs are per architecture and live here |
| "vllm-metal needs another hook" | upstream **vllm-metal**, with a generic seam if the change is about the pipeline transport | this adapter carries one generic patch, not model-runner changes |
| "the platform plugin should pick the Mac backend automatically" | this adapter | backend selection is plugin behavior (`VLLM_TBCCL_BACKEND`) |
| "call torch-tbccl's process group from here" | not allowed | adapters must not require each other (`tests/test_independence.py`) |
| "patch vLLM in place" | not here | the adapter attaches through public extension points only |

## What does not belong in the core runtime

A vLLM, vllm-metal, model, layer or placement concept; scheduling; worker lifecycle; and anything about boundary codecs. Generic behavior that every adapter would want does go to tbccl.

## How this adapter depends on TBCCL without depending on other adapters

The wheel bundles its own native `tbccl` c10d backend, statically linked to TBCCL's C ABI. It does not use or require torch-tbccl (`VLLM_TBCCL_PROCESS_GROUP=torch-tbccl` is an optional compatibility mode for an environment that already has it) and installing it installs no other adapter. The shared contract is the C ABI version and the wire protocol version; a peer with a different wire protocol version is rejected at connect time ([versioning](https://tbccl.tensorsofthewall.com/en/latest/reference/versioning.html)).

## Documentation

Document a feature in the repository that owns it. The ecosystem overview and the compatibility matrix live in tbccl and are linked from the adapters; do not copy them. When you change a version, an interface or a supported configuration, update that repository's `compatibility.json` and the pages that quote it in the same pull request.
