# Code map

Where to start reading, by topic. It is a map to the directories and the first files, not a walk through the source; the rules for the code base are in `AGENTS.md` and the design is in the [architecture](../concepts/architecture.md) page. Every path below is checked by a test, so a rename that leaves this page stale fails the documentation checks.

```{table}
:class: tb-stack tb-stack-map

| Topic | Start at | Read with |
|---|---|---|
| Platform plugin and backend selection (`auto`, `torch`, `metal`) | `vllm_tbccl/platform.py`, `vllm_tbccl/__init__.py` | [Configuration](../reference/configuration.md), [CUDA and Metal pairing](../guides/cuda-metal.md) |
| The device communicator | `vllm_tbccl/communicator.py`, `vllm_tbccl/peer.py` | [Pipeline parallelism](../guides/pipeline-parallelism.md) |
| The native c10d backend | `vllm_tbccl/_backend.py`, `csrc/process_group.cpp`, `csrc/bindings.cpp` | [Architecture](../concepts/architecture.md) |
| Pipeline communication and its wire format | `vllm_tbccl/wire.py`, `vllm_tbccl/roworder.py`, `tests/test_wire_vllm_parity.py` | [Pipeline placement](../benchmarks/placement.md) |
| CUDA, CPU and Metal placement | `vllm_tbccl/worker.py`, `vllm_tbccl/worker_cuda.py`, `vllm_tbccl/worker_cpu.py`, `vllm_tbccl/backends/metal.py` | [CUDA and Metal pairing](../guides/cuda-metal.md) |
| Boundary codecs per model architecture | `vllm_tbccl/backends/codecs.py` | [Limitations](../project/limitations.md) |
| The vllm-metal transport seam and its patch | `vllm_tbccl/patches/vllm-metal-0001-pluggable-pp-transport.patch`, `tests/test_patches.py` | [Patch workflow](../guides/patches.md) |
| Independence from other adapters | `tests/test_independence.py` | [Where should this change go?](where-should-this-change-go.md) |
| Diagnostics and tracing | `vllm_tbccl/info.py`, `vllm_tbccl/diagnostics.py`, `tools/collect_diagnostics.py` | [Verify the backend](../guides/verify-active.md) |
| Two-host launch scripts and clients | `examples/serve_pp2.sh`, `examples/pp_client.py`, `scripts/cross_host_engine.sh` | [Quickstart](../getting-started/quickstart.md) |
| Compatibility facts | `compatibility.json`, `tools/check_compatibility_manifest.py`, `tools/check_framework_pin.py` | [Compatibility](../reference/compatibility.md) |
```

The runtime itself (transport, algorithms, protocol, the C ABI) is not in this repository; its [code map](https://tbccl.tensorsofthewall.com/en/latest/development/code-map.html) lives in TBCCL. A change that needs one of those belongs there ([where should this change go?](where-should-this-change-go.md)).
