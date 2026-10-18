# How do I know TBCCL is being used?

A silent fallback to another transport (Gloo, NCCL, vllm-metal's MLX ring) would invalidate a benchmark or a debugging session. Use this page when the backend does not seem to be active (no `Platform plugin tbccl is activated` line, or a different `backend=`), and to show that your pipeline runs on TBCCL.

## 1. The plugin is active

Start vLLM with `VLLM_TBCCL_ENABLE=1`. The log of every vLLM process contains:

```text
INFO ... [__init__.py:54] - tbccl -> vllm_tbccl.platform:tbccl_platform_plugin
INFO ... [__init__.py:272] Platform plugin tbccl is activated
```

The first line means vLLM found the plugin; the second means it is enabled. If you see the first but not the second, `VLLM_TBCCL_ENABLE=1` is not set in that process. vLLM also prints harmless `Unknown vLLM environment variable detected: VLLM_TBCCL_...` warnings for vllm-tbccl's own variables.

On a Mac with vllm-metal, `Platform plugin metal is activated` is correct for the Metal pairing (`VLLM_TBCCL_BACKEND=metal`): vllm-metal owns the platform and vllm-tbccl attaches through its transport hooks.

## 2. vLLM initialised its distributed state with the tbccl backend

The worker log shows the backend vLLM chose:

```text
(Worker pid=...) INFO ... [parallel_state.py:...] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://192.168.3.2:... backend=tbccl
```

`backend=tbccl` is the evidence that vLLM's device group is the `tbccl` backend. Any other value (`gloo`, `nccl`) means the plugin did not take over.

## 3. The installation

```sh
python -m vllm_tbccl.info
```

shows `"process_group": "native"` (the bundled backend, the default), `"backend": "tbccl"` and the TBCCL versions. `"registered"` is `false` in a plain process and becomes true inside a vLLM process once the plugin activated. If it says `"torch-tbccl"`, `VLLM_TBCCL_PROCESS_GROUP=torch-tbccl` selected the optional compatibility mode that uses a separately installed torch-tbccl.

## 4. Operation counters and traces

- Inside a worker process, `vllm_tbccl._backend.op_stats()` returns `{operation: {"count", "bytes"}}` submitted to libtbccl by the bundled backend (an internal helper, useful from a debugger or a test); it is empty in the torch-tbccl compatibility mode.
- `VLLM_TBCCL_TRACE=1` with `VLLM_TBCCL_TRACE_FILE=<path>` records diagnostics in memory and dumps them from a background thread. For the Metal pairing it records each transfer; add `VLLM_TBCCL_CHECKSUM=1` to compare byte sums of the sender and receiver. Per-operation timelines of the generic send/recv path need the optional torch-tbccl compatibility mode.

## For benchmarking

Record `python -m vllm_tbccl.info` from every node and the two log lines above next to your results. Compare against `backend=tbccl`, not against an assumption.
