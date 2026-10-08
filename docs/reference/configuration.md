# Configuration

All variables are read from the environment of each vLLM process. None is required except `VLLM_TBCCL_ENABLE` to turn the plugin on.

## Integration

| Variable | Effect |
|---|---|
| `VLLM_TBCCL_ENABLE=1` | activates the `tbccl` platform plugin in this process |
| `TBCCL_LOCAL_ENDPOINT=<host>:0` | lets every communicator pick its own port pair on the given address (read by the bundled backend; the port must be 0, the kernel chooses) |
| `VLLM_TBCCL_BACKEND=auto\|torch\|metal` | `torch`: vllm-tbccl owns the CUDA or CPU platform; `metal`: vllm-metal owns the platform and vllm-tbccl attaches through its hooks; `auto` (default) chooses by environment |
| `VLLM_TBCCL_PLATFORM=cpu\|cuda` | forces which base platform the plugin subclasses |
| `VLLM_TBCCL_WORKER=0` | do not swap in the thin TBCCL worker subclasses (default on; a user-chosen worker class is never replaced) |
| `VLLM_TBCCL_KV_LAYOUTS` | overrides the KV-cache layouts a worker reports for the heterogeneous layout agreement |
| `VLLM_USE_V2_MODEL_RUNNER=0` | vLLM setting; required on both hosts for a CUDA and CPU pair |

## Metal pairing

| Variable | Effect |
|---|---|
| `VLLM_TBCCL_ARCHITECTURE` | overrides the boundary codec's architecture detection (otherwise read from the local model's `config.json`) |
| `VLLM_TBCCL_RECV_POOL=1` | reuses receive buffers for decode-sized receives (default off) |
| `VLLM_TBCCL_RECV_POOL_MAX_ROWS` | bounds the pooled row count (default 64) |
| `VLLM_METAL_PP_TRANSPORT_CLS`, `VLLM_METAL_DIST_BACKEND`, `VLLM_CPU_GROUP_BACKEND` | set to the vllm-tbccl transport class and `tbccl` automatically for the Metal backend if you have not set them |

## Diagnostics

| Variable | Effect |
|---|---|
| `VLLM_TBCCL_TRACE=1` | records per-operation diagnostics in memory |
| `VLLM_TBCCL_TRACE_FILE` | file the diagnostics are dumped to by a background thread |
| `VLLM_TBCCL_TRACE_PERIOD_S` | dump period in seconds (default 5) |
| `VLLM_TBCCL_TRACE_STEPS=1` | times the Metal model runner's forward build and sampling per step (diagnostic only) |
| `VLLM_TBCCL_CHECKSUM=1` | computes a byte sum of Metal transfers so sender and receiver can be compared (diagnostic only) |
