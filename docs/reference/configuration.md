# Environment variables

All variables are read from the environment of each vLLM process (set them on **every** node). None is required except `VLLM_TBCCL_ENABLE`, which turns the plugin on. vLLM prints an `Unknown vLLM environment variable detected` warning for each `VLLM_TBCCL_*` variable; this is harmless.

## Integration

| Variable | Default | Accepted values | Purpose | When to use | Scope |
|---|---|---|---|---|---|
| `VLLM_TBCCL_ENABLE` | unset (plugin inactive) | `1` | activates the `tbccl` platform plugin | always, on every node, to use vllm-tbccl | per process, read at plugin load |
| `TBCCL_LOCAL_ENDPOINT` | unset | `<host>:0` (the port must be 0) | the address a rank listens on; every communicator picks its own free port pair on that host | every node, with the node's link address | per process, read when a communicator is created (read by the bundled backend) |
| `VLLM_HOST_IP` | unset | an IP address | vLLM's own setting for the address a node advertises | every node of a multi-host run | per process (vLLM) |
| `VLLM_TBCCL_BACKEND` | `auto` | `auto`, `torch`, `metal` | `torch`: vllm-tbccl owns the CUDA or CPU platform. `metal`: vllm-metal owns the platform and vllm-tbccl attaches through its hooks. `auto`: chosen by environment (`metal` when vllm-metal is installed) | `metal` on the Mac Metal node; `torch` on a Mac that has vllm-metal but should use the CPU pairing | per process, read at plugin load |
| `VLLM_TBCCL_PROCESS_GROUP` | `native` | `native`, `torch-tbccl` | `native`: the c10d backend bundled in this package. `torch-tbccl`: an optional compatibility mode that uses a separately installed torch-tbccl instead (never required) | normally never; only to compare against the torch-tbccl backend | per process |
| `VLLM_TBCCL_PLATFORM` | auto | `cpu`, `cuda` | forces which base platform the plugin subclasses | rarely; when auto-detection picks the wrong one | per process |
| `VLLM_TBCCL_WORKER` | on | `0` | do not swap in the thin TBCCL worker subclasses (a user-chosen worker class is never replaced) | to rule the worker subclasses out when debugging | per process |
| `VLLM_TBCCL_KV_LAYOUTS` | detected | layout names | overrides the KV-cache layouts a worker reports for the heterogeneous layout agreement | only to work around a layout mismatch | per process |
| `VLLM_TBCCL_ALLOW_TORCH_MISMATCH` | unset | `1` | skips the check that the native module was built for the running torch minor series | only if you accept an unvalidated, possibly crashing combination | per process, read at import |
| `VLLM_USE_V2_MODEL_RUNNER` | vLLM default | `0` | a vLLM setting; required on **both** hosts of a mixed pair | always for the validated mixed pairings | per process (vLLM) |

## Metal pairing

| Variable | Default | Accepted values | Purpose | When to use | Scope |
|---|---|---|---|---|---|
| `VLLM_TBCCL_ARCHITECTURE` | from the local model's `config.json` | an architecture name on the codec allowlist (Llama, Qwen3) | overrides the boundary codec's architecture detection | only if detection fails for a compatible model | per process |
| `VLLM_TBCCL_RECV_POOL` | off | `1` | reuses receive buffers for decode-sized receives | optional optimization for Metal receives | per process |
| `VLLM_TBCCL_RECV_POOL_MAX_ROWS` | 64 | integer | bounds the pooled row count | with `VLLM_TBCCL_RECV_POOL=1` | per process |
| `VLLM_METAL_PP_TRANSPORT_CLS`, `VLLM_METAL_DIST_BACKEND`, `VLLM_CPU_GROUP_BACKEND` | set automatically | | the vllm-metal hooks and the control-group backend; set to the vllm-tbccl transport class and `tbccl` for the Metal backend if you have not set them | normally never set by hand | per process |

## Diagnostics

| Variable | Default | Accepted values | Purpose | Scope |
|---|---|---|---|---|
| `VLLM_TBCCL_TRACE` | off | `1` | records per-operation diagnostics in memory | per process |
| `VLLM_TBCCL_TRACE_FILE` | unset | a file path | the file the diagnostics are dumped to by a background thread (`<path>.<pid>.json`) | per process |
| `VLLM_TBCCL_TRACE_PERIOD_S` | 5 | seconds | dump period | per process |
| `VLLM_TBCCL_TRACE_STEPS` | off | `1` | times the Metal model runner's forward build and sampling per step (diagnostic only) | per process |
| `VLLM_TBCCL_CHECKSUM` | off | `1` | computes a byte sum of Metal transfers so sender and receiver can be compared (diagnostic only) | per process |

Test-only variables (for example `VLLM_TBCCL_TEST_MODEL`) are described in [building and testing](../development/testing.md) and are not user configuration.
