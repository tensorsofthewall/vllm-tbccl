# Troubleshooting

Find the symptom, check the likely cause with the diagnostic, apply the fix. Start with `python -m vllm_tbccl.info` on every node ([how do I know TBCCL is being used?](verify-active.md)). Each row says whether the cause is in TBCCL, vllm-tbccl or upstream vLLM.

## Installing and starting

| Symptom | Likely cause (owner) | Diagnostic | Fix |
|---|---|---|---|
| `ImportError` about `libtorch_cuda.so` / `libc10_cuda.so` on Linux | a CPU-only PyTorch (the wheel needs the CUDA 13 build) | `python -c "import torch; print(torch.__version__)"` | install vLLM from the `cu130` index ([installing](../getting-started/install.md)) |
| no matching distribution for the wheel | Python is not 3.12 or 3.13, or not Linux x86-64 / macOS arm64 | `python --version` | use Python 3.12 or 3.13 on a supported platform |
| no `Platform plugin tbccl is activated` line | `VLLM_TBCCL_ENABLE=1` missing in that process | check the environment of **each** process | export it on every node |
| a warning about the vLLM version when the plugin activates | vLLM is not 0.31.0 | `pip show vllm` | use a supported version ([compatibility](../reference/compatibility.md)) |
| `ValueError: VLLM_TBCCL_BACKEND must be one of ...` | an invalid backend name (vllm-tbccl) | the message | use `auto`, `torch` or `metal` |
| `ImportError: ... compiled against torch X but torch Y is installed` | the native module is tied to one torch minor series | `python -m vllm_tbccl.info` | install the matching vllm-tbccl or torch |
| a Mac with vllm-metal installed hangs or mismatches backends in a CUDA and CPU pair | `auto` selects `metal`, which declines the CPU platform | the Mac log shows `Platform plugin metal` | set `VLLM_TBCCL_BACKEND=torch` on that Mac |

## Forming the pipeline

| Symptom | Likely cause (owner) | Diagnostic | Fix |
|---|---|---|---|
| `protocol_mismatch ... wire protocol` | wheels on the two hosts bundle different TBCCL wire protocols (TBCCL) | `python -m vllm_tbccl.info` on each host | install the same vllm-tbccl version on both |
| startup assertion in the model runner on a CUDA and CPU pair | `VLLM_USE_V2_MODEL_RUNNER` differs between hosts (vLLM) | the environment of both hosts | set `VLLM_USE_V2_MODEL_RUNNER=0` on both |
| the engine cannot form the cross-host pipeline over a LAN | the multiprocessing executor's reverse-direction queues are blocked by the firewall (vLLM) | try the direct link | use a direct link such as Thunderbolt 4 between the nodes |
| timeout waiting for the other node | wrong `--master-addr`, `VLLM_HOST_IP` or `TBCCL_LOCAL_ENDPOINT` host; node not started; firewall prompt on the Mac unanswered | `ping` the peer; check the prompt | give each node its own link address; start both; answer the prompt, run in a live session |
| the Metal node rejects an architecture | the boundary-codec allowlist covers Llama and Qwen3 only (vllm-tbccl) | the error names the architecture | use a supported architecture or set `VLLM_TBCCL_ARCHITECTURE` only if it is genuinely compatible |
| the vllm-metal patch does not apply | the patch predates your vllm-metal revision | `git apply --check` | [patch workflow](patches.md) |

## While running and stopping

| Symptom | Likely cause (owner) | Diagnostic | Fix |
|---|---|---|---|
| the leader hangs or requests time out after the headless node's worker died | **upstream vLLM**: the leader's multi-node executor has no liveness check on the headless worker; there is no TBCCL frame in the stuck stacks | the TBCCL backend itself fails a lost peer within about a second, but the engine does not surface it | use client timeouts and an external liveness check; restart both nodes |
| the headless node keeps running after the leader stopped gracefully | **upstream vLLM** lifecycle behavior | `ps` on the headless node | stop it explicitly |
| three file descriptors leaked per process-group create/destroy cycle on macOS (vLLM 0.30.0 and 0.31.0) | **upstream vLLM** | `lsof` count over repeated cycles | restart the server periodically in a long-lived deployment |
| outputs differ from a single-node run | check dtype, prefix caching and the model; strict prompts matched token for token in validation | compare against a single-rank reference with the same flags (`--no-enable-prefix-caching`, same dtype) | align the flags; report with `python -m vllm_tbccl.info` output |

Peer-failure behavior of the TBCCL backend is validated on loopback and, at backend level, on the physical link; the engine-level hang above is not a transport failure.
