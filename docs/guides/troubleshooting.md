# Troubleshooting

Find the symptom, check the likely cause with the diagnostic, apply the fix. Start with `python -m vllm_tbccl.info` on every node ([how do I know TBCCL is being used?](verify-active.md)). Each row says whether the cause is in TBCCL, vllm-tbccl or upstream vLLM.

## Start here

### Work through it in this order

Each step is marked **Observe** (it reads state; nothing starts, stops or changes) or **Action** (you start, stop or configure a process, or run a test). No step changes network, firewall, driver or hardware settings. Stop at the first step that fails: later steps depend on it.

```{table}
:class: tb-stack tb-stack-steps

| Step | Question | Check | If it fails |
|---|---|---|---|
| 1 | Which versions are installed? | **Observe:** `.venv/bin/python -m vllm_tbccl.info` on **every** node (same wire protocol; on the Metal Mac use `vllm-metal/.venv-vllm-metal/bin/python`) | [Installing and starting](#installing-and-starting) |
| 2 | Is the platform plugin active? | **Observe:** the log line `Platform plugin tbccl is activated` (`metal` on the Metal Mac) | [Installing and starting](#installing-and-starting); [verify](verify-active.md) |
| 3 | Is the link ready? | **Observe:** `python3 thunderbolt_link_doctor.py --peer <peer-link-address>` (with `--peer` it also sends a ping and opens one TCP connection to the peer, and changes nothing; [how to get it](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html)) | [Thunderbolt link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| 4 | Did the nodes find each other? | **Observe:** the leader waits for the headless node (observe). **Action:** the one-line TCP connect test to vLLM's rendezvous port, `--master-port`, on node 0 ([below](#ping-works-but-the-nodes-time-out)) | [Ping works but the nodes time out](#ping-works-but-the-nodes-time-out) |
| 5 | Did the TBCCL group form? | **Observe:** `backend=tbccl` with `world_size=2` in both worker logs | [Forming the pipeline](#forming-the-pipeline) |
| 6 | Does generation work? | **Action:** send one `curl` completion to the running server ([quickstart](../getting-started/quickstart.md)), and start with `VLLM_TBCCL_TRACE=1` for operation records | [While running and stopping](#while-running-and-stopping) |
| 7 | Did stopping leave a process? | **Action:** stop the leader first, then the headless node; then **Observe:** `ps` on both | [While running and stopping](#while-running-and-stopping) |
| 8 | Is it only slow? | **Observe:** the [benchmark methodology](../benchmarks/methodology.md); a single-device reference first | [Results](../benchmarks/results.md) |
```

### Find your symptom

```{table}
:class: tb-stack tb-stack-find

| You see | Go to |
|---|---|
| `ImportError` about `libtorch_cuda.so` or a torch series; no matching wheel | [Installing and starting](#installing-and-starting) |
| no `Platform plugin tbccl is activated` line (the backend is not active) | [Installing and starting](#installing-and-starting); [verify](verify-active.md) |
| the Mac is the wrong backend (`metal` or `torch`) | [CUDA and Metal pairing](cuda-metal.md); [Installing and starting](#installing-and-starting) |
| `protocol_mismatch ... wire protocol` | [Forming the pipeline](#forming-the-pipeline) |
| timeout waiting for the other node; ping works | [Ping works but the nodes time out](#ping-works-but-the-nodes-time-out) |
| out of GPU memory while loading | lower `--gpu-memory-utilization` ([quickstart](../getting-started/quickstart.md)); vLLM owns the allocation |
| outputs differ from a single-node run | [While running and stopping](#while-running-and-stopping) (last row) |
| the leader hangs, or a node keeps running after you stop | [While running and stopping](#while-running-and-stopping): upstream vLLM lifecycle behavior |
```

### Which layer owns the error

The tables below name an owner in every row. In short:

```{table}
:class: tb-stack tb-stack-owner

| Layer | What it looks like | What to do |
|---|---|---|
| vllm-tbccl | the plugin, the device communicator, the Metal transport or a boundary codec raise it | this guide; report with `python -m vllm_tbccl.info` |
| Native TBCCL | a result tag (`protocol_mismatch`, `timeout`) in the message | [TBCCL troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/troubleshooting.html) |
| vLLM or vllm-metal | the executor, scheduling, worker liveness, model runner, memory allocation | upstream; the rows marked "upstream vLLM" say so |
| Operating system | macOS Local Network or firewall prompts, a port in use | [link troubleshooting entries 9 and 10](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| Network configuration | wrong subnet, traffic over Wi-Fi, mismatched MTU | [link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html) |
| External dependencies | the CUDA 13 PyTorch build, `uv`, Xcode and its Metal toolchain | their own notes |
```

## Installing and starting

| Symptom | Likely cause (owner) | Diagnostic | Fix |
|---|---|---|---|
| `ImportError` about `libtorch_cuda.so` / `libc10_cuda.so` on Linux | a CPU-only PyTorch (the wheel needs the CUDA 13 build) | `python -c "import torch; print(torch.__version__)"` | install vLLM from the `cu130` index ([installing](../getting-started/install.md)) |
| no matching distribution for the wheel | Python is not 3.12 or 3.13, or not Linux x86-64 / macOS arm64 | `python --version` | use Python 3.12 or 3.13 on a supported platform |
| no `Platform plugin tbccl is activated` line | `VLLM_TBCCL_ENABLE=1` missing in that process | check the environment of **each** process; `VLLM_TBCCL_ENABLE=1 .venv/bin/python -c "from vllm.platforms import current_platform; print(type(current_platform).__name__)"` prints `TbcclCudaPlatform` (Linux CUDA) or `TbcclCpuPlatform` (Mac CPU pairing); on the Metal Mac vllm-metal owns the platform and it prints `MetalPlatform` | export it on every node |
| on macOS, `objc[...]: Class AVFFrameReceiver is implemented in both ... libavdevice` at import | Homebrew's ffmpeg and the OpenCV wheel in the environment both ship the same Objective-C classes | the message is printed by the dynamic loader, once per process | harmless for vllm-tbccl (seen with a working install); ignore it, or remove the Homebrew ffmpeg if it is not needed |
| running `tests/test_metal_transport.py` alone: `the model architecture is unknown to the Metal pipeline transport` | the test needs an architecture and nothing set one (another test in the full suite sets it for the process) | `echo $VLLM_TBCCL_ARCHITECTURE` | export `VLLM_TBCCL_ARCHITECTURE=Qwen3ForCausalLM` before running it alone |
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
| vllm-metal lacks the transport seam | an upstream vllm-metal was installed instead of the fork tag `v0.31.0-tbccl1` | `python -c "import vllm_metal.envs as e; print('VLLM_METAL_PP_TRANSPORT_CLS' in e.environment_variables)"` | [patch workflow](patches.md) |

## Ping works but the nodes time out

**Symptom.** `ping` answers both ways, yet the leader waits for the headless node (or the reverse) and a timeout follows.

**Two different waits.** vLLM first forms its own cluster on `--master-addr:--master-port` (node 0's link address, port 29600 in the quickstart). TBCCL then connects the nodes' own ports (`TBCCL_LOCAL_ENDPOINT`). A wait before any `backend=tbccl` line points at the first.

**Checks.** The route and environment commands are observations. The connection test is an **action**: it opens one TCP connection and closes it at once, which the listening side may log as a failed handshake, so run it only when no job you care about is starting. The link doctor's listener on a spare port (`--listen` / `--tcp-port`) tests the same path without touching a running job. Run them while the listening side is waiting.

```sh
# ACTION (opens and closes one TCP connection) - From the Mac: can it open a TCP connection to node 0's rendezvous port?
python3 -c "import socket; socket.create_connection(('<node-0-link-address>', <master-port>), timeout=3).close(); print('TCP connect ok')"
# observe - Does traffic to node 0 leave by the link you intend?
route -n get <node-0-link-address>        # macOS;  ip route get <node-0-link-address> on Linux
# observe - On each node, in the process that starts vLLM: the link address, never a hostname or 127.0.0.1
echo $VLLM_HOST_IP $TBCCL_LOCAL_ENDPOINT
```

- `TCP connect ok` but a later timeout: the TBCCL ports (chosen at start with `:0`) are the problem; use the doctor's TCP mode ([link troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html), entry 8) and check `VLLM_HOST_IP` and `TBCCL_LOCAL_ENDPOINT` on both nodes.
- `timed out`: a firewall (the Linux host must allow the Mac's link address to reach port 29600), a wrong address, or a route through another interface.
- `Connection refused`: node 0 is not running yet, or listens on another address.

On the Mac run the node in a live session and answer the macOS prompt. **Do not** change firewall rules, MTU or interface settings as a first step.

## While running and stopping

| Symptom | Likely cause (owner) | Diagnostic | Fix |
|---|---|---|---|
| the leader hangs or requests time out after the headless node's worker died | **upstream vLLM**: the leader's multi-node executor has no liveness check on the headless worker; there is no TBCCL frame in the stuck stacks | the TBCCL backend itself fails a lost peer within about a second, but the engine does not surface it | use client timeouts and an external liveness check; restart both nodes |
| the headless node keeps running after the leader stopped gracefully | **upstream vLLM** lifecycle behavior | `ps` on the headless node | stop it explicitly |
| three file descriptors leaked per process-group create/destroy cycle on macOS (vLLM 0.30.0 and 0.31.0) | **upstream vLLM** | `lsof` count over repeated cycles | restart the server periodically in a long-lived deployment |
| outputs differ from a single-node run | check dtype, prefix caching and the model; strict prompts matched token for token in validation | compare against a single-rank reference with the same flags (`--no-enable-prefix-caching`, same dtype) | align the flags; report with `python -m vllm_tbccl.info` output |

Peer-failure behavior of the TBCCL backend is validated on loopback and, at backend level, on the physical link; the engine-level hang above is not a transport failure.

## When to ask for help

Escalate when the checks isolate the layer and it is vllm-tbccl or TBCCL, or when you cannot isolate it. Include `python -m vllm_tbccl.info` from every node, the exact error, `python tools/collect_diagnostics.py` output (versions and interface names only; nothing is uploaded), the vLLM and vllm-metal versions and the commands you ran. See [getting help](https://tbccl.tensorsofthewall.com/en/latest/project/getting-help.html). Upstream behaviors (the rows marked "upstream vLLM") belong to the vLLM project. Security problems follow the [security policy](../project/security-policy.md).
