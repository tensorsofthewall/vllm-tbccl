# Known limitations

Each limitation is labelled with the project that owns it, so you can tell a TBCCL transport limit from a vllm-tbccl limit from an upstream vLLM behavior. The same categories are used in the other TBCCL projects' pages. Core transport limits (TCP only, no RDMA, no direct CUDA-to-Metal DMA, no encryption) are in the [TBCCL limitations](https://tbccl.tensorsofthewall.com/en/latest/project/limitations.html).

## Unsupported

| Limitation | Owner | Detail |
|---|---|---|
| Parallelism | vllm-tbccl | two-stage pipeline parallelism (`PP=2`, `TP=1`) with two-rank groups. `reduce_scatter` and `gather` raise `NotImplementedError`; tensor parallelism above 1 was not exercised |
| Models | vllm-tbccl | one model validated (Qwen3-0.6B); larger models and other architectures are not validated. The Metal boundary codecs are an allowlist: Llama and Qwen3 only |
| Python and PyTorch | vllm-tbccl | CPython 3.13 and PyTorch 2.13.x only |
| vLLM versions | vllm-tbccl | 0.30.0 (Metal pairing) and 0.31.0 (CPU pairing); other versions warn and are not claimed |
| Metal pairing and vLLM 0.31.0 | upstream vllm-metal | vllm-metal does not run on vLLM 0.31.0 yet, so the Metal pairing is tied to 0.30.0 |
| Linux with a CPU-only PyTorch | vllm-tbccl | the Linux wheel needs the CUDA 13 build of PyTorch |
| Reductions beyond `Sum`, FP8/INT4 reductions, recovery after a communicator failure | TBCCL | see the TBCCL limitations |

## Supported but not validated

| Item | State |
|---|---|
| Native vllm-metal MLX-ring pipeline parallelism | not exercised with this plugin |
| The CUDA and CPU pairing on wire protocol 4 over the physical link | the unit and loopback suites were re-run at wire 4; the physical CUDA and CPU runs were recorded at wire protocol 3 and not repeated |
| The CUDA and Metal pairing with the Mac as node 0 | validated on the development tree in both orientations; the release-candidate runs used the Linux node as node 0 |
| Any link other than a direct Thunderbolt 4 connection | works in principle between hosts that can reach each other directly; not validated |

## Known upstream issues (vLLM)

These are vLLM behaviors, **not** TBCCL transport failures: the TBCCL backend fails a lost peer within about a second.

| Issue | Detail |
|---|---|
| Dead headless worker is not detected | vLLM's multi-node executor has no liveness check on a dead headless worker. If the headless node's worker process dies, the leader's engine stays stuck waiting for its response (no TBCCL call is involved). Use client timeouts and an external liveness check |
| Headless follower lingers after a graceful leader stop | the headless node's process may keep running in vLLM's ZMQ poll after the leader exits; stop it explicitly |
| File descriptors on macOS | vLLM 0.30.0 leaks three file descriptors per process-group create and destroy cycle on macOS |
| LAN firewalls | the multiprocessing executor's reverse-direction connections are blocked by an ordinary LAN firewall, so cross-host engines need a direct link |

## Performance

- CUDA activations are staged through pinned host memory and sent over TCP; there is no direct GPU-to-GPU path.
- Performance was characterized, not tuned.

## Operational workarounds

| Situation | Workaround |
|---|---|
| a Mac with vllm-metal installed should use the CPU pairing | `VLLM_TBCCL_BACKEND=torch` |
| a mixed CUDA and CPU pair asserts in the model runner | `VLLM_USE_V2_MODEL_RUNNER=0` on both hosts |
| a Mac node hangs at start | answer the macOS firewall / Local Network prompt from a local session; run the node in a live terminal or SSH session |
| the vllm-metal patch does not apply | [patch workflow](../guides/patches.md) |
| after a dead headless node or an unclean stop | stop both nodes explicitly and restart them; do not rely on the leader to notice |
