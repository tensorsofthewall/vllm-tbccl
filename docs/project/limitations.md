# Known limitations

Each limitation is labelled with the project that owns it, so you can tell a TBCCL transport limit from a vllm-tbccl limit from an upstream vLLM behavior. The same categories are used in the other TBCCL projects' pages. Core transport limits (TCP only, no RDMA, no direct CUDA-to-Metal DMA, no encryption) are in the [TBCCL limitations](https://tbccl.tensorsofthewall.com/en/latest/project/limitations.html).

## Unsupported

| Limitation | Owner | Detail |
|---|---|---|
| Parallelism | vllm-tbccl | two-stage pipeline parallelism (`PP=2`, `TP=1`) with two-rank groups. `reduce_scatter` and `gather` raise `NotImplementedError`; tensor parallelism above 1 was not exercised |
| Models | vllm-tbccl | one model validated (Qwen3-0.6B); larger models and other architectures are not validated. The Metal boundary codecs are an allowlist: Llama and Qwen3 only |
| Python and PyTorch | vllm-tbccl | CPython 3.12 and 3.13 and PyTorch 2.13.x only. The CUDA and Metal pairing needs Python 3.12 on the Mac, because vllm-metal 0.31.0 and vLLM's macOS wheel are CPython 3.12; mixed minors across hosts (3.13 Linux with 3.12 Mac) were exercised once (see validation) and the minors should match |
| vLLM versions | vllm-tbccl | 0.31.0 only; other versions (including 0.30.0, the validated version of 0.2.0rc1) warn and are not claimed |
| Metal pairing and vllm-metal | vllm-tbccl / vllm-metal | the Metal pairing needs the vllm-metal fork tag `v0.31.0-tbccl1` (0.31.0 plus the transport seam) (a generic extension point not yet in upstream vllm-metal); the patch targets that exact revision |
| Linux with a CPU-only PyTorch | vllm-tbccl | the Linux wheel needs the CUDA 13 build of PyTorch |
| Reductions beyond `Sum`, FP8/INT4 reductions, recovery after a communicator failure | TBCCL | see the TBCCL limitations |

## Supported but not validated

| Item | State |
|---|---|
| Native vllm-metal MLX-ring pipeline parallelism | not exercised with this plugin |
| Models other than Qwen3-0.6B, context lengths above about 3,500 tokens, more than 12 concurrent requests, sampling other than greedy | not run; the validation used greedy decoding of Qwen3-0.6B with prompts up to 3,507 tokens and 12 concurrent requests |
| Mixed CPython minors (Linux 3.13 with Mac 3.12) | exercised for two engine cycles only; keep the minors the same |
| Any link other than a direct Thunderbolt 4 connection | works in principle between hosts that can reach each other directly; not validated |

## Known upstream issues (vLLM)

These are vLLM behaviors, **not** TBCCL transport failures: the TBCCL backend fails a lost peer within about a second.

| Issue | Detail |
|---|---|
| Dead headless worker is not detected | vLLM's multi-node executor has no liveness check on a dead headless worker. If the headless node's worker process dies, the leader's engine stays stuck waiting for its response (no TBCCL call is involved). Use client timeouts and an external liveness check |
| Headless follower lingers after a graceful leader stop | the headless node's process may keep running in vLLM's ZMQ poll after the leader exits; stop it explicitly |
| File descriptors on macOS | vLLM (0.30.0 and 0.31.0 measured) leaks three descriptors (a kqueue, a handle on `/` and `/dev/null`) per process-group create and destroy cycle on macOS; a control with vLLM's gloo groups and no TBCCL leaks about 45 per cycle. A running engine does not grow (checked before and after load, ten cycles) |
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
| vllm-metal lacks the transport seam | [patch workflow](../guides/patches.md) |
| after a dead headless node or an unclean stop | stop both nodes explicitly and restart them; do not rely on the leader to notice |
