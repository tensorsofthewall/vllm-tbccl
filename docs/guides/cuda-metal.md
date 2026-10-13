# CUDA and Metal pairing

A Linux CUDA node and a Mac Metal node can form one pipeline. The Mac node runs vllm-metal; vllm-tbccl attaches to it through generic hooks instead of owning the platform:

- `VLLM_TBCCL_BACKEND=metal` (or `auto` when vllm-metal is installed) makes vllm-tbccl decline to be the platform plugin on that node and attach through vllm-metal's hooks `VLLM_METAL_PP_TRANSPORT_CLS` (set to the vllm-tbccl transport class) and `VLLM_METAL_DIST_BACKEND=tbccl`. The vllm-metal patch shipped in the package provides the seam ([patch workflow](patches.md)).
- A vllm-metal activation (an evaluated MLX array) is viewed as a zero-copy CPU torch tensor, sent through `ProcessGroupTBCCL`, and received on the CUDA side in the layout vLLM expects.
- Toward an upstream-vLLM peer, the transport speaks vLLM's tensor-dict wire format and applies a per-architecture **boundary codec** that converts between vllm-metal's raw residual stream and vLLM's `hidden_states` plus `residual` pair (and the request-row permutation, see below). The codecs are an explicit allowlist, currently Llama and Qwen3; an architecture not on the list is rejected rather than assumed to behave like another. The architecture comes from the local model's `config.json`, or `VLLM_TBCCL_ARCHITECTURE`.

The Linux node needs neither mlx nor vllm-metal.

## Row order between the stages

A pipeline activation has one row per scheduled token, and the two stages order the rows differently: upstream vLLM follows its persistent input batch (slots are reused and compacted; a chunked-prefill continuation keeps its slot), vllm-metal packs the decode rows first and then the prefill rows. The order cannot be derived from the scheduler output both stages see. The **sending** stage therefore states the actual order of the rows it sends, as request ids with row counts (a few bytes per active request, independent of the activation size), together with the activation, and the **receiving** stage permutes the rows into its own order. If the two stages disagree about which requests are active, or a peer does not send its order (a vllm-tbccl without this protocol), the step fails with an error; nothing is guessed. Both stages must run the same vllm-tbccl version. Earlier versions (including 0.2.0rc1) guessed the order from the scheduler output, which could exchange activations between requests under concurrent load. `VLLM_TBCCL_TRACE_ROWS=<file>` records the orders and the permutation per step for qualification.

## Validated configuration

vLLM 0.31.0 on both hosts, vllm-metal 0.31.0 (`e7a0c90b`) carrying the transport seam, CPython 3.12, Qwen3-0.6B, Linux CUDA and Mac Metal over Thunderbolt 4 in both pipeline orders: strict prompts and prompts up to 3,507 tokens token-identical to the single-rank CUDA reference, rounds of 12 concurrent requests with no row-mixing divergence, cancellation followed by deterministic requests, ten engine create/destroy cycles with CUDA first and five with Metal first, and a byte-integrity check of 2 x 2,560 transfers per order with no mismatching element. See the [validation report](../validation/0.2.0.md).

## Optional

`VLLM_TBCCL_RECV_POOL=1` enables reuse of receive buffers for decode-sized Metal receives (off by default; `VLLM_TBCCL_RECV_POOL_MAX_ROWS` bounds the pooled row count). `VLLM_PP_LAYER_PARTITION=<stage0>,<stage1>` sets the layer split on both hosts (a vLLM setting, no code change).

## Limits

See [known limitations](../project/limitations.md) for the full list: a single validated model (Qwen3-0.6B), two boundary codecs, vllm-metal 0.31.0 with the transport seam patch required, and the vLLM lifecycle behaviors (a dead headless worker is not detected by the leader; vLLM leaks three file descriptors per process-group create and destroy cycle on macOS). None of these is a TBCCL transport failure.
