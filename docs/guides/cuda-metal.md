# CUDA and Metal pairing

A Linux CUDA node and a Mac Metal node can form one pipeline. The Mac node runs vllm-metal; vllm-tbccl attaches to it through generic hooks instead of owning the platform:

- `VLLM_TBCCL_BACKEND=metal` (or `auto` when vllm-metal is installed) makes vllm-tbccl decline to be the platform plugin on that node and attach through vllm-metal's hooks `VLLM_METAL_PP_TRANSPORT_CLS` (set to the vllm-tbccl transport class) and `VLLM_METAL_DIST_BACKEND=tbccl`. The patch in `patches/vllm-metal-0001-pluggable-pp-transport.patch` provides the seam.
- A vllm-metal activation (an evaluated MLX array) is viewed as a zero-copy CPU torch tensor, sent through `ProcessGroupTBCCL`, and received on the CUDA side in the layout vLLM expects.
- Toward an upstream-vLLM peer, the transport speaks vLLM's tensor-dict wire format and applies a per-architecture **boundary codec** that converts between vllm-metal's raw residual stream and vLLM's `hidden_states` plus `residual` pair (and the request-row permutation). The codecs are an explicit allowlist, currently Llama and Qwen3; an architecture not on the list is rejected rather than assumed to behave like another. The architecture comes from the local model's `config.json`, or `VLLM_TBCCL_ARCHITECTURE`.

The Linux node needs neither mlx nor vllm-metal.

## Validated configuration

vLLM 0.30.0 on both hosts, vllm-metal at the upstream commit the patch applies to, Qwen3-0.6B, Linux CUDA to Mac Metal over Thunderbolt 4 in both orientations: strict prompts token-identical to the single-rank reference, sequential and concurrent requests, cancellation, repeated engine create and destroy, and a byte-integrity check of the transferred activations with no mismatching element.

## Optional

`VLLM_TBCCL_RECV_POOL=1` enables reuse of receive buffers for decode-sized Metal receives (off by default; `VLLM_TBCCL_RECV_POOL_MAX_ROWS` bounds the pooled row count). `VLLM_PP_LAYER_PARTITION=<stage0>,<stage1>` sets the layer split on both hosts (a vLLM setting, no code change).

## Limits

Single model validated (Qwen3-0.6B); two codecs; vLLM 0.30.0 required until vllm-metal supports 0.31.0; native vllm-metal MLX-ring pipeline parallelism was not exercised. vLLM 0.30.0 leaks three file descriptors per process-group create and destroy cycle on macOS (a vLLM behavior, not TBCCL).
