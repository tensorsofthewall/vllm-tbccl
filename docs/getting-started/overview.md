# Overview

## What it is

vllm-tbccl is a vLLM **platform plugin**. When it is enabled (`VLLM_TBCCL_ENABLE=1`), vLLM keeps using its normal CUDA or CPU platform but runs its pipeline-parallel device groups and control groups over a `tbccl` torch.distributed backend. That backend is a small native c10d module bundled in the wheel and linked statically to [libtbccl](https://tbccl.tensorsofthewall.com/en/latest/)'s stable C ABI, so a pipeline can cross machines that NCCL and Gloo cannot join: a Linux host with an NVIDIA GPU and a Mac.

```text
vLLM (device groups, PP)
  |
vllm-tbccl     platform plugin + bundled "tbccl" c10d backend (this package)
  |
libtbccl       transport, collectives, CUDA staging (statically linked, C ABI v1)
  |
TCP / Thunderbolt 4
```

vllm-tbccl does **not** use or require torch-tbccl. The two projects are peers over libtbccl.

## What it does for you

- Runs `PP=2` (two pipeline stages, `TP=1`) across a CUDA node and a CPU node, or a CUDA node and a Mac Metal node.
- Handles the heterogeneity problems inside the package, without modifying vLLM: control groups that Gloo cannot connect across Linux and macOS, differing KV-cache layout reports between a CUDA and a CPU worker, a CPU worker as a later stage, and tensor metadata that bakes in the sender's device type.
- For Metal, attaches to vllm-metal through a small pipeline-transport seam, converting vllm-metal's activations to the layout vLLM expects.

## Who owns what

| Piece | Owner |
|---|---|
| transport, collectives, CUDA staging | **libtbccl** (tbccl) |
| the vLLM plugin, device communicator, thin worker subclasses, the bundled c10d backend, the Metal transport and boundary codecs | **vllm-tbccl** |
| a generic pluggable pipeline-transport seam in vllm-metal | a patch carried in this package ([patch workflow](../guides/patches.md)), meant to be proposed upstream |
| scheduling, the multi-node executor, worker-death detection | **upstream vLLM** |

vllm-tbccl is not a vLLM fork. vLLM itself is used unmodified; every adaptation uses a public extension point.

## Next steps

1. [Install](install.md) the wheel (and, for the Metal pairing, vllm-metal with the patch).
2. Follow the [quickstart](quickstart.md): a two-node PP=2 launch with a client call.
3. Confirm TBCCL is active: [how do I know TBCCL is being used?](../guides/verify-active.md)
