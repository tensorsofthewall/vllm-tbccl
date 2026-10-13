# Development

This repository, **vllm-tbccl**, is the vLLM adapter. It owns the plugin, the bundled c10d backend and the vLLM and vllm-metal integration glue, and nothing of the transport.

| Belongs here | Does not belong here |
|---|---|
| the vLLM platform plugin, device communicator and thin worker subclasses | communication algorithms, TCP code, chunk scheduling, CUDA staging (these are TBCCL) |
| the bundled c10d backend mapping operations to the C ABI | a dependency on torch-tbccl or any other adapter (`tests/test_independence.py` enforces this) |
| the Metal transport, boundary codecs and receive-buffer pool | in-place patches of vLLM |
| the patch files shipped as package data | scheduler, executor or worker-death handling (upstream vLLM) |

Start with [contributing](contributing.md) and [where should this change go?](where-should-this-change-go.md). The rules for the code base are in `AGENTS.md`.

```{toctree}
:maxdepth: 1

contributing
where-should-this-change-go
testing
building-docs
release-process
documentation-versions
```
