# vllm-tbccl

A vLLM platform plugin that lets one vLLM pipeline span machines with different accelerators (for example a Linux CUDA node and a Mac), carrying vLLM's device-group communication over TBCCL through its own bundled backend. It is standalone and does not require torch-tbccl.

```{note}
**Release candidate.** The current release candidate is 0.2.0rc1 (vLLM 0.30.0 and 0.31.0, PyTorch 2.13.x, CPython 3.13, TBCCL C ABI 1 / wire protocol 4). It is a pre-release for validation, not for production use. The final 0.2.0 has not been released.
```

vllm-tbccl is one of three independent adapters over the [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/) runtime; the others are torch-tbccl (PyTorch) and exo-tbccl (exo). It depends on none of them. See the [ecosystem overview](project/ecosystem.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

guides/pipeline-parallelism
guides/cuda-metal
guides/patches
reference/configuration
```

```{toctree}
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/verify-active
```

```{toctree}
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:maxdepth: 2
:caption: Project

project/index
validation/index
```
