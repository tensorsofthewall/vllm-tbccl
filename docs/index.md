# vllm-tbccl

:::{container} tb-lead
vllm-tbccl is a vLLM platform plugin that lets one vLLM pipeline span machines with different accelerators, for example a Linux CUDA node and a Mac. It carries vLLM's device-group communication over TBCCL through its own bundled backend. It is standalone and does not require torch-tbccl.
:::

:::{container} tb-actions
[Get started](getting-started/quickstart.md) [Supported versions](reference/compatibility.md)
:::

```{note}
**Release.** The current release is 0.2.1 (vLLM 0.31.0, PyTorch 2.13.x, CPython 3.12 and 3.13, TBCCL C ABI 1 / wire protocol 4), a patch release of 0.2.0 that adds the public benchmark provider and no change to the vLLM integration. It is experimental.
```

<nav class="tb-eco" aria-label="TBCCL ecosystem: the four documentation sites">
<ul>
<li class="tb-tbccl"><a href="https://tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">tbccl</span><span class="tb-eco-role">native runtime</span></a></li>
<li class="tb-torch"><a href="https://torch-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">torch-tbccl</span><span class="tb-eco-role">PyTorch backend</span></a></li>
<li class="tb-vllm" aria-current="page"><span><span class="tb-eco-name">vllm-tbccl</span><span class="tb-eco-role">vLLM integration</span><span class="tb-eco-here">This site</span></span></li>
<li class="tb-exo"><a href="https://exo-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">exo-tbccl</span><span class="tb-eco-role">exo integration</span></a></li>
</ul>
</nav>

## What you get

::::{container} tb-cards
:::{container} tb-card tb-vllm

<h3>Heterogeneous pipeline parallelism</h3>

Two-stage pipeline parallelism (`PP=2`, `TP=1`): one stage on each machine, with the activations carried by TBCCL.

[Pipeline parallelism](guides/pipeline-parallelism.md)
:::

:::{container} tb-card tb-vllm

<h3>CUDA with Metal or CPU</h3>

A Linux CUDA node paired with a Mac running Metal through vllm-metal, or with a Mac CPU node. The Metal pairing needs the vllm-metal fork tag `v0.31.0-tbccl1`.

[CUDA and Metal pairing](guides/cuda-metal.md) · [Patch workflow](guides/patches.md)
:::

:::{container} tb-card tb-vllm

<h3>Setup you can verify</h3>

Install the wheel, start a two-node pipeline and confirm that the TBCCL backend is the one in use.

[Install](getting-started/install.md) · [Verify it is active](guides/verify-active.md)
:::

:::{container} tb-card tb-vllm

<h3>Built on TBCCL</h3>

The wheel carries its own small native backend, statically linked to libtbccl. It does not use torch-tbccl or any other adapter.

[TBCCL documentation](https://tbccl.tensorsofthewall.com/en/latest/) · [GitHub](https://github.com/tensorsofthewall/vllm-tbccl)
:::
::::

## Get started

::::{container} tb-cards
:::{container} tb-card 

<h3>Install</h3>

The wheels, the matching PyTorch and the vLLM version this release supports.

[Installing vllm-tbccl](getting-started/install.md)
:::

:::{container} tb-card 

<h3>Run a two-node PP=2 pipeline</h3>

The shortest path to a working pipeline across two machines.

[Quickstart](getting-started/quickstart.md)
:::

:::{container} tb-card 

<h3>Pair CUDA with Metal</h3>

The Mac side: vllm-metal, the fork tag and the Python version.

[CUDA and Metal pairing](guides/cuda-metal.md)
:::

:::{container} tb-card 

<h3>Understand the pipeline</h3>

Which stage runs where and how activations move.

[Pipeline parallelism](guides/pipeline-parallelism.md)
:::

:::{container} tb-card 

<h3>Check it is active</h3>

Confirm that the job runs over the TBCCL backend.

[Verify the plugin](guides/verify-active.md)
:::
::::

## What is supported

Only configurations that were actually exercised are listed as validated. Everything else is not supported or not claimed.

::::{container} tb-cols
:::{container}
**Validated**

- vLLM 0.31.0, PyTorch 2.13.x and CPython 3.12 and 3.13; the Metal pairing needs Python 3.12 on the Mac.
- A Linux CUDA node and a Mac with Metal over a direct Thunderbolt 4 link, `PP=2`, `TP=1`, Qwen3-0.6B, both pipeline orders, with outputs token-identical to a single-rank reference.
- The same checks with a Mac CPU node in place of Metal.
:::
:::{container}
**Not supported or not claimed**

- One model was validated (Qwen3-0.6B, greedy decoding, prompts up to about 3,500 tokens, 12 concurrent requests); other models are not claimed.
- Only vLLM 0.31.0; other versions warn and are not claimed. Tensor parallelism above 1 was not exercised.
- Do not use 0.2.0rc1: it had a correctness defect for CUDA and Metal pipelines under concurrent requests, fixed since 0.2.0rc2.
- If a peer dies mid-stream, vLLM's own engine may stay stuck even though the TBCCL backend reports the failure within about a second.
:::
::::

Authoritative pages: [compatibility](reference/compatibility.md), [validated configurations](project/validated-configurations.md) and [known limitations](project/limitations.md).

## Where it fits

vllm-tbccl is one of three independent adapters over the [TBCCL](https://tbccl.tensorsofthewall.com/en/latest/) runtime; the others are torch-tbccl (PyTorch) and exo-tbccl (exo). It depends on none of them. See the [ecosystem overview](project/ecosystem.md).

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/quickstart
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User guide

guides/pipeline-parallelism
guides/cuda-metal
guides/patches
reference/configuration
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/verify-active
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Project

project/index
validation/index
```
