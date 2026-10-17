# Results

These are reference measurements from one pair of machines. They show what the benchmarks produce and how large the effects were there; they are not a statement about other hardware, other link types, other models or other vLLM versions. Run the benchmarks on your own systems for numbers that apply to them.

## What was measured

| | Linux host | macOS host |
|---|---|---|
| machine | Intel Core i7-12800HX laptop, NVIDIA RTX 3070 Ti Laptop GPU (8 GiB), CUDA 13, EndeavourOS, Linux 6.18 LTS | Apple M4 Mac mini, 16 GiB unified memory, macOS 26.5.1 |
| link | one direct Thunderbolt 4 cable, IP over Thunderbolt, MTU 9000, TCP/IPv4 | |
| software | vLLM 0.31.0, vllm-tbccl 0.2.0, TBCCL 0.6.0 (C ABI 1, wire protocol 4), PyTorch 2.13.0 | the maintained vllm-metal fork (tag `v0.31.0-tbccl1`), MLX 0.32.1 |
| model and engine | Qwen3-0.6B, bf16, PP=2, TP=1, eager mode, no async scheduling, no prefix caching, prompts of exact token length | |

**Hardware health.** The Linux host was booted with `pci=noaer`, so PCIe Advanced Error Reporting was disabled for the whole campaign. Do not read these results as "zero PCIe errors". The statement that holds is: *AER unavailable; no detected link-state or interface-health regression under the available checks.*

One workload point is one engine lifecycle. "5 / 5" below means five independent lifecycles in each orientation.

## Stage order, concurrency 1 and 12

Prompt 128 tokens, 32 generated tokens. TPOT is the median over the five lifecycles of each lifecycle's median request. Throughput at concurrency 12 is the same aggregation of output tokens per second in a closed loop.

| Metric | CUDA-first | Metal-first | Change | Exact test p | Lifecycles |
|---|---|---|---|---|---|
| TPOT, concurrency 1 | 21.19 ms | 15.83 ms | -25.3% | 0.0079 | 5 / 5 |
| Output tokens per second, concurrency 12 | 344.1 | 450.4 | +30.9% | 0.0079 | 5 / 5 |

Per-lifecycle TPOT (ms): CUDA-first 21.10, 21.46, 21.44, 20.98, 21.19; Metal-first 15.70, 15.61, 15.92, 15.83, 16.10. The two sets do not overlap. With five lifecycles each, 0.0079 is the smallest p-value an exact two-sided rank test can give.

The difference was present with the client on the other machine, with tracing on, and at every prompt and output length tried. **Its cause is not known.** The API node follows stage 0 by construction, so stage order, API host and sampler location are confounded; the pipeline boundary exchange was about 3 to 5 percent of a decode step (below), so the gap is not explained by network transfer. Do not quote it as a transport effect.

## Throughput scaling

Output tokens per second, closed loop, median over all passes of five lifecycles per orientation:

| Concurrency | CUDA-first | Metal-first |
|---|---|---|
| 1 | 44.5 | 58.8 |
| 4 | 156.2 | 200.9 |
| 12 | 341.3 | 451.5 |

TPOT stayed nearly flat as concurrency rose to 12, so throughput scaled almost linearly; nothing saturated inside the validated limit of 12 concurrent requests. (The concurrency-12 figures differ slightly from the table above because they aggregate all passes instead of lifecycle medians.)

## Single-device references

The same model on one device, TPOT at concurrency 1:

| Device | TPOT |
|---|---|
| Linux CUDA GPU | 7.97 ms |
| Mac (Metal) | 14.48 ms |
| Mac CPU | 23.48 ms |

**The two-device pipeline was slower than a single device for this model** (best pipeline 15.83 ms against 7.97 ms on the CUDA GPU alone). Qwen3-0.6B fits on one device; pipeline parallelism is for models that do not.

## The pipeline boundary

The exchange at the stage boundary through vllm-tbccl's own backend, measured without an engine, took 0.64 to 0.69 ms with CUDA first and 0.79 to 0.82 ms with the CPU stage first. This is a **round trip of a tensor dictionary** (four point-to-point messages). One-way latency was not measured and must not be derived by halving.

## Prefill

Time to first token grew linearly with the prompt: about 0.22 ms per prompt token for Metal-first and 0.30 ms for CUDA-first (from the campaign's original analysis; not recomputed). Prefill is compute, while decode is a near-constant cost per step.

## Limitations of these results

* One machine pair, one cable, one small model, one vLLM version. Nothing applies to other Thunderbolt generations, GPUs, operating systems or models, or to larger models.
* Five lifecycles per orientation: enough for effects of tens of percent, not for small ones.
* The stage-order effect is unexplained and confounded with API and sampler placement.
* The Mac CPU stage and the Metal stage use different software environments; they are not interchangeable.
* AER was unavailable (see above).
