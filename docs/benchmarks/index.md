# Benchmarks

vllm-tbccl ships its vLLM benchmark workloads in `benchmarks/provider/`. They measure pipeline-parallel inference (PP=2, TP=1) over vllm-tbccl on **your** hardware: time to first token, time per output token, inter-token latency, end-to-end latency, throughput under concurrency, correctness against a single-device reference, and the exchange at the pipeline boundary.

```{toctree}
:maxdepth: 1

getting-started
workloads
placement
results
methodology
```

| I want to... | Read |
|---|---|
| see reference measurements with charts | [Results](results.md) |
| understand what a number does and does not mean | [Methodology and limitations](methodology.md) |
| check my environment and run a first measurement | [Getting started](getting-started.md) |
| know what each workload and metric means | [Inference workloads](workloads.md) |
| understand stage order and where the API runs | [Pipeline placement](placement.md) |
| generate a report from my run, or compare two runs | the TBCCL runner: [report generation](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/reports.html) and [comparing runs](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/comparing.html) |
| find out why a run failed | [Getting started](getting-started.md) for the client, [runner troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/benchmarks/troubleshooting.html) for the runner |

Two ways to run the same workloads:

* **Standalone.** Clone this repository, install vllm-tbccl and vLLM, start the server yourself, and point `benchmarks/provider/client.py` at it. The client uses only the Python standard library. Nothing else needs to be installed: no torch-tbccl, no exo-tbccl, no TBCCL benchmark suite.
* **Through the TBCCL benchmark runner.** For a two-host pipeline, the [TBCCL benchmark runner](https://tbccl.tensorsofthewall.com/en/latest/development/benchmark-suite.html) launches both stages, a client and a passive sampler on each host, supervises and cleans up every process, checks the data path and health, stores results in one schema and compares runs. vllm-tbccl is a *provider* for that runner; the runner has no second implementation of the vLLM workload.

Distributed inference is not universally faster than single-device inference. For the model measured here, which fits on one device, the single-device runs were faster than the two-device pipeline; see [Results](results.md). The pipeline exists for models that do not fit one machine.
