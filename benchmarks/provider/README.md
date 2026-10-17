# vLLM benchmark provider

Benchmark workloads for vLLM pipeline-parallel inference over vllm-tbccl: `client.py` (a standard-library streaming client with latency, throughput, correctness and cancellation phases), `sampler.py` (a passive per-host resource sampler), `fault.py`, `pg_pingpong.py` (the pipeline-boundary ping-pong without a model), `make_workload.py` (fixed prompts) and `provider.py` (the entry point of the TBCCL benchmark provider protocol 1).

```sh
python benchmarks/provider/provider.py describe                       # what this environment can run (no vLLM import)
python benchmarks/provider/client.py --spec spec.json --out-dir out   # measure a server you started
python -m unittest discover -s benchmarks/provider/tests -p 'test_*.py'
```

Documentation: [docs/benchmarks](../../docs/benchmarks/index.md). These scripts need only this package and vLLM (the client needs neither); they import no other TBCCL adapter.
