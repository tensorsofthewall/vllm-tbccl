# Getting started

These commands run from a clone of this repository. The supported tuple is in the [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html) and `compatibility.json`: vLLM 0.31.0, PyTorch 2.13.x, CPython 3.12 or 3.13, and for a Metal stage the maintained vllm-metal fork. Install vLLM and vllm-tbccl as described in the [install guide](../getting-started/install.md); this page does not change that environment.

## 1. Check what this environment can run

```sh
python benchmarks/provider/provider.py describe
```

One JSON object is printed; nothing is loaded, started or downloaded. `availability.status` is `available`, `environment_missing` (with the missing package) or `unsupported`. Versions come from package metadata. If vLLM is not the supported version, the reason says so.

## 2. Build the fixed prompt set

The benchmarks use token-id prompts of exact lengths cut from files that ship inside the model directory, so every host rebuilds the same prompts byte for byte and no tokenizer is needed at run time:

```sh
python benchmarks/provider/make_workload.py --model <MODEL_DIR> --out workload.json --lengths 16 128 512
```

`<MODEL_DIR>` is a local directory that already holds the model. Nothing is downloaded (`transformers` is needed only for this step).

## 3. Measure a server you started

Start `vllm serve` yourself in the way your setup requires (single node, or the two-node pipeline from the [pipeline guide](../guides/pipeline-parallelism.md)). Then write a spec and run the client against its API:

```json
{
  "api": {"host": "<API_HOST>", "port": 8000},
  "model": "<MODEL_NAME_AS_SERVED>",
  "workload": "workload.json",
  "request_timeout_s": 120,
  "phases": [
    {"name": "ready", "kind": "ready", "timeout_s": 600},
    {"name": "warm", "kind": "warmup", "prompt": "p128", "tokens": 32, "requests": 3},
    {"name": "lat", "kind": "rounds", "prompt": "p128", "tokens": 32, "conc": 1, "rounds": 10}
  ]
}
```

```sh
python benchmarks/provider/client.py --spec spec.json --out-dir out
```

The client prints one `TBCCL_VLLM_BENCH {json}` summary line per phase (TTFT, TPOT, ITL and E2E distributions with their sample counts, and throughput from the tokens the server actually reported) and writes every request's raw event timeline to `out/<phase>.requests.jsonl`. A request that did not complete normally makes the client exit non-zero. See [workloads](workloads.md) for the phase kinds.

Run the client on the machine you want the latency to include a network hop for, and say so when you report it: the client's placement is recorded in every summary.

## 4. Through the TBCCL benchmark runner

In a TBCCL checkout, name this provider per host (all paths and addresses are yours):

```yaml
hosts:
  host-a:
    executables:
      vllm: <PATH_TO_VLLM_EXECUTABLE>
      vllm_python: <PATH_TO_PYTHON_WITH_VLLM_TBCCL>
      vllm_model_dir: <LOCAL_MODEL_DIRECTORY>
      vllm_bench_dir: <PATH_TO_THIS_REPOSITORY>/benchmarks/provider
```

```sh
python3 -m benchmarks.suite providers my-config.yaml          # shows vllm-tbccl with its state and versions per host
python3 -m benchmarks.suite run my-config.yaml --dry-run      # prints the engine, sampler and client commands; launches nothing
python3 -m benchmarks.suite run my-config.yaml --allow-physical
python3 -m benchmarks.suite report <RUN_ID>
python3 -m benchmarks.suite compare <RUN_A> <RUN_B>           # only equivalent experiments are compared
```

One workload point is one *engine lifecycle*: the engines start, the client runs its phases, everything is stopped and verified clean on both hosts, and the next lifecycle starts from scratch. Repetition means more lifecycles, not more requests.

## Tests

```sh
python -m unittest discover -s benchmarks/provider/tests -p 'test_*.py'
```

The tests run offline against a stand-in server; they need no vLLM, GPU or model.
