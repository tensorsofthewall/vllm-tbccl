# Inference workloads

## Phases of the client

`client.py` runs an ordered list of phases against a running vLLM OpenAI-compatible server and writes the raw per-request event timeline of each. Closed-loop and open-loop results are different workloads and are labelled; they are never merged.

| Phase kind | What it does |
|---|---|
| `ready` | waits for the API; reports which metrics endpoints exist and the server version |
| `warmup` | untimed sequential requests (kept, flagged as warm-up) |
| `rounds` | closed loop in rounds: `conc` simultaneous requests, the next round starts when every request of the round has ended |
| `continuous` | closed loop with `conc` workers, each issuing its next request when the previous one ends |
| `open_loop` | Poisson arrivals at `rate` requests per second; the offered and the achieved rate are both reported |
| `reference` | greedy generations one request at a time, stored with the top-1/top-2 log-probability margin of every position |
| `check` | requests compared with a stored reference (exact match, or a divergence at a near-tie position) |
| `cancel` | streams dropped after a few events, the engine drained, then deterministic follow-up requests compared with the reference |

## Metrics

All are derived from the raw event timelines and nothing else.

* An *output event* is a server-sent event that carries at least one generated token. The token count of a request is the server-reported `usage.completion_tokens`, cross-checked against the event count and the returned token ids; it is never the configured maximum.
* **TTFT**: first output event minus request submission.
* **E2E**: last output event minus submission.
* **TPOT**: (last output event minus first output event) divided by the tokens that followed the first event. It is undefined, never zero, when no token follows the first event.
* **ITL**: the intervals between successive events. An event that carries several tokens yields a *batched interval*, which is counted but not turned into per-token timestamps.
* **Throughput**: tokens the server reported, over the phase's wall time on the client's clock.
* Percentiles are linearly interpolated and withheld when there are too few observations (p90 needs 10, p95 needs 20, p99 needs 100).

Every duration comes from one host's monotonic clock. Timestamps from different hosts are never subtracted.

## Correctness

A `reference` phase records greedy outputs once. A `check` phase compares later requests with it. Because two bf16 candidates can tie, a divergence at a position whose reference margin is below the tie margin is reported as a *near-tie*, not as a pass or a failure of the transport; any other divergence fails the gate. Correctness gates run before timed phases and are never mixed into a timed loop.

## The workloads the runner defines

| Runner benchmark | What it measures |
|---|---|
| `vllm_startup` | an engine lifecycle: start, first request, shutdown |
| `vllm_correctness` | output tokens against a single-rank reference: sequential, concurrent, mixed, staggered, repeated and cancelled requests |
| `vllm_latency` | one request at a time: TTFT, TPOT, ITL and E2E |
| `vllm_throughput` | request and token throughput at a fixed concurrency (closed loop) or a fixed arrival rate (open loop), and mixed workloads |
| `vllm_concurrency` | latency and throughput against closed-loop concurrency |
| `vllm_pipeline` | pipeline stage order, single-node references and stage-level diagnostics |
| `vllm_backend` | the pipeline-boundary ping-pong through vllm-tbccl's own transport at the level of vLLM's group coordinator, without a model (a round trip of a tensor dictionary) |

Fixed launch settings written into every command so that runs are comparable: `--enforce-eager`, `--no-async-scheduling`, `--no-enable-prefix-caching`, offline model loading (`HF_HUB_OFFLINE=1`).
