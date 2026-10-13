# Validated configurations

Configurations that were exercised, not a benchmark report. Evidence is in the [validation report](../validation/0.2.0.md). "Release-candidate wheels" means the published 0.2.0rc1 wheels were used. Versions and platforms are in the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html).

| Configuration | What was run | Result |
|---|---|---|
| Linux CUDA and Mac Metal, Thunderbolt 4, **vLLM 0.31.0 and vllm-metal 0.31.0** with the transport seam, TBCCL wire protocol 4, CPython 3.12, wheels in clean environments | `PP=2`, `TP=1`, Qwen3-0.6B (bf16), both pipeline orders (10 engine cycles with CUDA first, 5 with Metal first): strict prompts and prompts up to 3,507 tokens token-identical to the single-rank CUDA reference; repeated rounds of 12 concurrent requests and a cancellation round per cycle; byte-integrity of 2 x 2,560 transfers per order | passed; no row-mixing divergence; descriptor and thread counts flat; AER clean |
| Linux CUDA and Mac CPU, Thunderbolt 4, **vLLM 0.31.0**, TBCCL wire protocol 4 | the same gates, 10 engine cycles | passed |
| Linux CPython 3.13 with Mac CPython 3.12 (CUDA first, Metal second), vLLM 0.31.0 | 2 engine cycles with the same gates | passed; the minors should still match in practice |
| *Historical (0.2.0rc1):* Linux CUDA node 0 and Mac Metal node 1, Thunderbolt 4, vLLM 0.30.0, release-candidate wheels | `PP=2`, `TP=1`, Qwen3-0.6B (bf16); ten engine start/stop cycles; prompts up to about 3,500 tokens; outputs compared with a single-rank CUDA reference | token-identical; link health unchanged |
| *Historical:* Linux CUDA and Mac Metal, both orientations, vLLM 0.30.0 | the development tree: strict prompts, concurrent requests, cancellation, repeated engine cycles, byte-integrity of 2 x 320 transfers | passed, **except** that the concurrent requests were not compared with a reference: 0.2.0rc1 and that tree could exchange rows between concurrent CUDA-to-Metal requests (fixed after the release candidate) |
| *Historical:* Linux CUDA and Mac CPU, Thunderbolt 4, vLLM 0.31.0 | the development tree at an earlier TBCCL wire protocol (3): `PP=2`, `TP=1`, Qwen3-0.6B | passed (repeated at wire 4 above) |
| Backend-level peer death (the same bundled backend through vLLM's PP group coordinator, killed peer, physical link) | a blocked `recv` and a burst of `send`s to a dead peer | raised a transport error after about 1.05 s |
| Engine-level peer death (headless node's worker killed mid-stream) | three attempts | **not bounded**: the leader's engine stays stuck (upstream vLLM, see [known limitations](limitations.md)) |
| Graceful leader stop | `SIGTERM` of the API server | the leader exits in about 3 s; the headless follower may linger (upstream vLLM) |
