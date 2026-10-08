# Validated configurations

Configurations that were exercised, not a benchmark report. Evidence is in the [validation report](../validation/0.2.0.md). "Release-candidate wheels" means the published 0.2.0rc1 wheels were used. Versions and platforms are in the ecosystem [compatibility matrix](https://tbccl.tensorsofthewall.com/en/latest/reference/compatibility-matrix.html).

| Configuration | What was run | Result |
|---|---|---|
| Linux CUDA node 0 and Mac Metal node 1, Thunderbolt 4, vLLM 0.30.0, release-candidate wheels | `PP=2`, `TP=1`, Qwen3-0.6B (bf16); ten engine start/stop cycles; prompts up to about 3,500 tokens; outputs compared with a single-rank CUDA reference | token-identical; link health unchanged |
| Linux CUDA and Mac Metal, both orientations, vLLM 0.30.0 | the development tree: strict prompts, concurrent requests, cancellation, repeated engine cycles, byte-integrity of 2 x 320 transfers | passed |
| Linux CUDA and Mac CPU, Thunderbolt 4, vLLM 0.31.0 | the development tree at an earlier TBCCL wire protocol (3): `PP=2`, `TP=1`, Qwen3-0.6B; suites re-run at wire 4 on loopback | passed; the physical run was not repeated at wire 4 |
| Backend-level peer death (the same bundled backend through vLLM's PP group coordinator, killed peer, physical link) | a blocked `recv` and a burst of `send`s to a dead peer | raised a transport error after about 1.05 s |
| Engine-level peer death (headless node's worker killed mid-stream) | three attempts | **not bounded**: the leader's engine stays stuck (upstream vLLM, see [known limitations](limitations.md)) |
| Graceful leader stop | `SIGTERM` of the API server | the leader exits in about 3 s; the headless follower may linger (upstream vLLM) |
