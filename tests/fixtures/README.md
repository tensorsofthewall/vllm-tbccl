# Test fixtures

| File | What it is | Used by |
|---|---|---|
| `reference_vllm031_linux_cuda.json` | Greedy reference completions of Qwen3-0.6B (per prompt: generated text, token ids, chosen-token log-probabilities and top-2 margins) on a single Linux CUDA rank with vLLM 0.31.0, for the strict deterministic prompts | `tests/test_e2e_loopback.py`, `scripts/physical_validation.sh` through `tools/pp_validation_client.py --ref` |
| `reference_vllm030_linux_cuda.json` | The same kind of reference with vLLM 0.30.0 | `scripts/metal_engine_cycles.sh` |
| `prompts.json` | The prompt set used by the pipeline benchmark and the split-parity tool | `examples/pp_bench.py`, `tools/split_parity.py` |

The references are only meaningful for the model and vLLM version named above; they contain no model weights. Regenerate one with `tools/reference_generator.py` against a single-rank server of the same model.
