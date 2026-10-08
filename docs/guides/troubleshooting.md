# Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| a warning about the vLLM version when the plugin activates | vLLM is not 0.30.0 or 0.31.0 | use a supported version ([compatibility](../reference/compatibility.md)) |
| `ValueError: VLLM_TBCCL_BACKEND must be one of ...` | an invalid backend name | use `auto`, `torch` or `metal` |
| the engine cannot form the cross-host pipeline over the LAN | the executor's reverse-direction queues are blocked by the firewall | use the direct link between the nodes |
| a CUDA/CPU pair fails during startup with a model-runner assertion | `VLLM_USE_V2_MODEL_RUNNER` differs between hosts | set `VLLM_USE_V2_MODEL_RUNNER=0` on both |
| `protocol_mismatch ... wire protocol` | vllm-tbccl on the two hosts was built against TBCCL prefixes with different wire versions | rebuild both against the same prefix |
| the leader hangs after the headless node's worker died | vLLM's leader does not detect a dead headless worker | use client timeouts and external liveness checks |
| the Metal node rejects an architecture | the boundary-codec allowlist covers Llama and Qwen3 only | use a supported architecture |
