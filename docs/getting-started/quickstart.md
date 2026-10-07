# Quickstart

Enable the plugin in every process that should use TBCCL:

```sh
export VLLM_TBCCL_ENABLE=1                 # vLLM then loads the "tbccl" platform plugin
export TBCCL_LOCAL_ENDPOINT=<host>:0       # each communicator picks its own port pair
```

With the plugin enabled, vLLM uses your normal CUDA or CPU platform with `dist_backend="tbccl"` and `TBCCLDeviceCommunicator`, so no tensor of model data goes through NCCL or gloo (gloo carries only vLLM's pickled control metadata).

For a CUDA node paired with a CPU node, set `VLLM_USE_V2_MODEL_RUNNER=0` on **both** hosts. `examples/serve_pp2.sh` launches a two-node pipeline-parallel server, and `examples/pp_client.py`, `pp_batch.py` and `pp_concurrent.py` are clients. Real runs need a local model; never download model weights for these scripts.

See [pipeline parallelism](../guides/pipeline-parallelism.md) for the topology and [CUDA and Metal](../guides/cuda-metal.md) for the Mac GPU pairing.
