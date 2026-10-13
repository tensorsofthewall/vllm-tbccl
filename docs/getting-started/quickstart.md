# Quickstart: a two-node PP=2 pipeline

This is a complete launch of one vLLM pipeline across two machines: a **Linux CUDA node** (node 0, serves the API) and a **Mac Metal node** (node 1, headless), joined by a direct Thunderbolt 4 link. It is the configuration that was validated with the release-candidate wheels. A CUDA and CPU pair is at the end.

## What you need

- vllm-tbccl [installed](install.md) on **both** machines (the same version), vLLM 0.31.0 on both (Python 3.12 on both is the validated combination), and for the Metal node the vllm-metal fork tag `v0.31.0-tbccl1` ([install](install.md)).
- A model directory that exists locally on **both** machines (this guide uses Qwen3-0.6B). Nothing here downloads weights.
- A direct network path between the nodes that both can reach. The example uses a Thunderbolt link: Linux `192.168.3.2`, Mac `192.168.3.1` ([link setup](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-link.html)). A plain LAN does not work for the pipeline because the executor's reverse-direction connections are blocked by ordinary firewalls ([limitations](../project/limitations.md)).
- On the Mac, run the node in a live terminal or SSH session and answer the macOS firewall / Local Network prompt once.

## Roles and addresses

| | Linux CUDA node | Mac Metal node |
|---|---|---|
| vLLM role | node rank 0: serves the OpenAI API, `--master-addr` | node rank 1: `--headless` |
| link address | `192.168.3.2` (`LIP`) | `192.168.3.1` (`MIP`) |
| TBCCL backend | `VLLM_TBCCL_BACKEND` unset (CUDA platform) | `VLLM_TBCCL_BACKEND=metal` (vllm-metal owns the platform) |
| pipeline stage | first layers | later layers (the layer split is automatic) |

## 1. Common flags

Both nodes use the same model arguments:

```sh
COMMON="--served-model-name metal-pp-model --dtype bfloat16 --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
        --max-model-len 1024 --pipeline-parallel-size 2 --tensor-parallel-size 1 \
        --nnodes 2 --master-addr 192.168.3.2 --master-port 29600 --distributed-executor-backend mp"
```

## 2. Start the Linux node (node 0)

```sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=192.168.3.2 TBCCL_LOCAL_ENDPOINT=192.168.3.2:0
vllm serve /path/to/Qwen3-0.6B $COMMON --node-rank 0 --gpu-memory-utilization 0.5 --host 192.168.3.2 --port 8170
```

## 3. Start the Mac node (node 1, headless)

```sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_BACKEND=metal VLLM_TBCCL_ENABLE=1 \
       VLLM_HOST_IP=192.168.3.1 TBCCL_LOCAL_ENDPOINT=192.168.3.1:0
vllm serve ~/models/Qwen3-0.6B $COMMON --node-rank 1 --gpu-memory-utilization 0.3 --headless
```

`TBCCL_LOCAL_ENDPOINT=<host>:0` lets every communicator choose its own free port pair on that host's link address. The port must be 0.

Start the two commands within a minute or so of each other; the leader waits for the headless node. When the API server logs that it is ready, the pipeline is up.

## 4. Query it

```sh
curl http://192.168.3.2:8170/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "metal-pp-model", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

`examples/pp_client.py` queries the server with reference prompts and compares token ids to a reference, and `pp_batch.py` and `pp_concurrent.py` are further clients.

## 5. Confirm TBCCL is carrying it

The Linux log contains `Platform plugin tbccl is activated` and `world_size=2 rank=0 ... backend=tbccl`; see [how do I know TBCCL is being used?](../guides/verify-active.md).

## 6. Stop

Stop the **leader first** (Ctrl-C in its terminal, or `kill -TERM` of the `vllm serve` process). The headless Mac process can keep running after a graceful leader stop; stop it explicitly. This is vLLM's lifecycle behavior ([limitations](../project/limitations.md)). If the headless node's worker dies, the leader does not notice: use client timeouts.

## CUDA node and CPU node (vLLM 0.31.0)

Use vLLM 0.31.0 on both machines and no vllm-metal. On the Mac, set `VLLM_TBCCL_BACKEND=torch` if vllm-metal is also installed there (so vllm-tbccl owns the CPU platform instead of declining it). `examples/serve_pp2.sh` launches one node:

```sh
# Linux CUDA, node 0 (serves the API on port 8000)
VLLM_USE_V2_MODEL_RUNNER=0 examples/serve_pp2.sh 0 <master addr> <master port> <this host's link address> /path/to/model
# Mac CPU, node 1 (headless)
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_BACKEND=torch examples/serve_pp2.sh 1 <master addr> <master port> <this host's link address> ~/path/to/model
```

The script uses `--dtype float32` and a short context so a small model fits; edit it for your model. This pairing was physically validated on TBCCL wire protocol 4 with vLLM 0.31.0; see [validated configurations](../project/validated-configurations.md).
