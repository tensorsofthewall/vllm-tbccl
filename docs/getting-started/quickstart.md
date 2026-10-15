# Quickstart: a two-node PP=2 pipeline

This is a complete launch of one vLLM pipeline across two machines: a **Linux CUDA node** (node 0, serves the API) and a **Mac Metal node** (node 1, headless), joined by a direct Thunderbolt 4 link. It is the validated configuration (PP=2, TP=1, Qwen3-0.6B). A CUDA and CPU pair is at the end. Commands use the environments from [installing](install.md): `.venv` on the Linux node and the fork's `vllm-metal/.venv-vllm-metal` on the Metal Mac; the paths are relative to where you ran the install.

## What you need

- vllm-tbccl [installed](install.md) on **both** machines (the same version), vLLM 0.31.0 on both (Python 3.12 on both is the validated combination), and for the Metal node the vllm-metal fork tag `v0.31.0-tbccl1` ([install](install.md)).
- A model directory that exists locally on **both** machines (this guide uses Qwen3-0.6B). Nothing here downloads weights.
- A verified direct link between the nodes. If the two machines have never been connected, start with [Two machines over Thunderbolt](https://tbccl.tensorsofthewall.com/en/latest/getting-started/two-machine-thunderbolt.html) (ping **and** TCP must work in both directions). The commands below use the addresses from that page, Linux `192.168.77.2` and Mac `192.168.77.1`; they are variables, so put your own in (`192.168.3.x` is only the validated pair's configuration). A plain LAN does not work for the pipeline because the executor's reverse-direction connections are blocked by ordinary firewalls ([limitations](../project/limitations.md)).
- On the Mac, run the node in a live terminal or SSH session and answer the macOS firewall / Local Network prompt once.

## Roles and addresses

| | Linux CUDA node | Mac Metal node |
|---|---|---|
| vLLM role | node rank 0: serves the OpenAI API, `--master-addr` | node rank 1: `--headless` |
| link address | `LIP` (example `192.168.77.2`) | `MIP` (example `192.168.77.1`) |
| TBCCL backend | `VLLM_TBCCL_BACKEND` unset (CUDA platform) | `VLLM_TBCCL_BACKEND=metal` (vllm-metal owns the platform) |
| pipeline stage | first layers | later layers (the layer split is automatic) |

## 1. Common settings

Set these on **both** machines, in each terminal that starts a node. The two link addresses and the model path are the only values to change; `MODEL` is a directory that already exists on that machine (nothing is downloaded):

```sh
LIP=192.168.77.2      # Linux node's link address
MIP=192.168.77.1      # Mac node's link address
MODEL=/path/to/Qwen3-0.6B        # a local model directory on THIS machine
```

Both nodes use the same model arguments:

```sh
COMMON="--served-model-name metal-pp-model --dtype bfloat16 --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
        --max-model-len 1024 --pipeline-parallel-size 2 --tensor-parallel-size 1 \
        --nnodes 2 --master-addr $LIP --master-port 29600 --distributed-executor-backend mp"
```

## 2. Start the Linux node (node 0)

```sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_ENABLE=1 VLLM_HOST_IP=$LIP TBCCL_LOCAL_ENDPOINT=$LIP:0
.venv/bin/vllm serve $MODEL $COMMON --node-rank 0 --gpu-memory-utilization 0.5 --host $LIP --port 8170
```

## 3. Start the Mac node (node 1, headless)

```sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_TBCCL_BACKEND=metal VLLM_TBCCL_ENABLE=1 \
       VLLM_HOST_IP=$MIP TBCCL_LOCAL_ENDPOINT=$MIP:0
vllm-metal/.venv-vllm-metal/bin/vllm serve $MODEL $COMMON --node-rank 1 --gpu-memory-utilization 0.3 --headless
```

`--gpu-memory-utilization` is the fraction of that machine's GPU memory vLLM may take; 0.5 and 0.3 are the values used when this pair was validated with a small model. Lower it if the memory is shared with other work, and raise it only after the first run works. `VLLM_USE_V2_MODEL_RUNNER=0` must be set on **both** nodes. The model on the Mac must be a local copy of the same model.

`TBCCL_LOCAL_ENDPOINT=<host>:0` lets every communicator choose its own free port pair on that host's link address. The port must be 0.

Start the two commands within a minute or so of each other; the leader waits for the headless node. Start the Linux node first; it waits for the headless node. When the API server logs that it is ready, the pipeline is up. Port 29600 is vLLM's own rendezvous on the Linux node's link address; it must be free there, and the Linux host firewall must allow the Mac's link address to reach it ([troubleshooting](https://tbccl.tensorsofthewall.com/en/latest/guides/thunderbolt-troubleshooting.html)).

## 4. Query it

```sh
curl http://$LIP:8170/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "metal-pp-model", "prompt": "The capital of France is", "max_tokens": 16, "temperature": 0}'
```

`examples/pp_client.py` queries the server with reference prompts and compares token ids to a reference, and `pp_batch.py` and `pp_concurrent.py` are further clients.

## 5. Confirm TBCCL is carrying it

The Linux log contains `Platform plugin tbccl is activated` (the Metal Mac's log says `Platform plugin metal is activated`, which is correct there) and `world_size=2 rank=0 ... backend=tbccl`; see [how do I know TBCCL is being used?](../guides/verify-active.md).

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
