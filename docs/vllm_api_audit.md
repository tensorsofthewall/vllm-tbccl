# vLLM 0.30.0 API audit (pinned: PyPI wheel `vllm==0.30.0`, requires `torch==2.13.0`)

Environment (Linux): uv venv, CPython 3.13, torch 2.13.0+cu130, vllm 0.30.0, transformers 5.18.0, driver 610.43.02, RTX 3070 Ti Laptop 8 GB.
torch-tbccl builds and loads unchanged against torch 2.13.0 (its documented tested version is 2.14.1; both work). The Mac has no vLLM wheel: it needs a
source build of the **same 0.30.0 tag** for the CPU target (not yet done).

## Group construction (`vllm/distributed/parallel_state.py`)
* `init_distributed_environment(backend=...)` calls `torch.distributed.init_process_group(backend=<platform.dist_backend>)` (WORLD). The backend string
  must be identical on every rank. With `dist_backend="tbccl"` this creates a TBCCL communicator for WORLD.
* `GroupCoordinator.__init__` (legacy `new_group` path, default) does, **for every group in `group_ranks` on every rank** (collective on WORLD):
  `new_group(ranks, backend=<same backend>)` (device_group) and `new_group(ranks, backend="gloo")` (cpu_group). `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`
  (default off) would use `split_group` instead; not used.
* For PP=2, TP=1, DP=1, world 2 the process builds: WORLD (2 ranks), TP groups `[0]`,`[1]`, PP group `[0,1]`, DP groups `[0]`,`[1]`, plus EP/other
  groups depending on config; each is a separate device_group + gloo group. Consequences for TBCCL: (a) **one-rank groups** are created (TP=1, DP=1): the
  backend factory is called with `world_size=1` -> torch-tbccl now accepts that (no communicator); (b) **several 2-rank groups coexist in one process**
  (WORLD + PP + more): each needs its own control/data port pair. torch-tbccl now supports `TBCCL_LOCAL_ENDPOINT=<host>:0` = pick a free pair per communicator;
  endpoints are published through the group-namespaced (PrefixStore) c10d Store, so keys never collide. One TBCCL group is sufficient for PP=2 traffic.
* `new_group` invokes the backend factory once per member rank per group; non-members get `NON_GROUP_MEMBER` (no factory call).
* Device communicator: `current_platform.get_device_communicator_cls()` (resolved via `resolve_obj_by_qualname`) is instantiated for groups with
  `use_device_communicator=True and world_size > 1` (TP, PP, DP, EP groups; WORLD is created with `use_device_communicator=False`).
* Built-in `CudaCommunicator` constructs `PyNcclCommunicator` for multi-rank CUDA groups regardless of other settings => not usable; replaced via the platform hook.

## Extension point (no vLLM patch needed)
`vllm.platform_plugins` entry point -> returns a platform class qualname; one out-of-tree plugin wins over the built-in detection. `vllm-tbccl` ships a subclass
of the host's own `CudaPlatform` / `CpuPlatform` overriding only `dist_backend="tbccl"` and `get_device_communicator_cls()`. Workers, attention backends,
kernels and allocators are inherited untouched. Active only with `VLLM_TBCCL_ENABLE=1`.

## PP=2 communication call sites (mp executor, V1 runner, `--no-async-scheduling`)
* Stage output: `gpu_worker.execute_model` -> `get_pp_group().isend_tensor_dict(output.tensors, ...)`; next stage: `irecv_tensor_dict(...)`
  (`IntermediateTensors`: `hidden_states`, `residual`, both `[num_tokens, hidden]` in the model dtype). `recv` handles are waited at the top of the next step.
* Generic `isend_tensor_dict`: metadata list `(key, TensorMetadata(device.type, dtype, size))` pickled over the **cpu (gloo) group** (size tensor + payload tensor),
  then one `torch.distributed.isend` per tensor on `device_group` if the tensor is on an accelerator, **else on the cpu group (gloo)**; receiver allocates
  `torch.empty(..., device=<sender's device type>)`.
* `broadcast_tensor_dict` / `_pp_broadcast_prev_sampled_token_ids` (sibling device group) are only used by the `external_launcher` backend and by async scheduling;
  both avoided here. No `all_reduce`/`all_gather`/`reduce_scatter` occurs with TP=1.
* Control plane (scheduler outputs) uses the executor's shared-memory/TCP `MessageQueue` and `broadcast_object` over gloo; it is not on TBCCL.

## Heterogeneity invariants found
1. **Device type baked into tensor metadata** (`parallel_state._split_tensor_dict`, `irecv_tensor_dict`): a CPU receiver of a CUDA sender would allocate a CUDA tensor, and a CPU
   sender's tensors would travel over gloo while the CUDA receiver listens on the device group. The generic path cannot connect a CUDA rank to a CPU rank.
2. **Bridge without patching vLLM:** `GroupCoordinator.use_cpu_custom_send_recv = current_platform.is_cpu() and device_communicator.supports_tensor_dict` makes the *CPU platform*
   delegate `send/recv_tensor_dict` to its device communicator. `TBCCLDeviceCommunicator` implements those with the **same wire format** as the generic path but (a) relabels each tensor's
   device with the peer's device type (exchanged once at construction) and (b) always moves tensors over the TBCCL device group and allocates on the receiver's own device.
   The CUDA rank keeps vLLM's generic path unmodified. Verified at group level, CUDA<->CPU in both directions (`examples/tbccl_group_smoke.py`).
3. Launch: `distributed_executor_backend="mp"` supports `--nnodes N --node-rank k` (+ `--headless` workers, TCP message queues); Ray/other executors not used.
   `--tensor-parallel-size 2` is impossible for this model anyway (9 attention heads).
4. Mac side still to establish: CPU worker (`CPUWorker`) / OpenMP affinity manager (`OMPProcessManager`) on macOS, torch 2.13 CPU wheel, vLLM CPU source build.

## Defaults used
dtype float32, `--enforce-eager`, `--no-async-scheduling`, no CUDA graphs/compile tuning, greedy, `max-model-len 512`, `--gpu-memory-utilization 0.3` when two vLLM
processes share the one GPU.
