"""Apply the generic vLLM 0.30.0 hooks vllm-tbccl needs (idempotent; plain-text replacements on the installed package).

1. parallel_state.py: GroupCoordinator's CPU/control group backend comes from VLLM_CPU_GROUP_BACKEND (default "gloo" = unchanged).
   PyTorch's gloo does not rendezvous between the Linux and macOS builds, so a Linux<->macOS world cannot use it.
2. v1/attention/backends/utils.py: resolve_kv_cache_layout() asserted that every worker reports the identical list of supported
   KV-cache layouts ("all ranks run the same backends"). A CUDA worker and a CPU worker legitimately differ; resolve from the
   intersection instead (same behavior when the lists agree).
3. v1/worker/gpu_model_runner.py: sync_and_gather_intermediate_tensors() asserted a persistent intermediate-tensor buffer that is only
   created inside _dummy_run(). The CPU model runner skips profile_run()/warm-up under --enforce-eager, so a non-first pipeline stage on
   CPU crashed on its first batch. Allocate it lazily (same call _dummy_run uses).
See docs/vllm_api_audit.md and vllm_tbccl/patches/*.patch.
"""
import pathlib
import sys

import vllm

ROOT = pathlib.Path(vllm.__file__).parent
EDITS = [
    (
        "distributed/parallel_state.py",
        '''                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(
                        ranks, backend="gloo", timeout=timeout
                    )''',
        '''                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(
                        ranks,
                        backend=os.environ.get("VLLM_CPU_GROUP_BACKEND", "gloo"),
                        timeout=timeout,
                    )''',
        ("\nimport pickle\n", "\nimport os\nimport pickle\n"),
    ),
    (
        "v1/attention/backends/utils.py",
        '''    assert all(names == supported_layouts[0] for names in supported_layouts[1:]), (
        f"Workers disagree on supported KV cache layouts: {supported_layouts}."
    )
    candidates = [_layout_from_name(name) for name in supported_layouts[0]]''',
        '''    # Workers may support different layout sets (e.g. a CUDA and a CPU worker); keep the layouts every worker supports,
    # in the first worker's preference order. Identical lists behave exactly as before.
    common = [
        name
        for name in supported_layouts[0]
        if all(name in names for names in supported_layouts[1:])
    ]
    assert common, f"No KV cache layout is supported by every worker: {supported_layouts}."
    candidates = [_layout_from_name(name) for name in common]''',
        None,
    ),
    (
        "v1/worker/gpu_model_runner.py",
        '''    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size''',
        '''    ) -> IntermediateTensors:
        if self.intermediate_tensors is None:
            # Normally created by _dummy_run(); runners that skip profiling/warm-up (e.g. CPU with --enforce-eager) get it here.
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

        tp = self.vllm_config.parallel_config.tensor_parallel_size''',
        None,
    ),
]

rc = 0
for rel, old, new, extra in EDITS:
    path = ROOT / rel
    src = path.read_text()
    if new in src:
        print("already applied:", rel)
        continue
    if src.count(old) != 1:
        print("unexpected vLLM source, NOT applied:", rel)
        rc = 1
        continue
    src = src.replace(old, new)
    if extra and extra[1] not in src:
        src = src.replace(*extra, 1)
    path.write_text(src)
    print("patched:", rel)
sys.exit(rc)
