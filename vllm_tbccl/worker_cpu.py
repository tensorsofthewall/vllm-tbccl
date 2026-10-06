"""CPU worker for heterogeneous pipelines (see ``worker.py``): KV-cache layout agreement and the persistent PP receive buffer a CPU eager worker never creates."""
from vllm.distributed import get_pp_group
from vllm.v1.worker.cpu_worker import CPUWorker

from .worker import _HeteroLayouts


class TbcclCpuWorker(_HeteroLayouts, CPUWorker):
    _tbccl_kind = "cpu"

    def load_model(self, *args, **kwargs):
        out = super().load_model(*args, **kwargs)
        runner = self.model_runner
        if hasattr(runner, "intermediate_tensors") and runner.intermediate_tensors is None and not get_pp_group().is_first_rank:
            runner.intermediate_tensors = runner.model.make_empty_intermediate_tensors(
                batch_size=runner.max_num_tokens, dtype=runner.model_config.dtype, device=runner.device)
        return out
