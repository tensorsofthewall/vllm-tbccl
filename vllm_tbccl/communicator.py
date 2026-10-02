"""TBCCLDeviceCommunicator: a vLLM DeviceCommunicator over torch.distributed with the ProcessGroupTBCCL device group.

It owns no transport and no algorithm. It never touches NCCL/PyNccl/custom all-reduce. Besides the base-class send/recv/
broadcast (already plain torch.distributed on `device_group`) it provides the communicator-supplied tensor-dict fast
path that vLLM's CPU platform uses, made wire-compatible with vLLM's generic path so a CPU rank and a CUDA rank can be
pipeline neighbours:

  generic path (CUDA rank):  metadata object over the cpu (gloo) group, tensors via isend on the device group, receiver
                             allocates a tensor of the *sender's* device type.
  this class (any rank):     same wire format, but the receiver allocates on ITS OWN device, and the sender labels each
                             tensor with the PEER's device type (exchanged once at construction) so a generic-path peer
                             allocates correctly. Tensors always travel over the device (TBCCL) group, never gloo.

Operations vLLM does not need for PP=2/TP=1 raise NotImplementedError.
"""
import pickle

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from vllm.distributed.parallel_state import TensorMetadata, _split_tensor_dict

from . import diagnostics as diag


class TBCCLDeviceCommunicator(DeviceCommunicatorBase):
    supports_tensor_dict = True

    def __init__(self, cpu_group, device=None, device_group=None, unique_name="", **kwargs):
        super().__init__(cpu_group, device, device_group, unique_name, **kwargs)
        if self.world_size != 2:
            raise NotImplementedError(
                f"TBCCLDeviceCommunicator supports 2-rank groups only (group {unique_name!r} has {self.world_size})")
        backend = dist.get_backend(device_group) if device_group is not None else None
        if backend != "tbccl":
            raise RuntimeError(f"TBCCLDeviceCommunicator needs a device_group created with backend 'tbccl' (got {backend})")
        gathered = [None, None]
        dist.all_gather_object(gathered, self.device.type, group=self.cpu_group)
        self.peer_device_type = gathered[1 - self.rank_in_group]

    # -- plain tensor ops (all over the TBCCL device group) ------------------------------------------------------------
    def all_reduce(self, input_):
        # DeviceCommunicatorBase.all_reduce (vLLM 0.30.0) is in place and returns its input; keep that contract.
        t0 = diag.now_ns()
        dist.all_reduce(input_, group=self.device_group)
        diag.record("all_reduce", input_.nbytes, input_.shape, input_.dtype, t0, diag.now_ns())
        return input_

    def send(self, tensor, dst=None):
        dst = (self.rank_in_group + 1) % self.world_size if dst is None else dst
        t0 = diag.now_ns()
        dist.send(tensor.contiguous(), dst=self.ranks[dst], group=self.device_group)
        diag.record("send", tensor.nbytes, tensor.shape, tensor.dtype, t0, diag.now_ns())

    def recv(self, size, dtype, src=None):
        src = (self.rank_in_group - 1) % self.world_size if src is None else src
        out = torch.empty(size, dtype=dtype, device=self.device)
        t0 = diag.now_ns()
        dist.recv(out, src=self.ranks[src], group=self.device_group)
        diag.record("recv", out.nbytes, out.shape, out.dtype, t0, diag.now_ns())
        return out

    def broadcast(self, tensor, src=0):
        t0 = diag.now_ns()
        dist.broadcast(tensor, src=self.ranks[src], group=self.device_group)
        diag.record("broadcast", tensor.nbytes, tensor.shape, tensor.dtype, t0, diag.now_ns())
        return tensor

    def all_gather(self, input_, dim=-1):
        if dim < 0:
            dim += input_.dim()
        outs = [torch.empty_like(input_) for _ in range(self.world_size)]
        t0 = diag.now_ns()
        dist.all_gather(outs, input_.contiguous(), group=self.device_group)
        diag.record("all_gather", input_.nbytes, input_.shape, input_.dtype, t0, diag.now_ns())
        return torch.cat(outs, dim=dim)

    def reduce_scatter(self, input_, dim=-1):
        raise NotImplementedError("vllm-tbccl: reduce_scatter is not implemented (not used by PP=2/TP=1)")

    def gather(self, input_, dst=0, dim=-1):
        raise NotImplementedError("vllm-tbccl: gather is not implemented")

    def all_gatherv(self, *a, **k):
        raise NotImplementedError("vllm-tbccl: all_gatherv is not implemented")

    def reduce_scatterv(self, *a, **k):
        raise NotImplementedError("vllm-tbccl: reduce_scatterv is not implemented")

    # -- tensor-dict fast path -----------------------------------------------------------------------------------------
    def _send_object(self, obj, dst):  # identical wire format to GroupCoordinator.send_object
        payload = torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
        size = torch.tensor([payload.numel()], dtype=torch.long, device="cpu")
        dist.send(size, dst=self.ranks[dst], group=self.cpu_group)
        dist.send(payload, dst=self.ranks[dst], group=self.cpu_group)

    def _recv_object(self, src):  # identical wire format to GroupCoordinator.recv_object
        size = torch.empty(1, dtype=torch.long, device="cpu")
        dist.recv(size, src=self.ranks[src], group=self.cpu_group)
        payload = torch.empty(int(size.item()), dtype=torch.uint8, device="cpu")
        dist.recv(payload, src=self.ranks[src], group=self.cpu_group)
        return pickle.loads(payload.numpy().tobytes())

    def send_tensor_dict(self, tensor_dict, dst):
        metadata, tensors = _split_tensor_dict(tensor_dict)
        relabeled = [
            (k, TensorMetadata(self.peer_device_type, v.dtype, v.size)) if isinstance(v, TensorMetadata) else (k, v)
            for k, v in metadata
        ]
        self._send_object(relabeled, dst)
        for t in tensors:
            if t.numel() == 0:
                continue
            self.send(t, dst)

    def recv_tensor_dict(self, src):
        metadata = self._recv_object(src)
        out = {}
        for key, value in metadata:
            if isinstance(value, TensorMetadata):
                if int(torch.Size(value.size).numel()) == 0:
                    out[key] = torch.empty(value.size, dtype=value.dtype, device=self.device)
                else:
                    out[key] = self.recv(value.size, value.dtype, src)
            else:
                out[key] = value
        return out
