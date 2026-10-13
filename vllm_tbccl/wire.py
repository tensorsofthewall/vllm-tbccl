"""vLLM's pipeline tensor-dict wire format, reproduced exactly (GroupCoordinator.send_object/recv_object, identical in vLLM 0.30.0 and 0.31.0; pinned by tests/test_wire_vllm_parity.py):
an 8-byte int64 size tensor then a pickled uint8 payload on the CPU (control) group; tensors afterwards on the device group."""
import pickle

import torch
import torch.distributed as dist


def send_object(obj, dst_global_rank: int, cpu_group) -> None:
    payload = torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
    size = torch.tensor([payload.numel()], dtype=torch.long, device="cpu")
    dist.send(size, dst=dst_global_rank, group=cpu_group)
    dist.send(payload, dst=dst_global_rank, group=cpu_group)


def recv_object(src_global_rank: int, cpu_group):
    size = torch.empty(1, dtype=torch.long, device="cpu")
    dist.recv(size, src=src_global_rank, group=cpu_group)
    payload = torch.empty(int(size.item()), dtype=torch.uint8, device="cpu")
    dist.recv(payload, src=src_global_rank, group=cpu_group)
    return pickle.loads(payload.numpy().tobytes())
