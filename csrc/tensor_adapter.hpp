// The single narrow conversion between PyTorch tensors and libtbccl's C ABI buffer description. Never copies and never calls .contiguous():
// the tbcclBuffer aliases the tensor's own storage. CPU and CUDA tensors only (vllm-metal hands the transport CPU aliases of its MLX arrays).
#pragma once

#include "common.hpp"

#include <torch/csrc/distributed/c10d/Types.hpp>

namespace vllm_tbccl
{

struct TbcclBuffer
{
    tbcclBuffer buffer;
    tbcclExecContext context;
    uint64_t count = 0;
    tbcclDataType_t datatype = TBCCL_FLOAT32;
};

// bytes_only: send/recv/broadcast/all_gather move any dense contiguous dtype as raw bytes; reductions need a TBCCL numeric dtype.
TbcclBuffer to_tbccl_buffer(const at::Tensor &t, bool bytes_only = false);
tbcclDataType_t to_tbccl_dtype(at::ScalarType type);
void require_sum(const c10d::ReduceOp &op);

} // namespace vllm_tbccl
