#include "tensor_adapter.hpp"

#ifdef VLLM_TBCCL_WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

namespace vllm_tbccl
{

tbcclDataType_t to_tbccl_dtype(at::ScalarType type)
{
    switch (type)
    {
    case at::kFloat: return TBCCL_FLOAT32;
    case at::kDouble: return TBCCL_FLOAT64;
    case at::kHalf: return TBCCL_FLOAT16;
    case at::kBFloat16: return TBCCL_BFLOAT16;
    case at::kInt: return TBCCL_INT32;
    case at::kLong: return TBCCL_INT64;
    case at::kChar: return TBCCL_INT8;
    case at::kByte: return TBCCL_UINT8;
    default:
        TORCH_CHECK_NOT_IMPLEMENTED(
            false, "vllm-tbccl: unsupported operation: dtype ", type,
            " has no reduction (all_reduce supports float16, bfloat16, float32, float64, int8, uint8, int32, int64; send/recv, broadcast and all_gather move any dense dtype as raw bytes)");
    }
}

void require_sum(const c10d::ReduceOp &op)
{
    TORCH_CHECK_NOT_IMPLEMENTED(op.op_ == c10d::ReduceOp::SUM, "vllm-tbccl: unsupported operation: only the SUM reduction is supported");
}

TbcclBuffer to_tbccl_buffer(const at::Tensor &t, bool bytes_only)
{
    TORCH_CHECK_VALUE(t.defined(), "vllm-tbccl: invalid argument: undefined tensor");
    TORCH_CHECK_NOT_IMPLEMENTED(
        t.layout() == at::kStrided && !t.is_quantized(), "vllm-tbccl: unsupported operation: only dense strided tensors are supported (got layout ", t.layout(), ")");
    const bool is_cuda = t.device().type() == at::kCUDA;
    bool device_ok = t.device().type() == at::kCPU;
#ifdef VLLM_TBCCL_WITH_CUDA
    device_ok = device_ok || is_cuda;
#endif
    TORCH_CHECK_NOT_IMPLEMENTED(
        device_ok, "vllm-tbccl: unsupported operation: device ", t.device(), is_cuda ? " (this build has no CUDA support)" : " (supported: CPU, CUDA)");
    TORCH_CHECK_VALUE(t.is_contiguous(), "vllm-tbccl: invalid argument: tensor must be contiguous (no implicit copy is made)");

    TbcclBuffer out{};
    if (!bytes_only) out.datatype = to_tbccl_dtype(t.scalar_type());
    out.count = static_cast<uint64_t>(t.numel());
    out.buffer.struct_size = sizeof(tbcclBuffer);
    out.buffer.bytes = static_cast<uint64_t>(t.nbytes());
    out.buffer.data = t.numel() == 0 ? nullptr : t.data_ptr();
    out.context.struct_size = sizeof(tbcclExecContext);
    if (!is_cuda)
    {
        out.buffer.memory_kind = TBCCL_MEMORY_HOST;
        out.buffer.device_ordinal = -1;
        out.context.kind = TBCCL_EXEC_DEFAULT;
        return out;
    }
#ifdef VLLM_TBCCL_WITH_CUDA
    // The producer stream is PyTorch's current stream for this device at submission; TBCCL owns the cross-stream event dependency.
    out.buffer.memory_kind = TBCCL_MEMORY_CUDA;
    out.buffer.device_ordinal = t.device().index();
    out.context.kind = TBCCL_EXEC_CUDA_STREAM;
    out.context.native_handle = reinterpret_cast<void *>(c10::cuda::getCurrentCUDAStream(t.device().index()).stream());
#endif
    return out;
}

} // namespace vllm_tbccl
