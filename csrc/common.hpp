// Shared helpers for vllm-tbccl's private c10d backend: result-code mapping and the Work state shared between the user-visible Work and the completion thread.
// Everything here talks to libtbccl through its stable C ABI (tbccl/tbccl.h) only.
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <tbccl/tbccl.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace vllm_tbccl
{

inline std::string work_error_text(tbcclWork_t work, tbcclResult_t rc)
{
    std::string text = tbcclGetResultString(rc);
    size_t required = 0;
    if (work && tbcclWorkGetErrorString(work, nullptr, 0, &required) == TBCCL_SUCCESS && required > 1)
    {
        std::string buf(required, '\0');
        if (tbcclWorkGetErrorString(work, buf.data(), buf.size(), &required) == TBCCL_SUCCESS) text = buf.c_str();
    }
    return text;
}

[[noreturn]] inline void throw_result(const std::string &what, tbcclResult_t rc, const std::string &detail)
{
    switch (rc)
    {
    case TBCCL_INVALID_ARGUMENT: TORCH_CHECK_VALUE(false, "vllm-tbccl: invalid argument: ", what, ": ", detail);
    case TBCCL_UNSUPPORTED: TORCH_CHECK_NOT_IMPLEMENTED(false, "vllm-tbccl: unsupported operation: ", what, ": ", detail);
    default: break;
    }
    const char *category = "communicator failure";
    switch (rc)
    {
    case TBCCL_TRANSPORT_ERROR: category = "transport error"; break;
    case TBCCL_TIMEOUT: category = "timeout"; break;
    case TBCCL_DEVICE_ERROR: category = "device error"; break;
    case TBCCL_ABORTED: category = "aborted"; break;
    case TBCCL_RESOURCE_EXHAUSTED: category = "resource exhausted"; break;
    case TBCCL_PROTOCOL_MISMATCH: category = "protocol mismatch"; break;
    case TBCCL_INTERNAL_ERROR: category = "internal error"; break;
    default: break;
    }
    TORCH_CHECK(false, "vllm-tbccl: ", category, ": ", what, ": ", detail);
}

inline void check_call(const char *what, tbcclResult_t rc)
{
    if (rc != TBCCL_SUCCESS) throw_result(what, rc, tbcclGetResultString(rc));
}

// Owns one or more libtbccl Work handles plus every tensor the operation reads or writes: a tbcclBuffer is non-owning, so the tensors must outlive the
// operation even when the caller drops its Work early.
struct WorkState
{
    std::vector<at::Tensor> tensors; // what Work::result() and the Future yield
    std::vector<at::Tensor> retained;
    std::vector<tbcclWork_t> works;  // empty = trivially complete (zero bytes)
    c10::intrusive_ptr<c10::ivalue::Future> future;
    std::string op_name;

    WorkState() = default;
    WorkState(const WorkState &) = delete;
    WorkState &operator=(const WorkState &) = delete;
    ~WorkState()
    {
        for (auto w : works) tbcclWorkDestroy(w);
    }

    bool is_completed() const
    {
        for (auto w : works)
        {
            int32_t done = 0;
            tbcclResult_t op = TBCCL_SUCCESS;
            if (tbcclWorkTest(w, &done, &op) == TBCCL_SUCCESS && !done) return false;
        }
        return true;
    }
    // Waits for every handle; returns the first failure (and its text) or TBCCL_SUCCESS.
    tbcclResult_t wait(std::string *error)
    {
        tbcclResult_t first = TBCCL_SUCCESS;
        for (auto w : works)
        {
            tbcclResult_t op = TBCCL_SUCCESS;
            tbcclResult_t rc = tbcclWorkWait(w, &op);
            if (rc != TBCCL_SUCCESS) op = rc;
            if (op != TBCCL_SUCCESS && first == TBCCL_SUCCESS)
            {
                first = op;
                if (error) *error = work_error_text(w, op);
            }
        }
        return first;
    }
    tbcclResult_t status(std::string *error) const
    {
        for (auto w : works)
        {
            int32_t done = 0;
            tbcclResult_t op = TBCCL_SUCCESS;
            if (tbcclWorkTest(w, &done, &op) == TBCCL_SUCCESS && done && op != TBCCL_SUCCESS)
            {
                if (error) *error = work_error_text(w, op);
                return op;
            }
        }
        return TBCCL_SUCCESS;
    }
};

class WorkTBCCL : public c10d::Work
{
public:
    WorkTBCCL(int rank, c10d::OpType op, std::shared_ptr<WorkState> state)
        : c10d::Work(rank, op, nullptr, state->tensors), state_(std::move(state))
    {
    }
    bool isCompleted() override { return state_->is_completed(); }
    bool isSuccess() const override { return state_->status(nullptr) == TBCCL_SUCCESS; }
    std::exception_ptr exception() const override
    {
        std::string err;
        const auto rc = state_->status(&err);
        if (rc == TBCCL_SUCCESS) return nullptr;
        return std::make_exception_ptr(std::runtime_error("vllm-tbccl: " + state_->op_name + " failed: " + err));
    }
    // timeout == 0 (c10d kNoTimeout): block until done. Otherwise a bounded wait that does not cancel the TBCCL operation.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override
    {
        if (timeout.count() == 0)
        {
            std::string err;
            const auto rc = state_->wait(&err);
            if (rc != TBCCL_SUCCESS) throw_result(state_->op_name, rc, err);
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (auto w : state_->works)
        {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            int32_t done = 0;
            tbcclResult_t op = TBCCL_SUCCESS;
            check_call("wait", tbcclWorkWaitFor(w, left.count() > 0 ? static_cast<uint64_t>(left.count()) : 0, &done, &op));
            TORCH_CHECK(
                done, "vllm-tbccl: timeout: ", state_->op_name, " did not complete within ", timeout.count(),
                " ms (the underlying TBCCL operation is not cancelled and may still complete)");
            if (op != TBCCL_SUCCESS) throw_result(state_->op_name, op, work_error_text(w, op));
        }
        return true;
    }
    void synchronize() override { wait(); }
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return state_->future; }
    std::vector<at::Tensor> result() override { return state_->tensors; }

private:
    std::shared_ptr<WorkState> state_;
};

} // namespace vllm_tbccl
