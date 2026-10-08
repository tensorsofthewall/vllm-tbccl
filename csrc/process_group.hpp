// vllm-tbccl's private c10d backend ("tbccl"): the minimal ProcessGroup vLLM's GroupCoordinator needs, written directly against libtbccl's C ABI.
// Supported: all_reduce (SUM), broadcast, all_gather, 2-rank gather, barrier, send/recv. Every other collective throws NotImplementedError.
// This is deliberately not torch-tbccl: it shares no code, no package and no registration with it.
#pragma once

#include "completion_worker.hpp"
#include "tensor_adapter.hpp"

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include <map>
#include <mutex>

namespace vllm_tbccl
{

// Per-process count and bytes of every operation submitted to libtbccl (a cheap proof that traffic took this backend); not a timeline.
std::map<std::string, std::pair<uint64_t, uint64_t>> op_stats();
void reset_op_stats();

class ProcessGroupTBCCL : public c10d::Backend
{
public:
    ProcessGroupTBCCL(const c10::intrusive_ptr<c10d::Store> &store, int rank, int world_size, std::chrono::milliseconds timeout);
    ~ProcessGroupTBCCL() override;

    const std::string getBackendName() const override { return "tbccl"; }
    void setTimeout(std::chrono::milliseconds timeout) override { timeout_ = timeout; }
    void shutdown() override;
    // Communicator-wide and destructive: outstanding Works fail and later operations throw.
    void abort() override;

    c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor> &tensors, const c10d::AllreduceOptions &opts = c10d::AllreduceOptions()) override;
    c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor> &tensors, const c10d::BroadcastOptions &opts = c10d::BroadcastOptions()) override;
    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>> &outputTensors, std::vector<at::Tensor> &inputTensors,
        const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;
    c10::intrusive_ptr<c10d::Work> gather(
        std::vector<std::vector<at::Tensor>> &outputTensors, std::vector<at::Tensor> &inputTensors,
        const c10d::GatherOptions &opts = c10d::GatherOptions()) override;
    c10::intrusive_ptr<c10d::Work> barrier(const c10d::BarrierOptions &opts = c10d::BarrierOptions()) override;
    c10::intrusive_ptr<c10d::Work> send(std::vector<at::Tensor> &tensors, int dstRank, int tag) override;
    c10::intrusive_ptr<c10d::Work> recv(std::vector<at::Tensor> &tensors, int srcRank, int tag) override;

private:
    void reap();
    void require_peers(const char *op) const;
    std::shared_ptr<WorkState> new_state(const char *name, std::vector<at::Tensor> tensors) const;
    c10::intrusive_ptr<c10d::Work> trivially_done(std::shared_ptr<WorkState> state, c10d::OpType op);
    c10::intrusive_ptr<c10d::Work> submit(std::shared_ptr<WorkState> state, c10d::OpType op);
    c10::intrusive_ptr<c10d::Work> p2p(std::vector<at::Tensor> &tensors, int peer, bool is_send);

    c10::intrusive_ptr<c10d::Store> store_;
    std::chrono::milliseconds timeout_;
    // Guards comm_/completion_ and serializes submission; never held while waiting for an operation.
    mutable std::mutex mutex_;
    tbcclComm_t comm_ = nullptr;
    bool shut_down_ = false;
    std::unique_ptr<CompletionWorker> completion_;
};

} // namespace vllm_tbccl
