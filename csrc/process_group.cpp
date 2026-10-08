#include "process_group.hpp"

#include <Python.h>

#include <tbccl/tbccl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>

namespace vllm_tbccl
{

namespace
{

constexpr uint32_t kMaxWorldSize = 4;
constexpr const char *kLocalEndpointEnv = "TBCCL_LOCAL_ENDPOINT";
// One record set per group generation: every rank joins a group exactly once by incrementing the counter, so generation = (joins - 1) / world_size is the
// same on every rank and differs between successive groups created over the SAME store (a persistent store reuses its key prefix across re-init).
constexpr const char *kJoinCounterKey = "vllm_tbccl/v1/joined";
constexpr const char *kGenerationPrefix = "vllm_tbccl/v1/g";

std::mutex g_stats_mutex;
std::map<std::string, std::pair<uint64_t, uint64_t>> g_stats; // operation -> (count, bytes), this process

void ensure_cuda_support()
{
#ifdef VLLM_TBCCL_WITH_CUDA
    static std::once_flag once;
    std::call_once(once, [] { check_call("register CUDA support", tbcclRegisterCudaSupport()); });
#endif
}

// Joining the completion thread must not hold the GIL (it may be waiting for it inside a Future callback). A group that Python garbage-collects at
// interpreter exit is destroyed with the GIL held, hence the explicit release.
class ReleaseGil
{
public:
    ReleaseGil()
    {
        if (Py_IsInitialized() && PyGILState_Check()) state_ = PyEval_SaveThread();
    }
    ~ReleaseGil()
    {
        if (state_) PyEval_RestoreThread(state_);
    }
    ReleaseGil(const ReleaseGil &) = delete;
    ReleaseGil &operator=(const ReleaseGil &) = delete;

private:
    PyThreadState *state_ = nullptr;
};

// TBCCL_LOCAL_ENDPOINT=<ipv4>:0 -> the address this rank listens on and advertises. The port must be 0: the C ABI always lets the kernel choose.
std::string local_host_from_env()
{
    const char *value = std::getenv(kLocalEndpointEnv);
    TORCH_CHECK_VALUE(
        value != nullptr && *value != '\0', "vllm-tbccl: invalid argument: ", kLocalEndpointEnv,
        " is not set; set it to this rank's address with port 0, e.g. 127.0.0.1:0");
    const std::string text = value;
    const auto colon = text.rfind(':');
    TORCH_CHECK_VALUE(colon != std::string::npos && colon > 0, "vllm-tbccl: invalid argument: ", kLocalEndpointEnv, "='", text, "' is not of the form host:0");
    const std::string host = text.substr(0, colon);
    const std::string port = text.substr(colon + 1);
    TORCH_CHECK_VALUE(host.find(':') == std::string::npos, "vllm-tbccl: invalid argument: ", kLocalEndpointEnv, "='", text, "': IPv6 literals are not supported");
    TORCH_CHECK_VALUE(
        port == "0", "vllm-tbccl: invalid argument: ", kLocalEndpointEnv, "='", text,
        "': vllm-tbccl lets the kernel choose each communicator's ports; use <host>:0");
    return host;
}

std::vector<uint8_t> bytes_of(const void *p, size_t n)
{
    const auto *b = static_cast<const uint8_t *>(p);
    return std::vector<uint8_t>(b, b + n);
}

struct BootstrapGuard
{
    tbcclBootstrap_t handle = nullptr;
    ~BootstrapGuard() { tbcclBootstrapDestroy(handle); }
};

tbcclComm_t bootstrap_communicator(const c10::intrusive_ptr<c10d::Store> &store, int rank, int world_size, std::chrono::milliseconds timeout)
{
    const std::string host = local_host_from_env();

    // Joining is the first Store access, after local validation, so a rank that fails locally consumes no generation slot.
    const int64_t joined = store->add(kJoinCounterKey, 1);
    const std::string ns = std::string(kGenerationPrefix) + std::to_string((joined - 1) / world_size) + "/";
    const std::string id_key = ns + "id";
    auto blob_key = [&ns](int r) { return ns + "endpoint/" + std::to_string(r); };

    tbcclUniqueId id{};
    if (rank == 0)
    {
        check_call("unique id", tbcclGetUniqueId(&id));
        store->set(id_key, bytes_of(id.bytes, sizeof(id.bytes)));
    }
    try
    {
        store->wait({id_key}, timeout);
    }
    catch (const std::exception &e)
    {
        TORCH_CHECK(false, "vllm-tbccl: bootstrap timeout: rank 0 did not publish the communicator id within ", timeout.count(), " ms (", e.what(), ")");
    }
    {
        const auto raw = store->get(id_key);
        TORCH_CHECK(raw.size() == sizeof(id.bytes), "vllm-tbccl: bootstrap: malformed communicator id in the store");
        std::memcpy(id.bytes, raw.data(), sizeof(id.bytes));
    }

    tbcclBootstrapOptions opts{};
    opts.struct_size = sizeof(opts);
    opts.bind_host = host.c_str();
    opts.timeout_ms = static_cast<uint32_t>(std::min<int64_t>(timeout.count(), UINT32_MAX));
    BootstrapGuard bs;
    check_call("bootstrap begin", tbcclBootstrapBegin(static_cast<uint32_t>(rank), static_cast<uint32_t>(world_size), &id, &opts, &bs.handle));

    tbcclEndpointBlob mine{};
    mine.struct_size = sizeof(mine);
    check_call("bootstrap endpoint", tbcclBootstrapGetEndpoint(bs.handle, &mine));
    store->set(blob_key(rank), bytes_of(&mine, sizeof(mine)));

    std::vector<std::string> keys;
    for (int r = 0; r < world_size; ++r) keys.push_back(blob_key(r));
    try
    {
        store->wait(keys, timeout);
    }
    catch (const std::exception &e)
    {
        TORCH_CHECK(
            false, "vllm-tbccl: bootstrap timeout: not all ranks published a TBCCL endpoint within ", timeout.count(), " ms (", e.what(), ")");
    }
    std::vector<tbcclEndpointBlob> blobs(world_size);
    for (int r = 0; r < world_size; ++r)
    {
        const auto raw = store->get(blob_key(r));
        TORCH_CHECK_VALUE(raw.size() == sizeof(tbcclEndpointBlob), "vllm-tbccl: invalid argument: rank ", r, " published a malformed endpoint record");
        std::memcpy(&blobs[r], raw.data(), sizeof(tbcclEndpointBlob));
    }
    tbcclComm_t comm = nullptr;
    const auto rc = tbcclBootstrapComplete(bs.handle, blobs.data(), static_cast<uint32_t>(world_size), &comm);
    if (rc != TBCCL_SUCCESS) throw_result("could not create the TBCCL communicator", rc, tbcclGetResultString(rc));
    return comm;
}

} // namespace

std::map<std::string, std::pair<uint64_t, uint64_t>> op_stats()
{
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    return g_stats;
}

void reset_op_stats()
{
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_stats.clear();
}

ProcessGroupTBCCL::ProcessGroupTBCCL(const c10::intrusive_ptr<c10d::Store> &store, int rank, int world_size, std::chrono::milliseconds timeout)
    : c10d::Backend(rank, world_size), store_(store), timeout_(timeout)
{
    TORCH_CHECK_NOT_IMPLEMENTED(
        world_size >= 1 && static_cast<uint32_t>(world_size) <= kMaxWorldSize, "vllm-tbccl supports world_size 1 to ", kMaxWorldSize, " (got ", world_size, ")");
    TORCH_CHECK_VALUE(rank >= 0 && rank < world_size, "vllm-tbccl: invalid argument: rank ", rank, " outside [0, ", world_size, ")");
    TORCH_CHECK_VALUE(store_ != nullptr, "vllm-tbccl: invalid argument: store is null");

    ensure_cuda_support();
    // A one-rank group (vLLM's TP=1 groups) has no peer and needs no communicator; collectives on it are rejected.
    if (world_size == 1) return;
    comm_ = bootstrap_communicator(store_, rank, world_size, timeout);
    completion_ = std::make_unique<CompletionWorker>();
}

ProcessGroupTBCCL::~ProcessGroupTBCCL()
{
    shutdown();
}

void ProcessGroupTBCCL::shutdown()
{
    // Reject new submissions, destroy the communicator (it fails any still-outstanding operation locally, so a silent peer cannot block teardown and
    // every pending Work settles; unlike abort() it does not tell the peers the group failed), then join the completion worker, which finishes the Futures.
    tbcclComm_t doomed = nullptr;
    std::unique_ptr<CompletionWorker> worker;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shut_down_ = true;
        doomed = comm_;
        comm_ = nullptr;
        worker = std::move(completion_);
    }
    if (doomed) tbcclCommDestroy(doomed);
    if (worker)
    {
        {
            ReleaseGil nogil;
            worker->stop();
        }
        worker.reset(); // finished states (and their tensors) die here, on a thread that may take the GIL
    }
}

void ProcessGroupTBCCL::abort()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (comm_) tbcclCommAbort(comm_, "ProcessGroup abort");
}

void ProcessGroupTBCCL::reap()
{
    std::vector<std::shared_ptr<WorkState>> finished;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completion_) finished = completion_->take_retired();
    }
} // `finished` dies here: no lock held, on the submitting thread

void ProcessGroupTBCCL::require_peers(const char *op) const
{
    TORCH_CHECK_NOT_IMPLEMENTED(
        getSize() >= 2, "vllm-tbccl: unsupported operation: ", op, " needs a group of at least 2 ranks (rank ", getRank(), " is alone in a one-rank group)");
}

std::shared_ptr<WorkState> ProcessGroupTBCCL::new_state(const char *name, std::vector<at::Tensor> tensors) const
{
    auto state = std::make_shared<WorkState>();
    state->tensors = std::move(tensors);
    state->op_name = name;
    state->future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    return state;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::trivially_done(std::shared_ptr<WorkState> state, c10d::OpType op)
{
    state->future->markCompleted(c10::IValue(state->tensors));
    return c10::make_intrusive<WorkTBCCL>(getRank(), op, state);
}

// Hands a state whose works were just created (under mutex_) to the completion worker.
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::submit(std::shared_ptr<WorkState> state, c10d::OpType op)
{
    {
        uint64_t bytes = 0;
        for (const auto &t : state->tensors) bytes += static_cast<uint64_t>(t.nbytes());
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        auto &entry = g_stats[state->op_name];
        entry.first += 1;
        entry.second += bytes;
    }
    completion_->enqueue(state);
    return c10::make_intrusive<WorkTBCCL>(getRank(), op, state);
}

#define LOCKED_COMM(name)                                                                                              \
    std::lock_guard<std::mutex> lock(mutex_);                                                                          \
    require_peers(name);                                                                                               \
    TORCH_CHECK(comm_ != nullptr && !shut_down_, "vllm-tbccl: communicator failure: process group has been shut down")

#define ADD_WORK(state, name, call)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        tbcclWork_t w_ = nullptr;                                                                                      \
        const tbcclResult_t rc_ = (call);                                                                              \
        if (rc_ != TBCCL_SUCCESS) throw_result(name, rc_, tbcclGetResultString(rc_));                                   \
        (state)->works.push_back(w_);                                                                                  \
    } while (0)

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allreduce(std::vector<at::Tensor> &tensors, const c10d::AllreduceOptions &opts)
{
    reap();
    TORCH_CHECK_VALUE(tensors.size() == 1, "vllm-tbccl: invalid argument: allreduce takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_NOT_IMPLEMENTED(!opts.sparseIndices.has_value(), "vllm-tbccl: unsupported operation: sparse allreduce");
    require_sum(opts.reduceOp);
    const auto buf = to_tbccl_buffer(tensors[0]);
    auto state = new_state("allreduce", tensors);
    if (buf.count == 0) return trivially_done(state, c10d::OpType::ALLREDUCE);

    LOCKED_COMM("allreduce");
    tbcclWork_t w_ = nullptr;
    const auto rc = tbcclAllReduce(comm_, &buf.buffer, &buf.buffer, buf.count, buf.datatype, TBCCL_SUM, &buf.context, &w_);
    if (rc != TBCCL_SUCCESS) throw_result("allreduce", rc, tbcclGetResultString(rc));
    state->works.push_back(w_);
    return submit(state, c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::broadcast(std::vector<at::Tensor> &tensors, const c10d::BroadcastOptions &opts)
{
    reap();
    TORCH_CHECK_VALUE(tensors.size() == 1, "vllm-tbccl: invalid argument: broadcast takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_VALUE(
        opts.rootRank >= 0 && opts.rootRank < getSize(), "vllm-tbccl: invalid argument: broadcast rootRank ", opts.rootRank, " outside [0, ", getSize(), ")");
    TORCH_CHECK_NOT_IMPLEMENTED(opts.rootTensor == 0, "vllm-tbccl: unsupported operation: broadcast rootTensor must be 0 (got ", opts.rootTensor, ")");
    const auto buf = to_tbccl_buffer(tensors[0], true);
    auto state = new_state("broadcast", tensors);
    if (buf.buffer.bytes == 0) return trivially_done(state, c10d::OpType::BROADCAST);

    LOCKED_COMM("broadcast");
    ADD_WORK(state, "broadcast", tbcclBroadcast(comm_, &buf.buffer, static_cast<uint32_t>(opts.rootRank), &buf.context, &w_));
    return submit(state, c10d::OpType::BROADCAST);
}

// all_gather into a list of separate output tensors: one broadcast per rank (root r sends its input; the others receive into outputs[r]), then the local
// copy. No staging buffer is allocated and the outputs are written in place.
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::allgather(
    std::vector<std::vector<at::Tensor>> &outputTensors, std::vector<at::Tensor> &inputTensors, const c10d::AllgatherOptions &)
{
    reap();
    TORCH_CHECK_VALUE(
        inputTensors.size() == 1 && outputTensors.size() == 1, "vllm-tbccl: invalid argument: allgather takes one input tensor and one output list (got ",
        inputTensors.size(), " inputs, ", outputTensors.size(), " output lists)");
    const auto &input = inputTensors[0];
    auto &outs = outputTensors[0];
    TORCH_CHECK_VALUE(
        static_cast<int>(outs.size()) == getSize(), "vllm-tbccl: invalid argument: allgather output list must have world_size (", getSize(), ") tensors (got ",
        outs.size(), ")");
    const auto in_buf = to_tbccl_buffer(input, true);
    std::vector<TbcclBuffer> out_bufs;
    for (const auto &o : outs)
    {
        out_bufs.push_back(to_tbccl_buffer(o, true));
        TORCH_CHECK_VALUE(
            o.scalar_type() == input.scalar_type() && o.numel() == input.numel() && o.device() == input.device(),
            "vllm-tbccl: invalid argument: allgather outputs must match the input's dtype, numel and device");
    }
    auto state = new_state("allgather", outs);
    state->retained = {input};
    if (in_buf.buffer.bytes == 0) return trivially_done(state, c10d::OpType::ALLGATHER);

    LOCKED_COMM("allgather");
    for (int r = 0; r < getSize(); ++r)
    {
        const TbcclBuffer &b = r == getRank() ? in_buf : out_bufs[r];
        ADD_WORK(state, "allgather", tbcclBroadcast(comm_, &b.buffer, static_cast<uint32_t>(r), &b.context, &w_));
    }
    outs[getRank()].copy_(input);
    return submit(state, c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::gather(
    std::vector<std::vector<at::Tensor>> &outputTensors, std::vector<at::Tensor> &inputTensors, const c10d::GatherOptions &opts)
{
    TORCH_CHECK_VALUE(inputTensors.size() == 1, "vllm-tbccl: invalid argument: gather takes exactly one input tensor per rank");
    TORCH_CHECK_VALUE(opts.rootRank >= 0 && opts.rootRank < getSize(), "vllm-tbccl: invalid argument: gather rootRank out of range");
    TORCH_CHECK_NOT_IMPLEMENTED(
        getSize() == 2, "vllm-tbccl: unsupported operation: gather needs a 2-rank group (this group has ", getSize(), " ranks; rank ", getRank(), ")");
    if (getRank() != opts.rootRank) return p2p(inputTensors, opts.rootRank, true);
    TORCH_CHECK_VALUE(
        outputTensors.size() == 1 && static_cast<int>(outputTensors[0].size()) == getSize(),
        "vllm-tbccl: invalid argument: gather on the root needs one output list of world_size tensors");
    auto &outs = outputTensors[0];
    const int peer = 1 - getRank();
    TORCH_CHECK_VALUE(
        outs[peer].scalar_type() == inputTensors[0].scalar_type() && outs[peer].numel() == inputTensors[0].numel() &&
            outs[getRank()].numel() == inputTensors[0].numel(),
        "vllm-tbccl: invalid argument: gather outputs must match the input's dtype and numel");
    outs[getRank()].copy_(inputTensors[0]);
    std::vector<at::Tensor> slot{outs[peer]};
    return p2p(slot, peer, false);
}

// Two ranks: a one-element SUM all-reduce (a synchronizing collective with a payload); more: the communicator's own barrier.
c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::barrier(const c10d::BarrierOptions &)
{
    reap();
    std::vector<at::Tensor> token{at::zeros({1}, at::kFloat)};
    if (getSize() == 1) return trivially_done(new_state("barrier", token), c10d::OpType::BARRIER);
    if (getSize() == 2) return allreduce(token);
    auto state = new_state("barrier", token);
    LOCKED_COMM("barrier");
    ADD_WORK(state, "barrier", tbcclBarrier(comm_, &w_));
    return submit(state, c10d::OpType::BARRIER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::send(std::vector<at::Tensor> &tensors, int dstRank, int)
{
    return p2p(tensors, dstRank, true);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::recv(std::vector<at::Tensor> &tensors, int srcRank, int)
{
    return p2p(tensors, srcRank, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupTBCCL::p2p(std::vector<at::Tensor> &tensors, int peer, bool is_send)
{
    reap();
    const char *name = is_send ? "send" : "recv";
    TORCH_CHECK_VALUE(tensors.size() == 1, "vllm-tbccl: invalid argument: ", name, " takes exactly one tensor (got ", tensors.size(), ")");
    TORCH_CHECK_VALUE(
        peer >= 0 && peer < getSize() && peer != getRank(), "vllm-tbccl: invalid argument: ", name, " peer rank ", peer, " must be another rank of this group");
    // The payload is opaque bytes (any dense dtype); transfers are matched by FIFO order per direction, `tag` is ignored.
    const auto buf = to_tbccl_buffer(tensors[0], true);
    auto state = new_state(name, tensors);
    const auto op = is_send ? c10d::OpType::SEND : c10d::OpType::RECV;
    if (buf.buffer.bytes == 0) return trivially_done(state, op);

    LOCKED_COMM(name);
    if (is_send)
        ADD_WORK(state, name, tbcclSend(comm_, &buf.buffer, static_cast<uint32_t>(peer), &buf.context, &w_));
    else
        ADD_WORK(state, name, tbcclRecv(comm_, &buf.buffer, static_cast<uint32_t>(peer), &buf.context, &w_));
    return submit(state, op);
}

} // namespace vllm_tbccl
