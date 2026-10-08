// One persistent thread per process group: waits, in FIFO order, for each submitted TBCCL operation and completes (or fails) its c10 Future from C++.
// It never touches the Python API and never destroys a WorkState: a tensor's last reference can be the one keeping its Python object alive, and dropping it
// takes the GIL, which deadlocks against a caller that joins this thread while holding the GIL. Finished states are retired to a list that callers
// drain (take_retired) and destroy on their own thread.
#pragma once

#include "common.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace vllm_tbccl
{

class CompletionWorker
{
public:
    CompletionWorker() : thread_([this] { run(); }) {}
    ~CompletionWorker() { stop(); }

    void enqueue(std::shared_ptr<WorkState> state)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(state));
        }
        cv_.notify_one();
    }
    // Finishes the queued items, then joins. Idempotent; the caller should not hold the GIL.
    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    std::vector<std::shared_ptr<WorkState>> take_retired()
    {
        std::vector<std::shared_ptr<WorkState>> out;
        std::lock_guard<std::mutex> lock(mutex_);
        out.swap(retired_);
        return out;
    }

private:
    void run()
    {
        for (;;)
        {
            std::shared_ptr<WorkState> state;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                state = std::move(queue_.front());
                queue_.pop_front();
            }
            std::string err;
            const auto rc = state->wait(&err);
            if (rc != TBCCL_SUCCESS)
                state->future->setError(std::make_exception_ptr(std::runtime_error("vllm-tbccl: " + state->op_name + " failed: " + err)));
            else
                state->future->markCompleted(c10::IValue(state->tensors));
            std::lock_guard<std::mutex> lock(mutex_);
            retired_.push_back(std::move(state));
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<WorkState>> queue_;
    std::vector<std::shared_ptr<WorkState>> retired_;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace vllm_tbccl
