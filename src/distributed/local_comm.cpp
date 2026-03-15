#include "local_comm.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace serius {
namespace distributed {

// ============================================================
// ProcessGroup
// ============================================================

ProcessGroup::ProcessGroup(int world_size) : world_size_(world_size)
{
    rank_buffers_.resize(static_cast<size_t>(world_size));
    mailboxes_.resize(static_cast<size_t>(world_size), std::vector<Mailbox>(static_cast<size_t>(world_size)));
}

void ProcessGroup::wait_for_all(std::unique_lock<std::mutex> &lock, int generation)
{
    cv_.wait(lock, [&] { return arrived_ == world_size_ && generation_ == generation; });
}

void ProcessGroup::wait_for_last_rank(std::unique_lock<std::mutex> &lock, int generation)
{
    // Wait until all ranks have consumed the result (generation advanced)
    cv_.wait(lock, [&] { return generation_ != generation; });
}

void ProcessGroup::all_reduce(float *buffer, size_t count, ReduceOp op, int rank)
{
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;

    // Copy this rank's data into staging
    rank_buffers_[static_cast<size_t>(rank)].assign(buffer, buffer + count);
    arrived_++;

    if (arrived_ == world_size_) {
        // Last rank to arrive performs the reduction
        result_buffer_.resize(count);

        // Initialize result based on operation
        switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Avg:
                std::fill(result_buffer_.begin(), result_buffer_.end(), 0.0f);
                break;
            case ReduceOp::Min:
                std::fill(result_buffer_.begin(), result_buffer_.end(), std::numeric_limits<float>::max());
                break;
            case ReduceOp::Max:
                std::fill(result_buffer_.begin(), result_buffer_.end(), std::numeric_limits<float>::lowest());
                break;
        }

        for (int r = 0; r < world_size_; ++r) {
            for (size_t i = 0; i < count; ++i) {
                float val = rank_buffers_[static_cast<size_t>(r)][i];
                switch (op) {
                    case ReduceOp::Sum:
                    case ReduceOp::Avg:
                        result_buffer_[i] += val;
                        break;
                    case ReduceOp::Min:
                        result_buffer_[i] = std::min(result_buffer_[i], val);
                        break;
                    case ReduceOp::Max:
                        result_buffer_[i] = std::max(result_buffer_[i], val);
                        break;
                }
            }
        }

        if (op == ReduceOp::Avg) {
            float inv = 1.0f / static_cast<float>(world_size_);
            for (size_t i = 0; i < count; ++i) {
                result_buffer_[i] *= inv;
            }
        }

        // Signal all waiting ranks
        cv_.notify_all();
    } else {
        // Wait for all ranks
        wait_for_all(lock, gen);
    }

    // Copy result back
    std::memcpy(buffer, result_buffer_.data(), count * sizeof(float));

    // Synchronize departure
    arrived_--;
    if (arrived_ == 0) {
        generation_++;
        cv_.notify_all();
    } else {
        wait_for_last_rank(lock, gen);
    }
}

void ProcessGroup::broadcast(float *buffer, size_t count, int root, int rank)
{
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;

    if (rank == root) {
        result_buffer_.assign(buffer, buffer + count);
    }

    arrived_++;
    if (arrived_ == world_size_) {
        cv_.notify_all();
    } else {
        wait_for_all(lock, gen);
    }

    // Copy broadcast data
    if (rank != root) {
        std::memcpy(buffer, result_buffer_.data(), count * sizeof(float));
    }

    arrived_--;
    if (arrived_ == 0) {
        generation_++;
        cv_.notify_all();
    } else {
        wait_for_last_rank(lock, gen);
    }
}

void ProcessGroup::all_gather(const float *send_buf, size_t send_count, float *recv_buf, int rank)
{
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;

    rank_buffers_[static_cast<size_t>(rank)].assign(send_buf, send_buf + send_count);
    arrived_++;

    if (arrived_ == world_size_) {
        cv_.notify_all();
    } else {
        wait_for_all(lock, gen);
    }

    // Concatenate all rank buffers into recv_buf
    for (int r = 0; r < world_size_; ++r) {
        std::memcpy(recv_buf + static_cast<size_t>(r) * send_count, rank_buffers_[static_cast<size_t>(r)].data(),
                    send_count * sizeof(float));
    }

    arrived_--;
    if (arrived_ == 0) {
        generation_++;
        cv_.notify_all();
    } else {
        wait_for_last_rank(lock, gen);
    }
}

void ProcessGroup::reduce_scatter(const float *send_buf, float *recv_buf, size_t recv_count, ReduceOp op, int rank)
{
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;

    size_t total = recv_count * static_cast<size_t>(world_size_);
    rank_buffers_[static_cast<size_t>(rank)].assign(send_buf, send_buf + total);
    arrived_++;

    if (arrived_ == world_size_) {
        // Reduce across all ranks, then scatter
        result_buffer_.resize(total);

        switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Avg:
                std::fill(result_buffer_.begin(), result_buffer_.end(), 0.0f);
                break;
            case ReduceOp::Min:
                std::fill(result_buffer_.begin(), result_buffer_.end(), std::numeric_limits<float>::max());
                break;
            case ReduceOp::Max:
                std::fill(result_buffer_.begin(), result_buffer_.end(), std::numeric_limits<float>::lowest());
                break;
        }

        for (int r = 0; r < world_size_; ++r) {
            for (size_t i = 0; i < total; ++i) {
                float val = rank_buffers_[static_cast<size_t>(r)][i];
                switch (op) {
                    case ReduceOp::Sum:
                    case ReduceOp::Avg:
                        result_buffer_[i] += val;
                        break;
                    case ReduceOp::Min:
                        result_buffer_[i] = std::min(result_buffer_[i], val);
                        break;
                    case ReduceOp::Max:
                        result_buffer_[i] = std::max(result_buffer_[i], val);
                        break;
                }
            }
        }

        if (op == ReduceOp::Avg) {
            float inv = 1.0f / static_cast<float>(world_size_);
            for (size_t i = 0; i < total; ++i) {
                result_buffer_[i] *= inv;
            }
        }

        cv_.notify_all();
    } else {
        wait_for_all(lock, gen);
    }

    // Each rank gets its own chunk
    size_t offset = static_cast<size_t>(rank) * recv_count;
    std::memcpy(recv_buf, result_buffer_.data() + offset, recv_count * sizeof(float));

    arrived_--;
    if (arrived_ == 0) {
        generation_++;
        cv_.notify_all();
    } else {
        wait_for_last_rank(lock, gen);
    }
}

void ProcessGroup::send(const float *buffer, size_t count, int src, int dst)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto &mb = mailboxes_[static_cast<size_t>(src)][static_cast<size_t>(dst)];

    // Wait until previous message is consumed
    cv_.wait(lock, [&] { return !mb.ready || mb.consumed; });

    mb.data.assign(buffer, buffer + count);
    mb.ready = true;
    mb.consumed = false;
    cv_.notify_all();
}

void ProcessGroup::recv(float *buffer, size_t count, int src, int dst)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto &mb = mailboxes_[static_cast<size_t>(src)][static_cast<size_t>(dst)];

    // Wait for data to be ready
    cv_.wait(lock, [&] { return mb.ready && !mb.consumed; });

    assert(mb.data.size() == count);
    std::memcpy(buffer, mb.data.data(), count * sizeof(float));
    mb.consumed = true;
    mb.ready = false;
    cv_.notify_all();
}

void ProcessGroup::barrier(int /*rank*/)
{
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;

    arrived_++;
    if (arrived_ == world_size_) {
        cv_.notify_all();
    } else {
        wait_for_all(lock, gen);
    }

    arrived_--;
    if (arrived_ == 0) {
        generation_++;
        cv_.notify_all();
    } else {
        wait_for_last_rank(lock, gen);
    }
}

// ============================================================
// LocalCommunicator
// ============================================================

LocalCommunicator::LocalCommunicator() : rank_(0), world_size_(1), group_(nullptr) {}

LocalCommunicator::LocalCommunicator(std::shared_ptr<ProcessGroup> group, int rank)
    : rank_(rank), world_size_(group->world_size()), group_(std::move(group))
{
}

void LocalCommunicator::all_reduce(float *buffer, size_t count, ReduceOp op)
{
    if (world_size_ == 1) return;  // No-op for single rank
    group_->all_reduce(buffer, count, op, rank_);
}

void LocalCommunicator::all_reduce(core::Tensor &tensor, ReduceOp op)
{
    // For local simulation, we normally operate on host data
    tensor.syncToHost();
    all_reduce(tensor.data(), tensor.size(), op);
    // Mark dirty if we were on GPU (simulated)
    // Actually, tensor.data() syncs to host, so it's clean on host now.
}

void LocalCommunicator::broadcast(float *buffer, size_t count, int root)
{
    if (world_size_ == 1) return;
    group_->broadcast(buffer, count, root, rank_);
}

void LocalCommunicator::broadcast(core::Tensor &tensor, int root)
{
    tensor.syncToHost();
    broadcast(tensor.data(), tensor.size(), root);
}

void LocalCommunicator::all_gather(const float *send_buf, size_t send_count, float *recv_buf)
{
    if (world_size_ == 1) {
        std::memcpy(recv_buf, send_buf, send_count * sizeof(float));
        return;
    }
    group_->all_gather(send_buf, send_count, recv_buf, rank_);
}

void LocalCommunicator::all_gather(const core::Tensor &send_tensor, core::Tensor &recv_tensor)
{
    const_cast<core::Tensor&>(send_tensor).syncToHost();
    recv_tensor.syncToHost();
    all_gather(send_tensor.data(), send_tensor.size(), recv_tensor.data());
}

void LocalCommunicator::reduce_scatter(const float *send_buf, float *recv_buf, size_t recv_count, ReduceOp op)
{
    if (world_size_ == 1) {
        std::memcpy(recv_buf, send_buf, recv_count * sizeof(float));
        return;
    }
    group_->reduce_scatter(send_buf, recv_buf, recv_count, op, rank_);
}

void LocalCommunicator::send(const float *buffer, size_t count, int dst)
{
    if (group_) {
        group_->send(buffer, count, rank_, dst);
    }
}

void LocalCommunicator::recv(float *buffer, size_t count, int src)
{
    if (group_) {
        group_->recv(buffer, count, src, rank_);
    }
}

void LocalCommunicator::barrier()
{
    if (world_size_ == 1) return;
    group_->barrier(rank_);
}

// ============================================================
// Factory
// ============================================================

std::vector<std::unique_ptr<LocalCommunicator>> create_local_group(int world_size)
{
    auto group = std::make_shared<ProcessGroup>(world_size);
    std::vector<std::unique_ptr<LocalCommunicator>> comms;
    comms.reserve(static_cast<size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
        comms.push_back(std::make_unique<LocalCommunicator>(group, r));
    }
    return comms;
}

std::unique_ptr<Communicator> Communicator::create(CommBackend backend, int world_size, int /*rank*/)
{
    switch (backend) {
        case CommBackend::Local:
            if (world_size == 1) {
                return std::make_unique<LocalCommunicator>();
            }
            // For multi-rank local, caller should use create_local_group()
            // Here we create a standalone single-rank communicator
            return std::make_unique<LocalCommunicator>();

        case CommBackend::NCCL:
#ifdef SERIUS_USE_NCCL
            // TODO: return std::make_unique<NCCLCommunicator>(world_size, rank);
#endif
            throw std::runtime_error("NCCL backend not available (compile with SERIUS_USE_NCCL)");

        case CommBackend::MPI:
#ifdef SERIUS_USE_MPI
            // TODO: return std::make_unique<MPICommunicator>();
#endif
            throw std::runtime_error("MPI backend not available (compile with SERIUS_USE_MPI)");
    }

    throw std::runtime_error("Unknown communicator backend");
}

}  // namespace distributed
}  // namespace serius
