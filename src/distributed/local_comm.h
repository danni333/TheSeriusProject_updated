#pragma once

#include "comm.h"

#include <condition_variable>
#include <mutex>
#include <vector>

namespace serius {
namespace distributed {

/// Local communicator for single-process testing and simulation.
///
/// In "single rank" mode (world_size=1), all collective operations
/// are no-ops (the data is already "reduced" trivially).
///
/// In "simulated multi-rank" mode, multiple LocalCommunicator instances
/// share a ProcessGroup that coordinates their collective operations
/// using shared memory (std::vector buffers + mutex/condvar).
/// This enables testing distributed algorithms without NCCL/MPI.

/// Shared state for simulating multi-rank communication within a single process.
/// One ProcessGroup is shared by all LocalCommunicator instances in the group.
class ProcessGroup
{
   public:
    explicit ProcessGroup(int world_size);

    int world_size() const { return world_size_; }

    /// Submit data for an AllReduce and wait for all ranks to participate.
    /// Returns the reduced result in `buffer`.
    void all_reduce(float *buffer, size_t count, ReduceOp op, int rank);

    /// Submit data for a Broadcast and wait.
    void broadcast(float *buffer, size_t count, int root, int rank);

    /// AllGather: each rank submits send_count floats, receives all concatenated.
    void all_gather(const float *send_buf, size_t send_count,
                    float *recv_buf, int rank);

    /// ReduceScatter: reduce then scatter.
    void reduce_scatter(const float *send_buf, float *recv_buf,
                        size_t recv_count, ReduceOp op, int rank);

    /// Point-to-point send/recv using internal mailboxes.
    void send(const float *buffer, size_t count, int src, int dst);
    void recv(float *buffer, size_t count, int src, int dst);

    /// Barrier synchronization.
    void barrier(int rank);

   private:
    int world_size_;
    std::mutex mutex_;
    std::condition_variable cv_;

    // Staging area for collective ops
    int arrived_ = 0;              // Number of ranks that have arrived
    int generation_ = 0;           // Incremented after each collective completes
    std::vector<std::vector<float>> rank_buffers_;  // Per-rank staging data
    std::vector<float> result_buffer_;              // Reduced result

    // Point-to-point mailboxes: mailbox_[src][dst] = data
    struct Mailbox
    {
        std::vector<float> data;
        bool ready = false;
        bool consumed = false;
    };
    std::vector<std::vector<Mailbox>> mailboxes_;  // [src][dst]

    void wait_for_all(std::unique_lock<std::mutex> &lock, int generation);
    void wait_for_last_rank(std::unique_lock<std::mutex> &lock, int generation);
};

class LocalCommunicator : public Communicator
{
   public:
    /// Single-rank mode (world_size=1, rank=0)
    LocalCommunicator();

    /// Multi-rank simulation mode: attach to a shared ProcessGroup
    LocalCommunicator(std::shared_ptr<ProcessGroup> group, int rank);

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }
    CommBackend backend() const override { return CommBackend::Local; }

    void all_reduce(float *buffer, size_t count, ReduceOp op) override;
    void broadcast(float *buffer, size_t count, int root) override;
    void all_gather(const float *send_buf, size_t send_count, float *recv_buf) override;
    void reduce_scatter(const float *send_buf, float *recv_buf, size_t recv_count, ReduceOp op) override;
    void send(const float *buffer, size_t count, int dst) override;
    void recv(float *buffer, size_t count, int src) override;
    void barrier() override;

   private:
    int rank_;
    int world_size_;
    std::shared_ptr<ProcessGroup> group_;
};

/// Helper: create a group of LocalCommunicator instances for testing.
/// Returns world_size communicators that share a ProcessGroup.
std::vector<std::unique_ptr<LocalCommunicator>> create_local_group(int world_size);

}  // namespace distributed
}  // namespace serius
