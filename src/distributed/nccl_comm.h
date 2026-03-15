#pragma once

#ifdef SERIUS_USE_NCCL

#include "comm.h"

#include <nccl.h>

namespace serius {
namespace distributed {

/// NCCL-based communicator for intra-node GPU communication.
///
/// NCCL provides highly optimized collective operations that leverage
/// NVLink, PCIe, and network interconnects. This is the recommended
/// backend for multi-GPU training on a single node.
///
/// Initialization requires all participating GPUs to call ncclCommInitRank
/// with the same ncclUniqueId (distributed via MPI or file system).
class NCCLCommunicator : public Communicator
{
   public:
    /// Initialize NCCL communicator.
    /// @param world_size  Total number of GPUs
    /// @param rank        This GPU's rank (0-based)
    /// @param id          Shared unique ID (obtained from ncclGetUniqueId on rank 0)
    NCCLCommunicator(int world_size, int rank, ncclUniqueId id);
    ~NCCLCommunicator() override;

    // Non-copyable
    NCCLCommunicator(const NCCLCommunicator &) = delete;
    NCCLCommunicator &operator=(const NCCLCommunicator &) = delete;

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }
    CommBackend backend() const override { return CommBackend::NCCL; }

    void all_reduce(float *buffer, size_t count, ReduceOp op) override;
    void all_reduce(core::Tensor &tensor, ReduceOp op) override;

    void broadcast(float *buffer, size_t count, int root) override;
    void broadcast(core::Tensor &tensor, int root = 0) override;

    void all_gather(const float *send_buf, size_t send_count, float *recv_buf) override;
    void all_gather(const core::Tensor &send_tensor, core::Tensor &recv_tensor) override;
    void reduce_scatter(const float *send_buf, float *recv_buf, size_t recv_count, ReduceOp op) override;
    void send(const float *buffer, size_t count, int dst) override;
    void recv(float *buffer, size_t count, int src) override;
    void barrier() override;

   private:
    int rank_;
    int world_size_;
    ncclComm_t comm_;
    cudaStream_t stream_;

    static ncclRedOp_t to_nccl_op(ReduceOp op);
};

}  // namespace distributed
}  // namespace serius

#endif  // SERIUS_USE_NCCL
