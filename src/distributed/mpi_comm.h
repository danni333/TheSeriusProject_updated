#pragma once

#ifdef SERIUS_USE_MPI

#include "comm.h"

#include <mpi.h>

namespace serius {
namespace distributed {

/// MPI-based communicator for inter-node (multi-machine) communication.
///
/// MPI is used for communication across network-connected nodes,
/// typically in conjunction with NCCL for intra-node GPU communication.
///
/// Typical hybrid setup:
///   - NCCL handles GPU<->GPU within a node (via NVLink/PCIe)
///   - MPI handles node<->node (via InfiniBand/Ethernet)
class MPICommunicator : public Communicator
{
   public:
    /// Initialize MPI communicator.
    /// Calls MPI_Init if not already initialized.
    /// @param argc  Pointer to argc from main (or nullptr)
    /// @param argv  Pointer to argv from main (or nullptr)
    MPICommunicator(int *argc = nullptr, char ***argv = nullptr);
    ~MPICommunicator() override;

    // Non-copyable
    MPICommunicator(const MPICommunicator &) = delete;
    MPICommunicator &operator=(const MPICommunicator &) = delete;

    int rank() const override { return rank_; }
    int world_size() const override { return world_size_; }
    CommBackend backend() const override { return CommBackend::MPI; }

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
    bool owns_mpi_;  // true if we called MPI_Init

    static MPI_Op to_mpi_op(ReduceOp op);
};

}  // namespace distributed
}  // namespace serius

#endif  // SERIUS_USE_MPI
