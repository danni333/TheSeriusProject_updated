#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef SERIUS_USE_NCCL
#include <nccl.h>
#endif

#include "../tensor/tensor.h"

namespace serius {
namespace distributed {

/// Reduction operations for collective communication
enum class ReduceOp : uint8_t
{
    Sum,
    Avg,
    Min,
    Max,
};

/// Backend type for the communicator
enum class CommBackend : uint8_t
{
    Local,  // Single-process (for testing and single-GPU)
    NCCL,   // NVIDIA Collective Communications Library (intra-node GPU)
    MPI,    // Message Passing Interface (inter-node CPU/GPU)
};

/// Abstract interface for distributed communication.
///
/// All collective operations are synchronous by default.
/// Implementations may use asynchronous primitives internally
/// and synchronize before returning.
///
/// Typical usage:
///   auto comm = Communicator::create(CommBackend::MPI);
///   comm->all_reduce(grad_buffer, count, ReduceOp::Avg);
class Communicator
{
   public:
    virtual ~Communicator() = default;

    // ---- Identification ----

    /// Global rank of this process (0-based)
    virtual int rank() const = 0;

    /// Rank of this process among processes on the same node (0-based)
    virtual int local_rank() const = 0;

    /// Total number of processes in the communicator
    virtual int world_size() const = 0;

    /// Backend type
    virtual CommBackend backend() const = 0;

    // ---- Collective Operations ----

    /// AllReduce: every process contributes data, every process receives the reduction.
    /// The result is written back to the same buffer (in-place).
    virtual void all_reduce(float *buffer, size_t count, ReduceOp op) = 0;

    /// Tensor-based AllReduce
    virtual void all_reduce(core::Tensor &tensor, ReduceOp op) = 0;

    /// Broadcast: rank `root` sends data to all other ranks.
    virtual void broadcast(float *buffer, size_t count, int root = 0) = 0;

    /// Tensor-based Broadcast
    virtual void broadcast(core::Tensor &tensor, int root = 0) = 0;

    /// AllGather: every rank contributes `send_count` elements,
    virtual void all_gather(const float *send_buf, size_t send_count,
                            float *recv_buf) = 0;

    /// Tensor-based AllGather
    /// recv_tensor must have world_size times the size of send_tensor
    virtual void all_gather(const core::Tensor &send_tensor,
                             core::Tensor &recv_tensor) = 0;

    /// ReduceScatter: reduce across ranks, then scatter the result.
    /// Each rank receives a different chunk of the reduced data.
    ///
    /// @param send_buf     Data to contribute [count * world_size]
    /// @param recv_buf     Output buffer [count]
    /// @param recv_count   Number of elements this rank receives
    /// @param op           Reduction operation
    virtual void reduce_scatter(const float *send_buf, float *recv_buf,
                                size_t recv_count, ReduceOp op) = 0;

    /// Send data to a specific rank (point-to-point)
    /// @param buffer  Data to send
    /// @param count   Number of float elements
    /// @param dst     Destination rank
    virtual void send(const float *buffer, size_t count, int dst) = 0;

    /// Receive data from a specific rank (point-to-point)
    /// @param buffer  Buffer to receive into
    /// @param count   Number of float elements
    /// @param src     Source rank
    virtual void recv(float *buffer, size_t count, int src) = 0;

    /// Barrier: block until all ranks reach this point
    virtual void barrier() = 0;

    // ---- Factory ----

    /// Create a communicator with the specified backend.
    /// For Local backend: world_size and rank are user-specified (for simulation).
    /// For MPI, rank and world size are determined by MPI_COMM_WORLD.
    static std::unique_ptr<Communicator> create(CommBackend backend,
                                                int world_size = 1,
                                                int rank = 0);

#ifdef SERIUS_USE_NCCL
    /// Create an NCCL communicator using a unique ID shared by all ranks.
    static std::unique_ptr<Communicator> create_nccl(int world_size, int rank,
                                                     ncclUniqueId id,
                                                     int local_rank = -1);

    /// Create an NCCL communicator by generating and distributing its unique ID.
    /// Requires SERIUS_USE_MPI, initialized MPI, and all ranks to participate.
    static std::unique_ptr<Communicator> create_nccl(int world_size, int rank);
#endif
};

}  // namespace distributed
}  // namespace serius
