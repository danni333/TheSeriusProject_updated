#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
///   auto comm = Communicator::create(CommBackend::NCCL, world_size, rank);
///   comm->all_reduce(grad_buffer, count, ReduceOp::Avg);
class Communicator
{
   public:
    virtual ~Communicator() = default;

    // ---- Identification ----

    /// Global rank of this process (0-based)
    virtual int rank() const = 0;

    /// Total number of processes in the communicator
    virtual int world_size() const = 0;

    /// Backend type
    virtual CommBackend backend() const = 0;

    // ---- Collective Operations ----

    /// AllReduce: every process contributes data, every process receives the reduction.
    /// The result is written back to the same buffer (in-place).
    ///
    /// @param buffer  Data buffer (in/out), same size on all ranks
    /// @param count   Number of float elements
    /// @param op      Reduction operation (Sum, Avg, Min, Max)
    virtual void all_reduce(float *buffer, size_t count, ReduceOp op) = 0;

    /// Broadcast: rank `root` sends data to all other ranks.
    /// After the call, all ranks have the same data in `buffer`.
    ///
    /// @param buffer  Data buffer (in on root, out on others)
    /// @param count   Number of float elements
    /// @param root    Rank of the sender
    virtual void broadcast(float *buffer, size_t count, int root = 0) = 0;

    /// AllGather: every rank contributes `send_count` elements,
    /// every rank receives all contributions concatenated.
    ///
    /// @param send_buf   Local data to contribute [send_count]
    /// @param send_count Number of elements to send from this rank
    /// @param recv_buf   Output buffer [send_count * world_size]
    virtual void all_gather(const float *send_buf, size_t send_count,
                            float *recv_buf) = 0;

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
    /// For NCCL/MPI: these are typically set by the runtime environment.
    static std::unique_ptr<Communicator> create(CommBackend backend,
                                                int world_size = 1,
                                                int rank = 0);
};

}  // namespace distributed
}  // namespace serius
