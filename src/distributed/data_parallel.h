#pragma once

#include <cstddef>
#include <vector>

#include "comm.h"

namespace serius {
namespace distributed {

/// Data Parallelism: each rank holds a full copy of the model and processes
/// different data batches. After backward pass, gradients are averaged
/// across all ranks via AllReduce.
///
/// Usage:
///   DataParallel dp(comm);
///   // ... forward + backward on local batch ...
///   dp.sync_gradients(grad_buffers);
///   // ... optimizer step (identical on all ranks) ...
class DataParallel
{
   public:
    /// @param comm  Communicator (owns lifetime externally)
    explicit DataParallel(Communicator *comm);

    /// Synchronize gradients across all ranks by averaging.
    /// Each buffer is AllReduced in-place with ReduceOp::Avg.
    ///
    /// @param grad_buffers  Vector of (pointer, count) pairs for each parameter's gradient
    void sync_gradients(const std::vector<std::pair<float *, size_t>> &grad_buffers);

    /// Synchronize a single gradient buffer.
    void sync_gradient(float *buffer, size_t count);

    /// Broadcast model parameters from rank 0 to all ranks.
    /// Call this once before training to ensure all ranks start with the same weights.
    ///
    /// @param param_buffers  Vector of (pointer, count) pairs for each parameter
    void broadcast_params(const std::vector<std::pair<float *, size_t>> &param_buffers);

    /// Get the effective batch size scaling factor.
    /// The effective batch size = local_batch_size * world_size.
    int world_size() const { return comm_->world_size(); }
    int rank() const { return comm_->rank(); }

   private:
    Communicator *comm_;
};

}  // namespace distributed
}  // namespace serius
