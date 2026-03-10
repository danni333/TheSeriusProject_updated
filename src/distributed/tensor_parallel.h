#pragma once

#include <cstddef>
#include <vector>

#include "comm.h"

namespace serius {
namespace distributed {

/// Tensor Parallelism: splits individual layers horizontally across ranks.
///
/// For a linear layer Y = X * W where W is [d_model, d_ff]:
///   - Each rank holds W_part of shape [d_model, d_ff / world_size]
///   - Each rank computes Y_part = X * W_part (local matmul)
///   - AllGather to reconstruct Y = [Y_0, Y_1, ..., Y_{n-1}]
///
/// For column-parallel (e.g., FFN first linear):
///   W is split along columns → each rank gets d_ff/N columns
///   AllGather on the output
///
/// For row-parallel (e.g., FFN second linear):
///   W is split along rows → each rank gets d_model/N rows of input
///   AllReduce on the output (since each rank computes a partial sum)

/// Configuration for tensor parallelism
struct TensorParallelConfig
{
    int tp_size = 1;  // Tensor parallel degree (number of ranks for TP)
};

/// Helper class for tensor-parallel linear layer operations.
class TensorParallelLinear
{
   public:
    /// @param comm       Communicator for the tensor parallel group
    /// @param config     TP configuration
    explicit TensorParallelLinear(Communicator *comm,
                                 const TensorParallelConfig &config = TensorParallelConfig{});

    /// Column-parallel forward: Y = X * W where W is split along columns.
    /// Each rank holds W_local of shape [in_features, out_features / tp_size].
    /// After local matmul, AllGather reconstructs the full output.
    ///
    /// @param X          Input tensor [batch_size, in_features]
    /// @param W_local    Local weight shard [in_features, local_out_features]
    /// @param Y          Output tensor [batch_size, out_features] (full, after AllGather)
    /// @param batch_size Number of rows in X
    /// @param in_features Number of input features
    /// @param local_out_features Number of output features on this rank (out_features / tp_size)
    void column_parallel_forward(const float *X, const float *W_local, float *Y,
                                 size_t batch_size, size_t in_features,
                                 size_t local_out_features);

    /// Row-parallel forward: Y = X * W where W is split along rows.
    /// Each rank holds W_local of shape [in_features / tp_size, out_features].
    /// Input X must be pre-scattered (each rank gets its portion of features).
    /// After local matmul, AllReduce sums the partial results.
    ///
    /// @param X_local    Local input shard [batch_size, local_in_features]
    /// @param W_local    Local weight shard [local_in_features, out_features]
    /// @param Y          Output tensor [batch_size, out_features] (after AllReduce)
    /// @param batch_size Number of rows in X
    /// @param local_in_features Number of input features on this rank
    /// @param out_features Number of output features
    void row_parallel_forward(const float *X_local, const float *W_local, float *Y,
                              size_t batch_size, size_t local_in_features,
                              size_t out_features);

    /// Scatter input features for row-parallel: splits X along the feature dimension.
    /// X [batch_size, in_features] → X_local [batch_size, in_features / tp_size]
    ///
    /// @param X           Full input [batch_size, in_features]
    /// @param X_local     Output local shard [batch_size, local_in_features]
    /// @param batch_size  Number of rows
    /// @param in_features Total input features
    void scatter_input(const float *X, float *X_local,
                       size_t batch_size, size_t in_features);

    int tp_rank() const { return comm_->rank(); }
    int tp_size() const { return config_.tp_size; }

   private:
    Communicator *comm_;
    TensorParallelConfig config_;
    std::vector<float> local_output_buffer_;  // Scratch buffer for local matmul result
};

}  // namespace distributed
}  // namespace serius
