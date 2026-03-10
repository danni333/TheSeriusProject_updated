#include "tensor_parallel.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace serius {
namespace distributed {

TensorParallelLinear::TensorParallelLinear(Communicator *comm,
                                           const TensorParallelConfig &config)
    : comm_(comm), config_(config)
{
    if (config_.tp_size != comm_->world_size()) {
        throw std::runtime_error(
            "TensorParallelLinear: tp_size must equal communicator world_size");
    }
}

void TensorParallelLinear::column_parallel_forward(
    const float *X, const float *W_local, float *Y,
    size_t batch_size, size_t in_features, size_t local_out_features)
{
    // Step 1: Local matmul Y_local = X * W_local
    // X: [batch_size, in_features]
    // W_local: [in_features, local_out_features]
    // Y_local: [batch_size, local_out_features]
    size_t local_count = batch_size * local_out_features;
    local_output_buffer_.resize(local_count, 0.0f);

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < local_out_features; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                sum += X[i * in_features + k] * W_local[k * local_out_features + j];
            }
            local_output_buffer_[i * local_out_features + j] = sum;
        }
    }

    // Step 2: AllGather to reconstruct full output
    // Y: [batch_size, out_features] where out_features = local_out_features * tp_size
    // AllGather concatenates along the feature dimension
    //
    // We need to AllGather per-row, but for simplicity we AllGather the flat buffer
    // and then rearrange. Actually, since AllGather concatenates contiguously,
    // we need to do it row by row or reshape.
    //
    // Approach: AllGather the full local buffer, then interleave.
    size_t out_features = local_out_features * static_cast<size_t>(config_.tp_size);
    std::vector<float> gathered(local_count * static_cast<size_t>(config_.tp_size));
    comm_->all_gather(local_output_buffer_.data(), local_count, gathered.data());

    // Rearrange: gathered is [rank0_all_rows, rank1_all_rows, ...]
    // We need Y[i, :] = [rank0_row_i, rank1_row_i, ...]
    for (size_t i = 0; i < batch_size; ++i) {
        for (int r = 0; r < config_.tp_size; ++r) {
            size_t src_offset = static_cast<size_t>(r) * local_count + i * local_out_features;
            size_t dst_offset = i * out_features + static_cast<size_t>(r) * local_out_features;
            std::memcpy(Y + dst_offset,
                         gathered.data() + src_offset,
                         local_out_features * sizeof(float));
        }
    }
}

void TensorParallelLinear::row_parallel_forward(
    const float *X_local, const float *W_local, float *Y,
    size_t batch_size, size_t local_in_features, size_t out_features)
{
    // Step 1: Local matmul Y_partial = X_local * W_local
    // X_local: [batch_size, local_in_features]
    // W_local: [local_in_features, out_features]
    // Y_partial: [batch_size, out_features]
    size_t out_count = batch_size * out_features;

    // Write directly to Y (will be overwritten by AllReduce result)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < local_in_features; ++k) {
                sum += X_local[i * local_in_features + k] * W_local[k * out_features + j];
            }
            Y[i * out_features + j] = sum;
        }
    }

    // Step 2: AllReduce to sum partial results across all ranks
    comm_->all_reduce(Y, out_count, ReduceOp::Sum);
}

void TensorParallelLinear::scatter_input(const float *X, float *X_local,
                                         size_t batch_size, size_t in_features)
{
    size_t local_in = in_features / static_cast<size_t>(config_.tp_size);
    size_t offset = static_cast<size_t>(comm_->rank()) * local_in;

    for (size_t i = 0; i < batch_size; ++i) {
        std::memcpy(X_local + i * local_in,
                     X + i * in_features + offset,
                     local_in * sizeof(float));
    }
}

}  // namespace distributed
}  // namespace serius
