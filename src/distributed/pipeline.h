#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "comm.h"

namespace serius {
namespace distributed {

/// Pipeline Parallelism: the model is split vertically by layers across ranks.
/// Each rank owns a contiguous set of transformer blocks. Input data flows
/// through the pipeline, with micro-batching to reduce bubble time.
///
/// Schedule (GPipe-style, 2 stages):
///   GPU 0 (layers 0-5):   [F0] [F1] [F2] [F3]  -----  [B3] [B2] [B1] [B0]
///   GPU 1 (layers 6-11):  ---- [F0] [F1] [F2] [F3] [B3] [B2] [B1] [B0] ----
///
/// The bubble ratio decreases as num_microbatches increases:
///   bubble = (num_stages - 1) / num_microbatches

/// Configuration for pipeline parallelism
struct PipelineConfig
{
    int num_stages = 2;          // Number of pipeline stages (= number of ranks)
    int num_micro_batches = 4;   // Number of micro-batches per global batch
    size_t hidden_size = 0;      // Size of intermediate activations per micro-batch
};

/// Callback types for forward and backward passes on a pipeline stage.
/// The pipeline executor calls these to run the actual computation.
///
/// ForwardFn: takes input activation buffer, writes output activation buffer
/// BackwardFn: takes output gradient buffer, writes input gradient buffer
using ForwardFn = std::function<void(const float *input, float *output, size_t count)>;
using BackwardFn = std::function<void(const float *grad_output, float *grad_input, size_t count)>;

/// Pipeline executor that coordinates micro-batch scheduling across stages.
class PipelineExecutor
{
   public:
    /// @param comm    Communicator for the pipeline group
    /// @param config  Pipeline configuration
    PipelineExecutor(Communicator *comm, const PipelineConfig &config);

    /// Execute one full pipeline step (forward + backward for all micro-batches).
    ///
    /// This implements the GPipe schedule:
    ///   1. Forward all micro-batches through all stages
    ///   2. Backward all micro-batches through all stages (in reverse)
    ///
    /// @param forward_fn   Function to compute forward pass for this stage
    /// @param backward_fn  Function to compute backward pass for this stage
    /// @param input_data   Input data for the first stage (ignored on other stages)
    ///                     Shape: [num_micro_batches * micro_batch_size]
    /// @param output_data  Output from the last stage (ignored on other stages)
    void execute(ForwardFn forward_fn, BackwardFn backward_fn,
                 const float *input_data, float *output_data);

    /// Execute only the forward pass (for inference).
    void execute_forward(ForwardFn forward_fn, const float *input_data, float *output_data);

    /// Get the pipeline stage index for this rank
    int stage() const { return comm_->rank(); }

    /// Whether this rank is the first stage in the pipeline
    bool is_first_stage() const { return stage() == 0; }

    /// Whether this rank is the last stage in the pipeline
    bool is_last_stage() const { return stage() == config_.num_stages - 1; }

    /// Calculate the bubble ratio for this configuration
    float bubble_ratio() const;

   private:
    Communicator *comm_;
    PipelineConfig config_;

    // Buffers for inter-stage communication
    std::vector<std::vector<float>> forward_buffers_;   // [num_micro_batches][hidden_size]
    std::vector<std::vector<float>> backward_buffers_;  // [num_micro_batches][hidden_size]
};

}  // namespace distributed
}  // namespace serius
