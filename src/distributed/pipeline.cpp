#include "pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace serius {
namespace distributed {

PipelineExecutor::PipelineExecutor(Communicator *comm, const PipelineConfig &config) : comm_(comm), config_(config)
{
    if (config_.num_stages != comm_->world_size()) {
        throw std::runtime_error("PipelineExecutor: num_stages must equal world_size");
    }
    if (config_.hidden_size == 0) {
        throw std::runtime_error("PipelineExecutor: hidden_size must be > 0");
    }

    // Pre-allocate buffers for each micro-batch
    size_t n_mb = static_cast<size_t>(config_.num_micro_batches);
    forward_buffers_.resize(n_mb, std::vector<float>(config_.hidden_size, 0.0f));
    backward_buffers_.resize(n_mb, std::vector<float>(config_.hidden_size, 0.0f));
}

void PipelineExecutor::execute(ForwardFn forward_fn, BackwardFn backward_fn, const float *input_data,
                               float *output_data)
{
    int my_stage = stage();
    int n_mb = config_.num_micro_batches;
    size_t hs = config_.hidden_size;

    // Temporary buffers for this stage's input and output activations
    std::vector<std::vector<float>> activations(static_cast<size_t>(n_mb), std::vector<float>(hs, 0.0f));
    std::vector<std::vector<float>> outputs(static_cast<size_t>(n_mb), std::vector<float>(hs, 0.0f));

    // =============================================
    // Phase 1: Forward pass (all micro-batches)
    // =============================================
    for (int mb = 0; mb < n_mb; ++mb) {
        auto mb_idx = static_cast<size_t>(mb);

        if (is_first_stage()) {
            // First stage reads from input_data
            std::memcpy(activations[mb_idx].data(), input_data + mb_idx * hs, hs * sizeof(float));
        } else {
            // Receive activations from previous stage
            comm_->recv(activations[mb_idx].data(), hs, my_stage - 1);
        }

        // Run forward computation for this micro-batch
        forward_fn(activations[mb_idx].data(), outputs[mb_idx].data(), hs);

        if (is_last_stage()) {
            // Last stage writes to output_data
            if (output_data != nullptr) {
                std::memcpy(output_data + mb_idx * hs, outputs[mb_idx].data(), hs * sizeof(float));
            }
        } else {
            // Send activations to next stage
            comm_->send(outputs[mb_idx].data(), hs, my_stage + 1);
        }
    }

    // =============================================
    // Phase 2: Backward pass (all micro-batches, reverse order)
    // =============================================
    for (int mb = n_mb - 1; mb >= 0; --mb) {
        // auto mb_idx = static_cast<size_t>(mb);

        std::vector<float> grad_output(hs, 0.0f);
        std::vector<float> grad_input(hs, 0.0f);

        if (is_last_stage()) {
            // Last stage: gradient comes from the loss (stored in backward_buffers)
            // For now, assume the caller has set up the gradient externally
            // or it's computed from the output. Use a default gradient scale of 1.0 if unset
            // (the actual loss gradient would be computed by the caller).
            std::fill(grad_output.begin(), grad_output.end(), 1.0f);
        } else {
            // Receive gradient from next stage
            comm_->recv(grad_output.data(), hs, my_stage + 1);
        }

        // Run backward computation
        backward_fn(grad_output.data(), grad_input.data(), hs);

        if (!is_first_stage()) {
            // Send gradient to previous stage
            comm_->send(grad_input.data(), hs, my_stage - 1);
        }
    }
}

void PipelineExecutor::execute_forward(ForwardFn forward_fn, const float *input_data, float *output_data)
{
    const int my_stage = stage();
    const int num_micro_batches = config_.num_micro_batches;
    const size_t hidden_size = config_.hidden_size;

    std::vector<std::vector<float>> activations(static_cast<size_t>(num_micro_batches), std::vector<float>(hidden_size, 0.0f));
    std::vector<std::vector<float>> outputs(static_cast<size_t>(num_micro_batches), std::vector<float>(hidden_size, 0.0f));

    for (int mb = 0; mb < num_micro_batches; ++mb) {
        const auto mb_idx = static_cast<size_t>(mb);

        if (is_first_stage()) {
            if (input_data != nullptr) {
                std::memcpy(activations[mb_idx].data(), input_data + (mb_idx * hidden_size), hidden_size * sizeof(float));
            }
        } else {
            comm_->recv(activations[mb_idx].data(), hidden_size, my_stage - 1);
        }

        forward_fn(activations[mb_idx].data(), outputs[mb_idx].data(), hidden_size);

        if (is_last_stage()) {
            if (output_data != nullptr) {
                std::memcpy(output_data + (mb_idx * hidden_size), outputs[mb_idx].data(), hidden_size * sizeof(float));
            }
        } else {
            comm_->send(outputs[mb_idx].data(), hidden_size, my_stage + 1);
        }
    }
}

float PipelineExecutor::bubble_ratio() const
{
    if (config_.num_micro_batches <= 0) {
        return 1.0f;
    }
    return static_cast<float>(config_.num_stages - 1) / static_cast<float>(config_.num_micro_batches);
}

}  // namespace distributed
}  // namespace serius
