#pragma once

#include <stdexcept>

#include "../hal/device.h"
#include "../tensor/tensor.h"

namespace serius {
namespace core {
namespace optim {

class AutomaticMixedPrecision
{
   public:
    // Scaling factor to prevent underflow in FP16 during backward pass
    static constexpr float DEFAULT_LOSS_SCALE = 65536.0f;

    /// Helper for mixed precision forward and backward passes
    /// Forward pass uses FP16 for speed, backward uses FP32 for precision.
    ///
    /// @param forward_fn Lambda that takes FP16 input and returns FP16 output
    /// @param backward_fn Lambda that computes gradients (output_fp16, grad_output_fp32 -> grad_input_fp32)
    /// @param compute_loss Lambda that computes scalar loss from output
    /// @param compute_grad Lambda that computes the initial gradient of the loss
    template <typename FwdFn, typename BwdFn, typename LossFn, typename GradFn>
    static void mixed_precision_forward_backward(FwdFn forward_fn, BwdFn backward_fn, LossFn compute_loss,
                                                 GradFn compute_grad, const Tensor &input, Tensor &output, Tensor &loss,
                                                 Tensor &grad_input)
    {
        // 1. Forward in FP16
        Tensor input_fp16 = input.cast(hal::DType::Float16);
        Tensor output_fp16 = forward_fn(input_fp16);
        output = output_fp16.cast(hal::DType::Float32);  // Keep FP32 handle for user if needed

        // 2. Compute Loss and apply scale
        Tensor base_loss = compute_loss(output_fp16);
        if (base_loss.size() != 1) {
            throw std::runtime_error("Loss must be a scalar tensor");
        }

        // Convert to FP32 first so the scaling doesn't overflow FP16 max (65504)
        Tensor base_loss_fp32 = base_loss.cast(hal::DType::Float32);
        loss = base_loss_fp32 * DEFAULT_LOSS_SCALE;

        // 3. Compute initial gradient in FP32 (unscaled so backward_fn doesn't overflow)
        Tensor loss_unscaled = loss / DEFAULT_LOSS_SCALE;  // Both are now FP32
        Tensor grad_output = compute_grad(loss_unscaled);

        // 4. Backward in FP32
        backward_fn(output_fp16, grad_output, grad_input);
    }

    /// Converts all FP32 parameters in an iterable collection to FP16
    /// Useful for preparing a model for inference-only execution
    template <typename Iterable>
    static void convert_model_to_fp16(Iterable &parameters)
    {
        for (auto &param : parameters) {
            if (param.dtype() == hal::DType::Float32) {
                param = param.cast(hal::DType::Float16);
            }
        }
    }
};

}  // namespace optim
}  // namespace core
}  // namespace serius
