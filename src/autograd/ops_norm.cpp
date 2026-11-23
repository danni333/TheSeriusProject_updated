#define _USE_MATH_DEFINES
#include "ops.h"

// Autograd ops — split from ops.cpp. Declarations live in ops.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "../blas/blas.h"
#include "../attention/flash_attn.h"
#ifdef SERIUS_USE_CUDA
#include "../hal/fused_kernels.h"
#endif
namespace serius {
namespace core {
namespace autograd {
namespace ops {
Node *contiguous(Node *input_node)
{
    if (input_node->data.is_contiguous()) {
        return input_node;
    }

    Node *res = new Node(input_node->data.copy(), input_node->requires_grad);
    res->name = "contiguous(" + input_node->name + ")";
    res->children.push_back(input_node);
    res->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([](const std::vector<Node *> &a) { return contiguous(a[0]); }, in);
    };

    res->backward_fn = [input_node, res]() {
        if (input_node->requires_grad) {
            input_node->grad += res->grad;
        }
    };

    return res;
}

Node *layernorm(Node *input, Node *gamma, Node *beta, float eps)
{
    const Tensor &in_data = input->data;
    Tensor output_data(in_data.shape(), in_data.dtype());
    bool handled = false;

#ifdef SERIUS_USE_CUDA
    if (in_data.is_gpu()) {
        size_t d_model = in_data.shape().back();
        size_t batch_seq = in_data.size() / d_model;

        in_data.syncToDevice();
        gamma->data.syncToDevice();
        beta->data.syncToDevice();
        output_data.syncToDevice();

        // Copy input to output since the kernel is in-place on x
        hal::get_backend(hal::DeviceType::CUDA)
            ->memcpy_d2d(output_data.data_gpu(), in_data.data_gpu(), in_data.size() * in_data.element_size());

        hal::fused_layernorm_residual(output_data.data_gpu(), nullptr, gamma->data.data_gpu(), beta->data.data_gpu(),
                                      d_model, batch_seq, eps, in_data.dtype());
        output_data.syncToHost();
        handled = true;
    }
#endif

    if (!handled) {
        // CPU Implementation (simplified for brevity, assuming float32)
        size_t d_model = in_data.shape().back();
        size_t batch_seq = in_data.size() / d_model;
        for (size_t batch_idx = 0; batch_idx < batch_seq; ++batch_idx) {
            float sum_val = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                sum_val += in_data.data()[batch_idx * d_model + i];
            }
            float mean_val = sum_val / static_cast<float>(d_model);
            float var_sum = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                float diff_val = in_data.data()[batch_idx * d_model + i] - mean_val;
                var_sum += diff_val * diff_val;
            }
            float inv_std_val = 1.0F / std::sqrt(var_sum / static_cast<float>(d_model) + eps);
            for (size_t i = 0; i < d_model; ++i) {
                float x_hat_val = (in_data.data()[(batch_idx * d_model) + i] - mean_val) * inv_std_val;
                output_data.data()[(batch_idx * d_model) + i] = (gamma->data[i] * x_hat_val) + beta->data[i];
            }
        }
    }

    Node *out = new Node(std::move(output_data));
    out->children = {input, gamma, beta};
    out->name = "layernorm";
    out->forward_fn = [eps](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward(
            [eps](const std::vector<Node *> &a) { return layernorm(a[0], a[1], a[2], eps); }, in);
    };

    out->backward_fn = [input, gamma, beta, out, eps]() {
        if (!input->requires_grad && !gamma->requires_grad && !beta->requires_grad) return;
        // Simplified LayerNorm backward (assuming CPU float32 for grad calculation)
        const Tensor &x_data = input->data;
        const Tensor &grad_out = out->grad;
        size_t d_model = x_data.shape().back();
        size_t batch_seq = x_data.size() / d_model;

        for (size_t batch_idx = 0; batch_idx < batch_seq; ++batch_idx) {
            float sum_x_val = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                sum_x_val += x_data.data()[batch_idx * d_model + i];
            }
            float mean_val = sum_x_val / static_cast<float>(d_model);
            float var_sum = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                float diff_val = x_data.data()[batch_idx * d_model + i] - mean_val;
                var_sum += diff_val * diff_val;
            }
            float inv_std_val = 1.0F / std::sqrt(var_sum / static_cast<float>(d_model) + eps);

            float sum_dy_val = 0.0F;
            float sum_dy_xhat_val = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                float xhat_val = (x_data.data()[(batch_idx * d_model) + i] - mean_val) * inv_std_val;
                float dyi_val = grad_out.data()[(batch_idx * d_model) + i];
                if (gamma->requires_grad) {
                    gamma->grad[i] += dyi_val * xhat_val;
                }
                if (beta->requires_grad) {
                    beta->grad[i] += dyi_val;
                }
                sum_dy_val += dyi_val;
                sum_dy_xhat_val += dyi_val * xhat_val;
            }

            if (input->requires_grad) {
                for (size_t i = 0; i < d_model; ++i) {
                    float xhat_val = (x_data.data()[(batch_idx * d_model) + i] - mean_val) * inv_std_val;
                    float dyi_val = grad_out.data()[(batch_idx * d_model) + i];
                    float dx_val =
                        (gamma->data[i] * inv_std_val / static_cast<float>(d_model)) *
                        ((static_cast<float>(d_model) * dyi_val) - sum_dy_val - (xhat_val * sum_dy_xhat_val));
                    input->grad.data()[(batch_idx * d_model) + i] += dx_val;
                }
            }
        }
    };
    return out;
}

Node *rmsnorm(Node *input, Node *gamma, float eps)
{
    const Tensor &in_data = input->data;
    Tensor output_data(in_data.shape(), in_data.dtype());
    size_t d_model = in_data.shape().back();
    size_t batch_seq = in_data.size() / d_model;
    for (size_t batch_idx = 0; batch_idx < batch_seq; ++batch_idx) {
        float sum_sq = 0.0F;
        for (size_t i = 0; i < d_model; ++i) {
            float v = in_data.data()[batch_idx * d_model + i];
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(d_model) + eps);
        for (size_t i = 0; i < d_model; ++i)
            output_data.data()[batch_idx * d_model + i] =
                (in_data.data()[batch_idx * d_model + i] / rms) * gamma->data[i];
    }
    Node *out = new Node(std::move(output_data));
    out->children = {input, gamma};
    out->name = "rmsnorm";
    out->forward_fn = [eps](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward(
            [eps](const std::vector<Node *> &a) { return rmsnorm(a[0], a[1], eps); }, in);
    };
    out->backward_fn = [input, gamma, out, eps]() {
        if (!input->requires_grad && !gamma->requires_grad) return;
        const Tensor &x = input->data;
        const Tensor &go = out->grad;
        size_t d_model = x.shape().back();
        size_t batch_seq = x.size() / d_model;
        for (size_t b = 0; b < batch_seq; ++b) {
            float sum_sq = 0.0F;
            for (size_t i = 0; i < d_model; ++i) sum_sq += x.data()[b * d_model + i] * x.data()[b * d_model + i];
            float rms = std::sqrt(sum_sq / static_cast<float>(d_model) + eps);
            float sum_gy = 0.0F;
            for (size_t j = 0; j < d_model; ++j) sum_gy += go.data()[b * d_model + j] * gamma->data[j] * x.data()[b * d_model + j];
            for (size_t i = 0; i < d_model; ++i) {
                float x_hat = x.data()[b * d_model + i] / rms;
                if (gamma->requires_grad) gamma->grad[i] += go.data()[b * d_model + i] * x_hat;
                if (input->requires_grad)
                    input->grad.data()[b * d_model + i] += (gamma->data[i] * go.data()[b * d_model + i] / rms) -
                        (x.data()[b * d_model + i] * sum_gy / (static_cast<float>(d_model) * rms * rms * rms));
            }
        }
    };
    return out;
}

Node *fused_add_layernorm(Node *input, Node *residual, Node *gamma, Node *beta, float eps)
{
#ifdef SERIUS_USE_CUDA
    if (input->data.is_gpu()) {
        const Tensor &in_data = input->data;
        Tensor output_data(in_data.shape(), in_data.dtype());
        size_t d_model = in_data.shape().back();
        size_t batch_seq = in_data.size() / d_model;

        in_data.syncToDevice();
        residual->data.syncToDevice();
        gamma->data.syncToDevice();
        beta->data.syncToDevice();
        output_data.syncToDevice();

        // Copy input to output since the kernel adds residual to x in-place
        hal::get_backend(hal::DeviceType::CUDA)
            ->memcpy_d2d(output_data.data_gpu(), in_data.data_gpu(), in_data.size() * in_data.element_size());

        hal::fused_layernorm_residual(output_data.data_gpu(), residual->data.data_gpu(), gamma->data.data_gpu(),
                                      beta->data.data_gpu(), d_model, batch_seq, eps, in_data.dtype());
        output_data.syncToHost();

        Node *out = new Node(std::move(output_data));
        out->children = {input, residual, gamma, beta};
        out->name = "fused_add_layernorm";
        out->forward_fn = [eps](const std::vector<const Tensor *> &in) -> Tensor {
            return detail::replay_forward(
                [eps](const std::vector<Node *> &a) {
                    return fused_add_layernorm(a[0], a[1], a[2], a[3], eps);
                },
                in);
        };

        // Backward: dL/d(x+res) is same as dL/dx for LayerNorm.
        // Then dL/dx += dL/d(x+res), dL/dres += dL/d(x+res).
        out->backward_fn = [input, residual, gamma, beta, out, eps]() {
            if (!input->requires_grad && !residual->requires_grad && !gamma->requires_grad && !beta->requires_grad)
                return;

            input->data.syncToHost();
            residual->data.syncToHost();
            out->grad.syncToHost();

            const Tensor &x = input->data;
            const Tensor &res = residual->data;
            const Tensor &dy = out->grad;
            size_t d_model = x.shape().back();
            size_t batch_seq = x.size() / d_model;

            for (size_t b = 0; b < batch_seq; ++b) {
                float sum_xr = 0.0f;
                for (size_t i = 0; i < d_model; ++i) {
                    sum_xr += x.data()[b * d_model + i] + res.data()[b * d_model + i];
                }
                float mean = sum_xr / d_model;
                float var_sum = 0.0f;
                for (size_t i = 0; i < d_model; ++i) {
                    float diff = (x.data()[b * d_model + i] + res.data()[b * d_model + i]) - mean;
                    var_sum += diff * diff;
                }
                float inv_std = 1.0f / std::sqrt(var_sum / d_model + eps);

                float sum_dy = 0.0f;
                float sum_dy_xhat = 0.0f;
                for (size_t i = 0; i < d_model; ++i) {
                    float xhat = ((x.data()[b * d_model + i] + res.data()[b * d_model + i]) - mean) * inv_std;
                    float dyi = dy.data()[b * d_model + i];
                    if (gamma->requires_grad) gamma->grad[i] += dyi * xhat;
                    if (beta->requires_grad) beta->grad[i] += dyi;
                    sum_dy += dyi;
                    sum_dy_xhat += dyi * xhat;
                }

                if (input->requires_grad || residual->requires_grad) {
                    for (size_t i = 0; i < d_model; ++i) {
                        float xhat = ((x.data()[b * d_model + i] + res.data()[b * d_model + i]) - mean) * inv_std;
                        float dyi = dy.data()[b * d_model + i];
                        float dx = (gamma->data[i] * inv_std / d_model) * (d_model * dyi - sum_dy - xhat * sum_dy_xhat);
                        if (input->requires_grad) input->grad.data()[b * d_model + i] += dx;
                        if (residual->requires_grad) residual->grad.data()[b * d_model + i] += dx;
                    }
                }
            }
        };
        return out;
    }
#endif
    // Fallback if not CUDA: standard add + layernorm
    return layernorm(add(input, residual), gamma, beta, eps);
}

Node *fused_masked_softmax(Node *input, const Tensor &mask, float scale)
{
    Tensor res(input->data.shape(), input->data.dtype());
    bool handled = false;

#ifdef SERIUS_USE_CUDA
    if (input->data.is_gpu()) {
        input->data.syncToDevice();
        mask.syncToDevice();
        res.syncToDevice();

        size_t batch_heads = input->data.shape()[0] * input->data.shape()[1];
        size_t seq_q = input->data.shape()[2];
        size_t seq_k = input->data.shape()[3];

        // Copy input to scores since the kernel is in-place
        hal::get_backend(hal::DeviceType::CUDA)
            ->memcpy_d2d(res.data_gpu(), input->data.data_gpu(), input->data.size() * input->data.element_size());

        hal::fused_masked_softmax(res.data_gpu(), mask.data_gpu(), batch_heads, seq_q, seq_k, input->data.dtype(),
                                  scale);
        res.mark_dirty(Tensor::DataLocation::GPU);
        handled = true;
    }
#endif

    if (!handled) {
        // Fallback: apply scale, apply mask, then softmax
        const float NEG_INF = -1e20F;
        Node *scaled_input = mul_scalar(input, scale);
        Node *masked_input = masked_fill(scaled_input, mask, NEG_INF);
        Node *softmax_output = softmax(masked_input);

        // The result of the fallback path is a new Node, so we need to return it directly.
        // The backward_fn for this path is handled by the individual ops (mul_scalar, masked_fill, softmax).
        return softmax_output;
    }

    Node *out = new Node(std::move(res));
    out->children = {input};
    out->name = "fused_masked_softmax";
    if (!detail::in_replay()) {
        out->forward_fn = [mask, scale](const std::vector<const Tensor *> &in) -> Tensor {
            return detail::replay_forward(
                [&mask, scale](const std::vector<Node *> &a) {
                    return fused_masked_softmax(a[0], mask, scale);
                },
                in);
        };
    }
    out->backward_fn = [input, out, scale]() {
        if (!input->requires_grad) {
            return;
        }

        out->data.syncToHost();
        out->grad.syncToHost();

        const float *softmax_data = out->data.data();
        const float *grad_softmax = out->grad.data();
        float *grad_input = input->grad.data();

        size_t total_size = out->data.size();
        size_t k_dim = out->data.shape().back();
        size_t n_dim = total_size / k_dim;

        for (size_t i = 0; i < n_dim; ++i) {
            float sum_grad_val = 0.0F;
            for (size_t j = 0; j < k_dim; ++j) {
                sum_grad_val += grad_softmax[i * k_dim + j] * softmax_data[i * k_dim + j];
            }
            for (size_t j = 0; j < k_dim; ++j) {
                float softmax_val = softmax_data[i * k_dim + j];
                grad_input[i * k_dim + j] += scale * softmax_val * (grad_softmax[i * k_dim + j] - sum_grad_val);
            }
        }
    };
    return out;
}

Node *fused_bias_gelu(Node *input, Node *bias)
{
    const Tensor &in_data = input->data;
    const Tensor &bias_data = bias->data;

    // Check shapes: bias should be 1D, size should match last dim of input
    size_t last_dim = in_data.shape().back();
    if (bias_data.size() != last_dim) {
        throw std::runtime_error("fused_bias_gelu: Bias size must match the last dimension of the input.");
    }

#ifdef SERIUS_USE_CUDA
    if (in_data.is_gpu()) {
        Tensor output_data(in_data.shape(), in_data.dtype());
        in_data.syncToDevice();
        bias_data.syncToDevice();
        output_data.syncToDevice();

        size_t total_elements = in_data.size();
        size_t bias_size = bias_data.size();

        // Copy input to output since the kernel runs in-place
        hal::get_backend(hal::DeviceType::CUDA)
            ->memcpy_d2d(output_data.data_gpu(), in_data.data_gpu(), total_elements * in_data.element_size());

        hal::fused_bias_gelu(output_data.data_gpu(), bias_data.data_gpu(), total_elements, bias_size, in_data.dtype());
        output_data.mark_dirty(Tensor::DataLocation::GPU);

        Node *out = new Node(std::move(output_data));
        out->children = {input, bias};
        out->name = "fused_bias_gelu";
        out->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
            return detail::replay_forward([](const std::vector<Node *> &a) { return fused_bias_gelu(a[0], a[1]); },
                                          in);
        };

        // Backward: dL/dx = dL/dy * GELU'(x + bias)
        // dL/dbias = sum(dL/dx) over appropriate dimensions
        // Since we don't have a fused backward kernel yet, we can fallback to standard
        // autograd chains for backward if requires_grad is true.
        out->backward_fn = [input, bias, out]() {
            if (!input->requires_grad && !bias->requires_grad) return;
            // Since we applied it in place on forward, backward needs the standard ops
            // Alternatively, we define a small backward approximation or just execute a full backward.

            // Mathematical approximation of GELU derivative
            const float PI_CONST = static_cast<float>(M_PI);
            const float SQRT_2_PI = std::sqrt(2.0F / PI_CONST);
            const float GELU_COEFF = 0.044715F;
            const float HALF = 0.5F;
            const float ONE = 1.0F;
            const float THREE = 3.0F;

            input->data.syncToHost();
            bias->data.syncToHost();
            out->grad.syncToHost();

            size_t input_size = input->data.size();
            size_t bias_size = bias->data.size();

            for (size_t i = 0; i < input_size; ++i) {
                float x_val = input->data[i] + bias->data[i % bias_size];
                float x_squared = x_val * x_val;
                float x_cubed = x_squared * x_val;
                float inner = SQRT_2_PI * (x_val + (GELU_COEFF * x_cubed));
                float tanh_val = std::tanh(inner);
                float sech2_val = ONE - (tanh_val * tanh_val);

                float deriv = (HALF * (ONE + tanh_val)) +
                              (HALF * x_val * sech2_val * SQRT_2_PI * (ONE + (THREE * GELU_COEFF * x_squared)));

                float grad_val = out->grad[i] * deriv;

                if (input->requires_grad) {
                    input->grad[i] += grad_val;
                }
                if (bias->requires_grad) {
                    bias->grad[i % bias_size] += grad_val;
                }
            }
        };

        return out;
    }
#endif

    // Fallback for CPU / unsupported cases: perform standard add and gelu ops.
    return gelu(add(input, bias));
}

Node *flash_attention(Node *query, Node *key, Node *value, float scale, bool causal)
{
#ifdef SERIUS_USE_CUDA
    serius::model::attention::FlashAttentionConfig config;
    config.softmax_scale = scale;
    config.is_causal = causal;
    return serius::model::attention::flash_attention(query, key, value, config);
#else
    (void)query;
    (void)key;
    (void)value;
    (void)scale;
    (void)causal;
    throw std::runtime_error("flash_attention requires SERIUS_USE_CUDA");
#endif
}

}  // namespace ops
}  // namespace autograd
}  // namespace core
}  // namespace serius

