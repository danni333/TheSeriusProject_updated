#define _USE_MATH_DEFINES
#include "ops.h"

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

Node *matmul(Node *input_a, Node *input_b)
{
    // Forward pass
    Tensor res({0});
    bool complex_reshape = (input_a->data.ndim() > 3 || input_b->data.ndim() > 3);

    if (complex_reshape) {
        // Flatten leading dimensions for batched matmul
        std::vector<size_t> shape_a = input_a->data.shape();
        std::vector<size_t> shape_b = input_b->data.shape();
        size_t last_a = shape_a.back();
        size_t next_last_a = shape_a[shape_a.size() - 2];
        size_t batch_a = input_a->data.size() / (last_a * next_last_a);

        size_t last_b = shape_b.back();
        size_t next_last_b = shape_b[shape_b.size() - 2];
        size_t batch_b = input_b->data.size() / (last_b * next_last_b);

        Tensor a_cont = input_a->data.contiguous();
        Tensor b_cont = input_b->data.contiguous();

        Tensor a_flat = (input_a->data.ndim() > 2) 
                        ? a_cont.reshape({batch_a, next_last_a, last_a})
                        : a_cont;
        Tensor b_flat = (input_b->data.ndim() > 2)
                        ? b_cont.reshape({batch_b, next_last_b, last_b})
                        : b_cont;
        
        // Reconstruct output shape
        std::vector<size_t> out_shape;
        if (input_a->data.ndim() >= input_b->data.ndim()) {
             out_shape = shape_a;
             out_shape[out_shape.size() - 1] = last_b;
        } else {
             out_shape = shape_b;
             out_shape[out_shape.size() - 2] = next_last_a;
        }
        res = Tensor(out_shape, input_a->data.dtype());
        Tensor res_3d = res.reshape({batch_a, next_last_a, last_b});
        blas::batched_matmul(a_flat, b_flat, res_3d);
    } else if (input_a->data.ndim() == 3 || input_b->data.ndim() == 3) {
        res = blas::batched_matmul(input_a->data, input_b->data);
    } else {
        res = blas::matmul(input_a->data, input_b->data);
    }
    
    Node *out = new Node(std::move(res));
    out->children = {input_a, input_b};
    out->name = "matmul";

    // Backward pass
    out->backward_fn = [input_a, input_b, out, complex_reshape]() {
        if (input_a->requires_grad) {
            // dL/dA = dL/dOut * B^T
            Tensor b_t = input_b->data.transpose();
            Tensor grad_a({0});
            
            if (complex_reshape || out->grad.ndim() > 3 || b_t.ndim() > 3) {
                // Similar flattening logic for backward
                size_t ma_rows = out->grad.shape()[out->grad.ndim() - 2];
                size_t ma_cols = out->grad.shape().back();
                size_t ma_batch = out->grad.size() / (ma_rows * ma_cols);
                
                size_t mb_rows = b_t.shape()[b_t.ndim() - 2];
                size_t mb_cols = b_t.shape().back();
                size_t mb_batch = b_t.size() / (mb_rows * mb_cols);
                
                Tensor ma_cont = out->grad.contiguous();
                Tensor mb_cont = b_t.contiguous();

                Tensor ma_flat = (out->grad.ndim() > 2)
                                 ? ma_cont.reshape({ma_batch, ma_rows, ma_cols})
                                 : ma_cont;
                Tensor mb_flat = (b_t.ndim() > 2)
                                 ? mb_cont.reshape({mb_batch, mb_rows, mb_cols})
                                 : mb_cont;
                
                grad_a = Tensor(input_a->data.shape(), input_a->data.dtype());
                Tensor grad_a_3d = grad_a.reshape({ma_batch, ma_rows, mb_cols});
                blas::batched_matmul(ma_flat, mb_flat, grad_a_3d);
            } else if (out->grad.ndim() == 3 || b_t.ndim() == 3) {
                grad_a = blas::batched_matmul(out->grad, b_t);
            } else {
                grad_a = blas::matmul(out->grad, b_t);
            }

            if (input_a->grad.ndim() < grad_a.ndim()) {
                input_a->grad += grad_a.sum(0);
            } else {
                input_a->grad += grad_a;
            }
        }
        if (input_b->requires_grad) {
            // dL/dB = A^T * dL/dOut
            Tensor a_t = input_a->data.transpose();
            Tensor grad_b({0});
            
            if (complex_reshape || a_t.ndim() > 3 || out->grad.ndim() > 3) {
                size_t ma_rows = a_t.shape()[a_t.ndim() - 2];
                size_t ma_cols = a_t.shape().back();
                size_t ma_batch = a_t.size() / (ma_rows * ma_cols);
                
                size_t mb_rows = out->grad.shape()[out->grad.ndim() - 2];
                size_t mb_cols = out->grad.shape().back();
                size_t mb_batch = out->grad.size() / (mb_rows * mb_cols);
                
                Tensor ma_cont = a_t.contiguous();
                Tensor mb_cont = out->grad.contiguous();

                Tensor ma_flat = (a_t.ndim() > 2)
                                 ? ma_cont.reshape({ma_batch, ma_rows, ma_cols})
                                 : ma_cont;
                Tensor mb_flat = (out->grad.ndim() > 2)
                                 ? mb_cont.reshape({mb_batch, mb_rows, mb_cols})
                                 : mb_cont;
                
                grad_b = Tensor(input_b->data.shape(), input_b->data.dtype());
                Tensor grad_b_3d = grad_b.reshape({ma_batch, ma_rows, mb_cols});
                blas::batched_matmul(ma_flat, mb_flat, grad_b_3d);
            } else if (a_t.ndim() == 3 || out->grad.ndim() == 3) {
                grad_b = blas::batched_matmul(a_t, out->grad);
            } else {
                grad_b = blas::matmul(a_t, out->grad);
            }

            if (input_b->grad.ndim() < grad_b.ndim()) {
                input_b->grad += grad_b.sum(0);
            } else {
                input_b->grad += grad_b;
            }
        }
    };
    return out;
}

Node *add(Node *input_a, Node *input_b)
{
    // Forward pass
    Tensor res = input_a->data + input_b->data;
    Node *out = new Node(std::move(res));
    out->children = {input_a, input_b};
    out->name = "add";

    // Backward pass
    out->backward_fn = [input_a, input_b, out]() {
        if (input_a->requires_grad) {
            if (input_a->grad.shape() != out->grad.shape()) {
                Tensor sum_grad = out->grad;
                while (sum_grad.ndim() > input_a->grad.ndim()) {
                    sum_grad = sum_grad.sum(0);
                }
                input_a->grad += sum_grad;
            } else {
                input_a->grad += out->grad;
            }
        }
        if (input_b->requires_grad) {
            if (input_b->grad.shape() != out->grad.shape()) {
                Tensor sum_grad = out->grad;
                while (sum_grad.ndim() > input_b->grad.ndim()) {
                    sum_grad = sum_grad.sum(0);
                }
                input_b->grad += sum_grad;
            } else {
                input_b->grad += out->grad;
            }
        }
    };
    return out;
}

Node *relu(Node *input_node)
{
    // Forward pass
    Tensor res(input_node->data.shape());
    for (size_t i = 0; i < input_node->data.size(); ++i) {
        res[i] = std::max(0.0F, input_node->data[i]);
    }
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "relu";

    // Backward pass
    out->backward_fn = [input_node, out]() {
        if (input_node->requires_grad) {
            for (size_t i = 0; i < input_node->data.size(); ++i) {
                if (input_node->data[i] > 0) {
                    input_node->grad[i] += out->grad[i];
                }
            }
        }
    };
    return out;
}

Node *gelu(Node *input_node)
{
    // GELU(x) = 0.5x(1 + erf(x/sqrt(2)))
    // Approx: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
    const float PI_CONST = static_cast<float>(M_PI);
    const float SQRT_2_PI = std::sqrt(2.0F / PI_CONST);

    Tensor res(input_node->data.shape());
    const float GELU_COEFF = 0.044715F;
    const float HALF = 0.5F;
    const float ONE = 1.0F;

    for (size_t i = 0; i < input_node->data.size(); ++i) {
        float x_val = input_node->data[i];
        float x_cubed = x_val * x_val * x_val;
        float inner = SQRT_2_PI * (x_val + (GELU_COEFF * x_cubed));
        res[i] = HALF * x_val * (ONE + std::tanh(inner));
    }

    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "gelu";

    // Backward pass (simplified approximation)
    out->backward_fn = [input_node, out, SQRT_2_PI]() {
        if (input_node->requires_grad) {
            const float GELU_COEFF = 0.044715F;
            const float HALF = 0.5F;
            const float ONE = 1.0F;
            const float THREE = 3.0F;

            for (size_t i = 0; i < input_node->data.size(); ++i) {
                float x_val = input_node->data[i];
                float x_squared = x_val * x_val;
                float x_cubed = x_squared * x_val;
                float inner = SQRT_2_PI * (x_val + (GELU_COEFF * x_cubed));
                float tanh_val = std::tanh(inner);
                float sech2_val = ONE - (tanh_val * tanh_val);

                // d/dx GELU(x) approx
                float deriv = (HALF * (ONE + tanh_val)) +
                              (HALF * x_val * sech2_val * SQRT_2_PI * (ONE + (THREE * GELU_COEFF * x_squared)));
                input_node->grad[i] += out->grad[i] * deriv;
            }
        }
    };
    return out;
}

Node *silu(Node *input_node)
{
    // SiLU(x) = x * sigmoid(x)
    Tensor res(input_node->data.shape());
    for (size_t i = 0; i < input_node->data.size(); ++i) {
        float x = input_node->data[i];
        float s = (x >= 0.0f) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x)));
        res[i] = x * s;
    }
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "silu";
    out->backward_fn = [input_node, out]() {
        if (input_node->requires_grad) {
            for (size_t i = 0; i < input_node->data.size(); ++i) {
                float x = input_node->data[i];
                float s = (x >= 0.0f) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x)));
                float ds = s * (1.0f - s);
                input_node->grad[i] += out->grad[i] * (s + x * ds);
            }
        }
    };
    return out;
}

Node *softmax(Node *input_node, int dimension)
{
    // dimension = -1 means last dim
    if (dimension != -1 && static_cast<size_t>(dimension) != input_node->data.ndim() - 1) {
        throw std::runtime_error("Softmax currently implemented for last dimension only.");
    }

    size_t target_dim = (dimension == -1) ? input_node->data.ndim() - 1 : static_cast<size_t>(dimension);
    size_t k_dim = input_node->data.shape()[target_dim];
    size_t n_dim = input_node->data.size() / k_dim;

    Tensor res(input_node->data.shape());
    const float *src_data = input_node->data.data();
    float *dst_data = res.data();

    const float M_INF = -1e30F;

    for (size_t i = 0; i < n_dim; ++i) {
        float max_val = M_INF;
        for (size_t j = 0; j < k_dim; ++j) {
            max_val = std::max(max_val, src_data[(i * k_dim) + j]);
        }

        float sum_val = 0.0F;
        for (size_t j = 0; j < k_dim; ++j) {
            float exp_val = std::exp(src_data[(i * k_dim) + j] - max_val);
            dst_data[(i * k_dim) + j] = exp_val;
            sum_val += exp_val;
        }

        for (size_t j = 0; j < k_dim; ++j) {
            dst_data[(i * k_dim) + j] /= sum_val;
        }
    }

    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "softmax";

    // Backward pass: dL/dx_i = s_i * (dL/ds_i - sum_j(s_j * dL/ds_j))
    out->backward_fn = [input_node, out, n_dim, k_dim]() {
        if (input_node->requires_grad) {
            const float *softmax_data = out->data.data();
            const float *grad_softmax = out->grad.data();
            float *grad_input = input_node->grad.data();

            for (size_t i = 0; i < n_dim; ++i) {
                float sum_s_grad = 0.0F;
                for (size_t j = 0; j < k_dim; ++j) {
                    sum_s_grad += (softmax_data[(i * k_dim) + j] * grad_softmax[(i * k_dim) + j]);
                }

                for (size_t j = 0; j < k_dim; ++j) {
                    float s_j = softmax_data[(i * k_dim) + j];
                    float ds_j = grad_softmax[(i * k_dim) + j];
                    grad_input[(i * k_dim) + j] += (s_j * (ds_j - sum_s_grad));
                }
            }
        }
    };
    return out;
}

Node *cross_entropy(Node *input_logits, Node *input_targets)
{
    // input_logits: (B, V) or (B, S, V)
    // input_targets: (B,) or (B, S)
    size_t ndim = input_logits->data.ndim();
    size_t batch_size = 0;
    size_t seq_len = 1;
    size_t vocab_size = 0;

    if (ndim == 3) {
        batch_size = input_logits->data.shape()[0];
        seq_len = input_logits->data.shape()[1];
        vocab_size = input_logits->data.shape()[2];
    } else {
        batch_size = input_logits->data.shape()[0];
        vocab_size = input_logits->data.shape()[1];
    }

    size_t total_elements = batch_size * seq_len;
    Tensor prob_tensor(input_logits->data.shape());
    float total_loss = 0.0F;

    const float OFFSET_EPS = 1e-12F;
    const float M_INF = -1e30F;

    // Let's rewrite with better flat loops for performance and 3D support
    const float *logits_raw = input_logits->data.data();
    const float *targets_raw = input_targets->data.data();
    float *probs_raw = prob_tensor.data();

    for (size_t i = 0; i < total_elements; ++i) {
        const float *logits_slice = logits_raw + (i * vocab_size);
        float *probs_slice = probs_raw + (i * vocab_size);

        float max_l = M_INF;
        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            max_l = std::max(max_l, logits_slice[v_idx]);
        }

        float sum_e = 0.0F;
        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            float e_val = std::exp(logits_slice[v_idx] - max_l);
            probs_slice[v_idx] = e_val;
            sum_e += e_val;
        }

        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            probs_slice[v_idx] /= sum_e;
        }

        size_t target_idx = static_cast<size_t>(targets_raw[i]);
        total_loss += -std::log(std::max(probs_slice[target_idx], OFFSET_EPS));
    }

    Node *out = new Node(Tensor({1}));
    out->data[0] = total_loss / static_cast<float>(total_elements);
    out->children = {input_logits};
    out->name = "cross_entropy";

    out->backward_fn = [input_logits, input_targets, out, prob_tensor, total_elements, vocab_size]() {
        if (input_logits->requires_grad) {
            float scale_val = out->grad[0] / static_cast<float>(total_elements);
            const float *targets_raw = input_targets->data.data();
            const float *probs_raw = prob_tensor.data();
            float *grad_raw = input_logits->grad.data();

            for (size_t i = 0; i < total_elements; ++i) {
                size_t target_idx = static_cast<size_t>(targets_raw[i]);
                const float *probs_slice = probs_raw + (i * vocab_size);
                float *grad_slice = grad_raw + (i * vocab_size);

                for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
                    float target_val = (v_idx == target_idx) ? 1.0F : 0.0F;
                    grad_slice[v_idx] += scale_val * (probs_slice[v_idx] - target_val);
                }
            }
        }
    };
    return out;
}

Node *masked_cross_entropy(Node *input_logits, Node *input_targets, const core::Tensor &loss_mask)
{
    const Tensor &logits = input_logits->data;
    const Tensor &targets = input_targets->data;
    if ((logits.ndim() != 2 && logits.ndim() != 3) || logits.shape().empty()) {
        throw std::invalid_argument("masked_cross_entropy: logits must have shape (N, V) or (B, S, V).");
    }

    const size_t vocab_size = logits.shape().back();
    if (vocab_size == 0) {
        throw std::invalid_argument("masked_cross_entropy: vocabulary dimension cannot be empty.");
    }
    const size_t row_count = logits.size() / vocab_size;
    const std::vector<size_t> expected_target_shape =
        logits.ndim() == 2 ? std::vector<size_t>{logits.shape()[0]}
                           : std::vector<size_t>{logits.shape()[0], logits.shape()[1]};
    if (targets.shape() != expected_target_shape || loss_mask.shape() != expected_target_shape) {
        throw std::invalid_argument("masked_cross_entropy: target and mask shapes must match the logits rows.");
    }

    Tensor probabilities(logits.shape());
    const float *logits_data = logits.data();
    const float *targets_data = targets.data();
    const float *mask_data = loss_mask.data();
    float *probabilities_data = probabilities.data();
    float total_loss = 0.0F;
    size_t active_rows = 0;

    for (size_t row = 0; row < row_count; ++row) {
        const float mask_value = mask_data[row];
        if (!std::isfinite(mask_value)) {
            throw std::invalid_argument("masked_cross_entropy: loss mask values must be finite.");
        }
        if (mask_value <= 0.0F) continue;

        const float target_value = targets_data[row];
        if (!std::isfinite(target_value) || target_value < 0.0F ||
            target_value >= static_cast<float>(vocab_size) || std::floor(target_value) != target_value) {
            throw std::invalid_argument("masked_cross_entropy: active targets must be valid vocabulary indices.");
        }

        const float *row_logits = logits_data + row * vocab_size;
        float *row_probabilities = probabilities_data + row * vocab_size;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < vocab_size; ++token) {
            if (!std::isfinite(row_logits[token])) {
                throw std::invalid_argument("masked_cross_entropy: active logits must be finite.");
            }
            max_logit = std::max(max_logit, row_logits[token]);
        }

        float exp_sum = 0.0F;
        for (size_t token = 0; token < vocab_size; ++token) {
            row_probabilities[token] = std::exp(row_logits[token] - max_logit);
            exp_sum += row_probabilities[token];
        }

        const float log_partition = max_logit + std::log(exp_sum);
        total_loss += log_partition - row_logits[static_cast<size_t>(target_value)];
        for (size_t token = 0; token < vocab_size; ++token) {
            row_probabilities[token] /= exp_sum;
        }
        ++active_rows;
    }

    if (active_rows == 0) {
        throw std::invalid_argument("masked_cross_entropy: loss mask must select at least one target.");
    }

    Node *out = new Node(Tensor({1}));
    out->data[0] = total_loss / static_cast<float>(active_rows);
    out->children = {input_logits};
    out->name = "masked_cross_entropy";
    const Tensor mask_copy = loss_mask.copy();
    out->backward_fn = [input_logits, input_targets, out, probabilities, mask_copy, row_count, vocab_size,
                        active_rows]() {
        if (!input_logits->requires_grad) return;

        const float scale = out->grad[0] / static_cast<float>(active_rows);
        const float *target_data = input_targets->data.data();
        const float *mask_values = mask_copy.data();
        const float *probability_values = probabilities.data();
        float *gradient_values = input_logits->grad.data();

        for (size_t row = 0; row < row_count; ++row) {
            if (mask_values[row] <= 0.0F) continue;
            const size_t target = static_cast<size_t>(target_data[row]);
            const float *row_probabilities = probability_values + row * vocab_size;
            float *row_gradient = gradient_values + row * vocab_size;
            for (size_t token = 0; token < vocab_size; ++token) {
                row_gradient[token] += scale * (row_probabilities[token] - (token == target ? 1.0F : 0.0F));
            }
        }
    };
    return out;
}

Node *reshape(Node *input_node, const std::vector<size_t> &new_shape)
{
    Tensor res = input_node->data.reshape(new_shape);
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "reshape";

    std::vector<size_t> old_shape = input_node->data.shape();
    out->backward_fn = [input_node, out, old_shape]() {
        if (input_node->requires_grad) {
            input_node->grad += out->grad.reshape(old_shape);
        }
    };
    return out;
}

Node *transpose(Node *input_node, size_t dim0, size_t dim1)
{
    // Tensor::transpose() uses size_t, but -1/-2 are useful defaults
    size_t dim0_idx = (dim0 == static_cast<size_t>(-2)) ? input_node->data.ndim() - 2 : dim0;
    size_t dim1_idx = (dim1 == static_cast<size_t>(-1)) ? input_node->data.ndim() - 1 : dim1;

    Tensor res = input_node->data.transpose(dim0_idx, dim1_idx);
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "transpose";

    out->backward_fn = [input_node, out, dim0_idx, dim1_idx]() {
        if (input_node->requires_grad) {
            input_node->grad += out->grad.transpose(dim0_idx, dim1_idx);
        }
    };
    return out;
}

Node *masked_fill(Node *input_node, const Tensor &mask_tensor, float fill_value)
{
    Tensor res = input_node->data.copy();
    size_t input_size = input_node->data.size();
    size_t mask_size = mask_tensor.size();

    for (size_t i = 0; i < input_size; ++i) {
        if (mask_tensor[i % mask_size] != 0.0F) {
            res[i] = fill_value;
        }
    }
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "masked_fill";

    out->backward_fn = [input_node, out, mask_tensor]() {
        if (input_node->requires_grad) {
            size_t input_size = input_node->data.size();
            size_t mask_size = mask_tensor.size();
            for (size_t i = 0; i < input_size; ++i) {
                if (mask_tensor[i % mask_size] == 0.0F) {
                    input_node->grad[i] += out->grad[i];
                }
            }
        }
    };
    return out;
}

Node *mul_scalar(Node *input_node, float scalar_value)
{
    Tensor res = input_node->data.copy();
    for (size_t i = 0; i < input_node->data.size(); ++i) {
        res[i] *= scalar_value;
    }
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "mul_scalar";

    out->backward_fn = [input_node, out, scalar_value]() {
        if (input_node->requires_grad) {
            for (size_t i = 0; i < input_node->data.size(); ++i) {
                input_node->grad[i] += out->grad[i] * scalar_value;
            }
        }
    };
    return out;
}

namespace {

float resolve_rope_base(float rope_base, size_t d_head, size_t training_ctx, size_t target_ctx)
{
    if (training_ctx == 0 || target_ctx <= training_ctx) {
        return rope_base;
    }
    if (d_head <= 2) {
        return rope_base;
    }
    const float scale_ratio = static_cast<float>(target_ctx) / static_cast<float>(training_ctx);
    return rope_base * std::pow(scale_ratio, static_cast<float>(d_head) / static_cast<float>(d_head - 2));
}

}  // namespace

void apply_rope_to_data(Tensor &rot_data, size_t batch_size, size_t seq_len, size_t d_head, size_t d_model,
                        bool is_batched, float rope_base, size_t pos_offset, size_t training_ctx,
                        size_t target_ctx)
{
    const float effective_base = resolve_rope_base(rope_base, d_head, training_ctx, target_ctx);
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t p_idx = 0; p_idx < seq_len; ++p_idx) {
            size_t absolute_pos = p_idx + pos_offset;
            for (size_t h_idx = 0; h_idx < d_model; h_idx += d_head) {
                for (size_t i_idx = 0; i_idx < d_head; i_idx += 2) {
                    float freq = 1.0F / std::pow(effective_base, static_cast<float>(i_idx) / static_cast<float>(d_head));
                    float theta = static_cast<float>(absolute_pos) * freq;
                    float cos_val = std::cos(theta);
                    float sin_val = std::sin(theta);

                    float x0_v =
                        is_batched ? rot_data.at(b_idx, p_idx, h_idx + i_idx) : rot_data.at(p_idx, h_idx + i_idx);
                    float x1_v = is_batched ? rot_data.at(b_idx, p_idx, h_idx + i_idx + 1)
                                            : rot_data.at(p_idx, h_idx + i_idx + 1);

                    if (is_batched) {
                        rot_data.at(b_idx, p_idx, h_idx + i_idx) = (x0_v * cos_val) - (x1_v * sin_val);
                        rot_data.at(b_idx, p_idx, h_idx + i_idx + 1) = (x0_v * sin_val) + (x1_v * cos_val);
                    } else {
                        rot_data.at(p_idx, h_idx + i_idx) = (x0_v * cos_val) - (x1_v * sin_val);
                        rot_data.at(p_idx, h_idx + i_idx + 1) = (x0_v * sin_val) + (x1_v * cos_val);
                    }
                }
            }
        }
    }
}

void apply_rope_backward_impl(Node *input_node, Node *out, size_t batch_size, size_t seq_len, size_t d_head,
                              size_t d_model, bool is_batched, float rope_base, size_t pos_offset,
                              size_t training_ctx, size_t target_ctx)
{
    if (!input_node->requires_grad) {
        return;
    }
    const float effective_base = resolve_rope_base(rope_base, d_head, training_ctx, target_ctx);
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t p_idx = 0; p_idx < seq_len; ++p_idx) {
            size_t absolute_pos = p_idx + pos_offset;
            for (size_t h_idx = 0; h_idx < d_model; h_idx += d_head) {
                for (size_t i_idx = 0; i_idx < d_head; i_idx += 2) {
                    float freq = 1.0F / std::pow(effective_base, static_cast<float>(i_idx) / static_cast<float>(d_head));
                    float theta = static_cast<float>(absolute_pos) * freq;
                    float cos_val = std::cos(theta);
                    float sin_val = std::sin(theta);

                    float g0_v =
                        is_batched ? out->grad.at(b_idx, p_idx, h_idx + i_idx) : out->grad.at(p_idx, h_idx + i_idx);
                    float g1_v = is_batched ? out->grad.at(b_idx, p_idx, h_idx + i_idx + 1)
                                            : out->grad.at(p_idx, h_idx + i_idx + 1);

                    if (is_batched) {
                        input_node->grad.at(b_idx, p_idx, h_idx + i_idx) += (g0_v * cos_val) + (g1_v * sin_val);
                        input_node->grad.at(b_idx, p_idx, h_idx + i_idx + 1) += (-(g0_v * sin_val)) + (g1_v * cos_val);
                    } else {
                        input_node->grad.at(p_idx, h_idx + i_idx) += (g0_v * cos_val) + (g1_v * sin_val);
                        input_node->grad.at(p_idx, h_idx + i_idx + 1) += (-(g0_v * sin_val)) + (g1_v * cos_val);
                    }
                }
            }
        }
    }
}

Node *apply_rope_impl(Node *input_node, size_t batch_size, size_t seq_len, size_t d_head, bool is_batched,
                      bool is_query, float rope_base, size_t pos_offset, size_t training_ctx, size_t target_ctx)
{
    Tensor rot_data = input_node->data.copy();
    size_t d_model = is_batched ? input_node->data.shape()[2] : input_node->data.shape()[1];

    apply_rope_to_data(rot_data, batch_size, seq_len, d_head, d_model, is_batched, rope_base, pos_offset,
                       training_ctx, target_ctx);

    Node *out = new Node(std::move(rot_data));
    out->name = is_query ? "rope_q" : "rope_k";
    out->children = {input_node};

    out->backward_fn = [input_node, out, batch_size, seq_len, d_head, d_model, is_batched, rope_base, pos_offset,
                        training_ctx, target_ctx]() {
        apply_rope_backward_impl(input_node, out, batch_size, seq_len, d_head, d_model, is_batched, rope_base,
                                 pos_offset, training_ctx, target_ctx);
    };
    return out;
}

RoPEResult rope(Node *query_node, Node *key_node, size_t d_head, float rope_base, size_t pos_offset,
                size_t training_ctx, size_t target_ctx)
{
    bool is_batched = (query_node->data.ndim() == 3);
    size_t batch_size = is_batched ? query_node->data.shape()[0] : 1;
    size_t seq_len = is_batched ? query_node->data.shape()[1] : query_node->data.shape()[0];

    Node *out_q = apply_rope_impl(query_node, batch_size, seq_len, d_head, is_batched, true, rope_base, pos_offset,
                                 training_ctx, target_ctx);
    Node *out_k = apply_rope_impl(key_node, batch_size, seq_len, d_head, is_batched, false, rope_base, pos_offset,
                                 training_ctx, target_ctx);

    return {out_q, out_k};
}

Node *contiguous(Node *input_node)
{
    if (input_node->data.is_contiguous()) {
        return input_node;
    }

    Node *res = new Node(input_node->data.copy(), input_node->requires_grad);
    res->name = "contiguous(" + input_node->name + ")";
    res->children.push_back(input_node);

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
