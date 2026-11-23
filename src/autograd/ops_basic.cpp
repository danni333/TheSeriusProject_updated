#define _USE_MATH_DEFINES
#include "ops.h"

// Autograd ops — split from ops.cpp. Declarations live in ops.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../blas/blas.h"
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

        Tensor a_flat = (input_a->data.ndim() > 2) ? a_cont.reshape({batch_a, next_last_a, last_a}) : a_cont;
        Tensor b_flat = (input_b->data.ndim() > 2) ? b_cont.reshape({batch_b, next_last_b, last_b}) : b_cont;

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
    out->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([](const std::vector<Node *> &a) { return matmul(a[0], a[1]); }, in);
    };

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

                Tensor ma_flat = (out->grad.ndim() > 2) ? ma_cont.reshape({ma_batch, ma_rows, ma_cols}) : ma_cont;
                Tensor mb_flat = (b_t.ndim() > 2) ? mb_cont.reshape({mb_batch, mb_rows, mb_cols}) : mb_cont;

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

                Tensor ma_flat = (a_t.ndim() > 2) ? ma_cont.reshape({ma_batch, ma_rows, ma_cols}) : ma_cont;
                Tensor mb_flat = (out->grad.ndim() > 2) ? mb_cont.reshape({mb_batch, mb_rows, mb_cols}) : mb_cont;

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
    out->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([](const std::vector<Node *> &a) { return add(a[0], a[1]); }, in);
    };

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
    out->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([](const std::vector<Node *> &a) { return relu(a[0]); }, in);
    };

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
    out->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([](const std::vector<Node *> &a) { return gelu(a[0]); }, in);
    };

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
    out->forward_fn = [](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([](const std::vector<Node *> &a) { return silu(a[0]); }, in);
    };
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
    out->forward_fn = [dimension](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([dimension](const std::vector<Node *> &a) { return softmax(a[0], dimension); },
                                      in);
    };

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
    // prob_tensor is only consumed by backward_fn — skip its (large)
    // allocation and fill entirely during plan replay.
    const bool keep_probs = !detail::in_replay();
    Tensor prob_tensor;
    if (keep_probs) {
        prob_tensor = Tensor(input_logits->data.shape());
    }
    float total_loss = 0.0F;

    const float OFFSET_EPS = 1e-12F;
    const float M_INF = -1e30F;

    // Let's rewrite with better flat loops for performance and 3D support
    const float *logits_raw = input_logits->data.data();
    const float *targets_raw = input_targets->data.data();
    float *probs_raw = keep_probs ? prob_tensor.data() : nullptr;

    for (size_t i = 0; i < total_elements; ++i) {
        const float *logits_slice = logits_raw + (i * vocab_size);
        float *probs_slice = probs_raw != nullptr ? probs_raw + (i * vocab_size) : nullptr;

        float max_l = M_INF;
        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            max_l = std::max(max_l, logits_slice[v_idx]);
        }

        float sum_e = 0.0F;
        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            float e_val = std::exp(logits_slice[v_idx] - max_l);
            if (probs_slice != nullptr) {
                probs_slice[v_idx] = e_val;
            }
            sum_e += e_val;
        }

        size_t target_idx = static_cast<size_t>(targets_raw[i]);
        // p_target = exp(logit_t - max) / sum_e — identical to the normalized
        // probs_slice[target_idx] but computable without storing all probs.
        float p_target;
        if (keep_probs) {
            for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
                probs_slice[v_idx] /= sum_e;
            }
            p_target = probs_slice[target_idx];
        } else {
            p_target = std::exp(logits_slice[target_idx] - max_l) / sum_e;
        }
        total_loss += -std::log(std::max(p_target, OFFSET_EPS));
    }

    Node *out = new Node(Tensor({1}));
    out->data[0] = total_loss / static_cast<float>(total_elements);
    out->children = {input_logits};
    out->name = "cross_entropy";
    // Targets are not graph children; bake them into the replay as a constant.
    if (!detail::in_replay()) {
        out->forward_fn = [targets = input_targets->data](const std::vector<const Tensor *> &in) -> Tensor {
            Node targets_node(targets, false);
            return detail::replay_forward(
                [&targets_node](const std::vector<Node *> &a) { return cross_entropy(a[0], &targets_node); }, in);
        };
    }

    if (!detail::in_replay()) {
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
    }
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
    const std::vector<size_t> expected_target_shape = logits.ndim() == 2
                                                          ? std::vector<size_t>{logits.shape()[0]}
                                                          : std::vector<size_t>{logits.shape()[0], logits.shape()[1]};
    if (targets.shape() != expected_target_shape || loss_mask.shape() != expected_target_shape) {
        throw std::invalid_argument("masked_cross_entropy: target and mask shapes must match the logits rows.");
    }

    // probabilities is only consumed by backward_fn — skip its allocation
    // and fill entirely during plan replay.
    const bool keep_probs = !detail::in_replay();
    Tensor probabilities;
    if (keep_probs) {
        probabilities = Tensor(logits.shape());
    }
    const float *logits_data = logits.data();
    const float *targets_data = targets.data();
    const float *mask_data = loss_mask.data();
    float *probabilities_data = keep_probs ? probabilities.data() : nullptr;
    float total_loss = 0.0F;
    size_t active_rows = 0;

    for (size_t row = 0; row < row_count; ++row) {
        const float mask_value = mask_data[row];
        if (!std::isfinite(mask_value)) {
            throw std::invalid_argument("masked_cross_entropy: loss mask values must be finite.");
        }
        if (mask_value <= 0.0F) continue;

        const float target_value = targets_data[row];
        if (!std::isfinite(target_value) || target_value < 0.0F || target_value >= static_cast<float>(vocab_size) ||
            std::floor(target_value) != target_value) {
            throw std::invalid_argument("masked_cross_entropy: active targets must be valid vocabulary indices.");
        }

        const float *row_logits = logits_data + row * vocab_size;
        float *row_probabilities = probabilities_data != nullptr ? probabilities_data + row * vocab_size : nullptr;
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < vocab_size; ++token) {
            if (!std::isfinite(row_logits[token])) {
                throw std::invalid_argument("masked_cross_entropy: active logits must be finite.");
            }
            max_logit = std::max(max_logit, row_logits[token]);
        }

        float exp_sum = 0.0F;
        for (size_t token = 0; token < vocab_size; ++token) {
            const float e_val = std::exp(row_logits[token] - max_logit);
            if (row_probabilities != nullptr) {
                row_probabilities[token] = e_val;
            }
            exp_sum += e_val;
        }

        const float log_partition = max_logit + std::log(exp_sum);
        total_loss += log_partition - row_logits[static_cast<size_t>(target_value)];
        if (row_probabilities != nullptr) {
            for (size_t token = 0; token < vocab_size; ++token) {
                row_probabilities[token] /= exp_sum;
            }
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
    // Targets and mask are not graph children; bake them into the replay.
    if (!detail::in_replay()) {
        out->forward_fn = [targets = input_targets->data, loss_mask](const std::vector<const Tensor *> &in) -> Tensor {
            Node targets_node(targets, false);
            return detail::replay_forward(
                [&targets_node, &loss_mask](const std::vector<Node *> &a) {
                    return masked_cross_entropy(a[0], &targets_node, loss_mask);
                },
                in);
        };
    }
    if (!detail::in_replay()) {
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
    }
    return out;
}

Node *reshape(Node *input_node, const std::vector<size_t> &new_shape)
{
    Tensor res = input_node->data.reshape(new_shape);
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "reshape";
    out->forward_fn = [new_shape](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward([&new_shape](const std::vector<Node *> &a) { return reshape(a[0], new_shape); },
                                      in);
    };

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
    out->forward_fn = [dim0_idx, dim1_idx](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward(
            [dim0_idx, dim1_idx](const std::vector<Node *> &a) { return transpose(a[0], dim0_idx, dim1_idx); }, in);
    };

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
    if (!detail::in_replay()) {
        out->forward_fn = [mask_tensor, fill_value](const std::vector<const Tensor *> &in) -> Tensor {
            return detail::replay_forward(
                [&mask_tensor, fill_value](const std::vector<Node *> &a) {
                    return masked_fill(a[0], mask_tensor, fill_value);
                },
                in);
        };
    }

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
    out->forward_fn = [scalar_value](const std::vector<const Tensor *> &in) -> Tensor {
        return detail::replay_forward(
            [scalar_value](const std::vector<Node *> &a) { return mul_scalar(a[0], scalar_value); }, in);
    };

    out->backward_fn = [input_node, out, scalar_value]() {
        if (input_node->requires_grad) {
            for (size_t i = 0; i < input_node->data.size(); ++i) {
                input_node->grad[i] += out->grad[i] * scalar_value;
            }
        }
    };
    return out;
}

}  // namespace ops
}  // namespace autograd
}  // namespace core
}  // namespace serius
