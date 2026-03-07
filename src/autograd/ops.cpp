#define _USE_MATH_DEFINES
#include "ops.h"

#include <algorithm>
#include <cmath>

#include "../blas/blas.h"

namespace serius {
namespace autograd {
namespace ops {

Node *matmul(Node *input_a, Node *input_b)
{
    // Forward pass
    Tensor res({0});
    if (input_a->data.ndim() == 3 || input_b->data.ndim() == 3) {
        res = blas::batched_matmul(input_a->data, input_b->data);
    } else {
        res = blas::matmul(input_a->data, input_b->data);
    }
    Node *out = new Node(std::move(res));
    out->children = {input_a, input_b};
    out->name = "matmul";

    // Backward pass
    out->backward_fn = [input_a, input_b, out]() {
        if (input_a->requires_grad) {
            // dL/dA = dL/dOut * B^T
            Tensor b_t = input_b->data.transpose();
            Tensor grad_a({0});
            if (out->grad.ndim() == 3 || b_t.ndim() == 3) {
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
            if (a_t.ndim() == 3 || out->grad.ndim() == 3) {
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
    // logits: (B, V), targets: (B,)
    size_t batch_size = input_logits->data.shape()[0];
    size_t vocab_size = input_logits->data.shape()[1];

    Tensor prob_tensor(input_logits->data.shape());
    float total_loss = 0.0F;

    const float OFFSET_EPS = 1e-12F;
    const float M_INF = -1e30F;

    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        float max_l = M_INF;
        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            max_l = std::max(max_l, input_logits->data.at(b_idx, v_idx));
        }

        float sum_e = 0.0F;
        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            float e_val = std::exp(input_logits->data.at(b_idx, v_idx) - max_l);
            prob_tensor.at(b_idx, v_idx) = e_val;
            sum_e += e_val;
        }

        for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
            prob_tensor.at(b_idx, v_idx) /= sum_e;
        }

        size_t target_idx = static_cast<size_t>(input_targets->data[b_idx]);
        total_loss += -std::log(std::max(prob_tensor.at(b_idx, target_idx), OFFSET_EPS));
    }

    Node *out = new Node(Tensor({1}));
    out->data[0] = total_loss / static_cast<float>(batch_size);
    out->children = {input_logits};
    out->name = "cross_entropy";

    out->backward_fn = [input_logits, input_targets, out, prob_tensor, batch_size, vocab_size]() {
        if (input_logits->requires_grad) {
            float scale_val = out->grad[0] / static_cast<float>(batch_size);
            for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
                size_t target_idx = static_cast<size_t>(input_targets->data[b_idx]);
                for (size_t v_idx = 0; v_idx < vocab_size; ++v_idx) {
                    float p_val = prob_tensor.at(b_idx, v_idx);
                    float target_val = (v_idx == target_idx) ? 1.0F : 0.0F;
                    input_logits->grad.at(b_idx, v_idx) += scale_val * (p_val - target_val);
                }
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
    size_t d0_val = static_cast<size_t>(dim0);
    size_t d1_val = static_cast<size_t>(dim1);
    size_t dim0_idx = (d0_val == static_cast<size_t>(-2)) ? input_node->data.ndim() - 2 : d0_val;
    size_t dim1_idx = (d1_val == static_cast<size_t>(-1)) ? input_node->data.ndim() - 1 : d1_val;

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
    for (size_t i = 0; i < input_node->data.size(); ++i) {
        if (mask_tensor[i] != 0.0F) {
            res[i] = fill_value;
        }
    }
    Node *out = new Node(std::move(res));
    out->children = {input_node};
    out->name = "masked_fill";

    out->backward_fn = [input_node, out, mask_tensor]() {
        if (input_node->requires_grad) {
            for (size_t i = 0; i < input_node->data.size(); ++i) {
                if (mask_tensor[i] == 0.0F) {
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

void apply_rope_to_data(Tensor &rot_data, size_t batch_size, size_t seq_len, size_t d_head, size_t d_model,
                        bool is_batched, float rope_base)
{
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t p_idx = 0; p_idx < seq_len; ++p_idx) {
            for (size_t h_idx = 0; h_idx < d_model; h_idx += d_head) {
                for (size_t i_idx = 0; i_idx < d_head; i_idx += 2) {
                    float freq = 1.0F / std::pow(rope_base, static_cast<float>(i_idx) / static_cast<float>(d_head));
                    float theta = static_cast<float>(p_idx) * freq;
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
                              size_t d_model, bool is_batched, float rope_base)
{
    if (!input_node->requires_grad) {
        return;
    }
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (size_t p_idx = 0; p_idx < seq_len; ++p_idx) {
            for (size_t h_idx = 0; h_idx < d_model; h_idx += d_head) {
                for (size_t i_idx = 0; i_idx < d_head; i_idx += 2) {
                    float freq = 1.0F / std::pow(rope_base, static_cast<float>(i_idx) / static_cast<float>(d_head));
                    float theta = static_cast<float>(p_idx) * freq;
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
                      bool is_query)
{
    const float ROPE_BASE = 10000.0F;
    Tensor rot_data = input_node->data.copy();
    size_t d_model = is_batched ? input_node->data.shape()[2] : input_node->data.shape()[1];

    apply_rope_to_data(rot_data, batch_size, seq_len, d_head, d_model, is_batched, ROPE_BASE);

    Node *out = new Node(std::move(rot_data));
    out->name = is_query ? "rope_q" : "rope_k";
    out->children = {input_node};

    out->backward_fn = [input_node, out, batch_size, seq_len, d_head, d_model, is_batched, ROPE_BASE]() {
        apply_rope_backward_impl(input_node, out, batch_size, seq_len, d_head, d_model, is_batched, ROPE_BASE);
    };
    return out;
}

RoPEResult rope(Node *query_node, Node *key_node, size_t d_head)
{
    bool is_batched = (query_node->data.ndim() == 3);
    size_t batch_size = is_batched ? query_node->data.shape()[0] : 1;
    size_t seq_len = is_batched ? query_node->data.shape()[1] : query_node->data.shape()[0];

    Node *out_q = apply_rope_impl(query_node, batch_size, seq_len, d_head, is_batched, true);
    Node *out_k = apply_rope_impl(key_node, batch_size, seq_len, d_head, is_batched, false);

    return {out_q, out_k};
}

}  // namespace ops
}  // namespace autograd
}  // namespace serius
