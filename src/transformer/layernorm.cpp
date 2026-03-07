#include "layernorm.h"

#include <cmath>
#include <stdexcept>

namespace serius {
namespace transformer {

LayerNorm::LayerNorm(size_t d_model, float eps) : d_model_(d_model), eps_(eps)
{
    Tensor gamma_t({d_model});
    gamma_t.ones();
    gamma = new autograd::Node(std::move(gamma_t));
    gamma->name = "layernorm_gamma";

    Tensor beta_t({d_model});
    beta_t.zeros();
    beta = new autograd::Node(std::move(beta_t));
    beta->name = "layernorm_beta";
}

LayerNorm::~LayerNorm()
{
    delete gamma;
    delete beta;
}

autograd::Node *LayerNorm::forward(autograd::Node *input) const
{
    const Tensor &in_data = input->data;
    bool is_batched = (in_data.ndim() == 3);
    size_t batch_size = is_batched ? in_data.shape()[0] : 1;
    size_t seq_len = is_batched ? in_data.shape()[1] : in_data.shape()[0];
    size_t d_model = is_batched ? in_data.shape()[2] : in_data.shape()[1];

    if (d_model != d_model_) {
        throw std::invalid_argument("LayerNorm input dimension mismatch");
    }

    Tensor output_data(in_data.shape());

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t seq = 0; seq < seq_len; ++seq) {
            float sum = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                sum += is_batched ? in_data.at(b, seq, i) : in_data.at(seq, i);
            }
            float mean = sum / static_cast<float>(d_model);

            float var_sum = 0.0F;
            for (size_t i = 0; i < d_model; ++i) {
                float diff = (is_batched ? in_data.at(b, seq, i) : in_data.at(seq, i)) - mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / static_cast<float>(d_model);

            float inv_stddev = 1.0F / std::sqrt(variance + eps_);
            for (size_t i = 0; i < d_model; ++i) {
                float val = is_batched ? in_data.at(b, seq, i) : in_data.at(seq, i);
                float normalized = (val - mean) * inv_stddev;
                float out_val = gamma->data[i] * normalized + beta->data[i];

                if (is_batched) {
                    output_data.at(b, seq, i) = out_val;
                } else {
                    output_data.at(seq, i) = out_val;
                }
            }
        }
    }

    autograd::Node *out_node = new autograd::Node(std::move(output_data));
    out_node->name = "layernorm";
    out_node->children = {input, gamma, beta};

    out_node->backward_fn = [input, out_node, this, is_batched, batch_size, seq_len, d_model]() {
        if (!input->requires_grad && !gamma->requires_grad && !beta->requires_grad) return;

        const Tensor &x = input->data;
        const Tensor &dy = out_node->grad;

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t seq = 0; seq < seq_len; ++seq) {
                float sum_x = 0.0F;
                for (size_t i = 0; i < d_model; ++i) sum_x += is_batched ? x.at(b, seq, i) : x.at(seq, i);
                float mean = sum_x / d_model;

                float var_sum = 0.0F;
                for (size_t i = 0; i < d_model; ++i) {
                    float diff = (is_batched ? x.at(b, seq, i) : x.at(seq, i)) - mean;
                    var_sum += diff * diff;
                }
                float inv_std = 1.0F / std::sqrt(var_sum / d_model + eps_);

                float sum_dy = 0.0F;
                float sum_dy_xhat = 0.0F;

                for (size_t i = 0; i < d_model; ++i) {
                    float xi = is_batched ? x.at(b, seq, i) : x.at(seq, i);
                    float dyi = is_batched ? dy.at(b, seq, i) : dy.at(seq, i);
                    float xhat = (xi - mean) * inv_std;

                    if (gamma->requires_grad) gamma->grad[i] += dyi * xhat;
                    if (beta->requires_grad) beta->grad[i] += dyi;

                    sum_dy += dyi;
                    sum_dy_xhat += dyi * xhat;
                }

                if (input->requires_grad) {
                    for (size_t i = 0; i < d_model; ++i) {
                        float xi = is_batched ? x.at(b, seq, i) : x.at(seq, i);
                        float dyi = is_batched ? dy.at(b, seq, i) : dy.at(seq, i);
                        float xhat = (xi - mean) * inv_std;

                        float dx = (gamma->data[i] * inv_std / d_model) * (d_model * dyi - sum_dy - xhat * sum_dy_xhat);
                        if (is_batched)
                            input->grad.at(b, seq, i) += dx;
                        else
                            input->grad.at(seq, i) += dx;
                    }
                }
            }
        }
    };

    return out_node;
}

}  // namespace transformer
}  // namespace serius
