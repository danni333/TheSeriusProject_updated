#include "lora_linear.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace serius {
namespace lora {

LoRALinear::LoRALinear(core::autograd::Node &W, int rank, float alpha)
    : W_(W), rank_(rank), alpha_(alpha > 0 ? alpha : static_cast<float>(rank))
{
    if (W.data.ndim() != 2) {
        throw std::invalid_argument("LoRALinear: W must be 2D (d_out x d_in)");
    }

    size_t d_out = W.data.shape()[0];
    size_t d_in = W.data.shape()[1];

    // A: (rank x d_in) - initialized with small random values (Kaiming uniform)
    A_ = new core::autograd::Node(core::Tensor({static_cast<size_t>(rank), d_in}), true);
    A_->name = "lora_A";
    {
        float limit = std::sqrt(1.0f / static_cast<float>(d_in));
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-limit, limit);
        float *data = A_->data.data();
        for (size_t i = 0; i < A_->data.size(); i++) {
            data[i] = dist(gen);
        }
    }

    // B: (d_out x rank) - initialized to zero (so initial delta = 0)
    B_ = new core::autograd::Node(core::Tensor({d_out, static_cast<size_t>(rank)}), true);
    B_->name = "lora_B";
    B_->data.fill(0.0f);
}

core::Tensor LoRALinear::forward(const core::Tensor &x) const
{
    // base = W * x^T -> we need to handle the shapes correctly
    // x: (seq_len, d_in) or (d_in,)
    // W: (d_out, d_in)
    // result: (seq_len, d_out) or (d_out,)

    core::Tensor base = core::blas::matmul(x, W_.data.transpose());

    if (merged_) {
        return base;  // LoRA is already merged into W
    }

    // lora_out = x * A^T * B^T
    // x * A^T: (seq_len, rank)
    // * B^T: (seq_len, d_out)
    core::Tensor xa = core::blas::matmul(x, A_->data.transpose());
    core::Tensor lora_out = core::blas::matmul(xa, B_->data.transpose());

    // Scale and add
    float s = scaling();
    core::Tensor scaled = core::blas::scale(lora_out, s);

    core::Tensor result(base.shape());
    core::blas::element_wise_add(base, scaled, result);
    return result;
}

void LoRALinear::merge()
{
    if (merged_) return;

    // W = W + (alpha/rank) * B * A
    // B: (d_out, rank), A: (rank, d_in) -> B*A: (d_out, d_in)
    core::Tensor delta = core::blas::matmul(B_->data, A_->data);
    core::Tensor scaled_delta = core::blas::scale(delta, scaling());

    float *w = W_.data.data();
    const float *d = scaled_delta.data();
    for (size_t i = 0; i < W_.data.size(); i++) {
        w[i] += d[i];
    }

    merged_ = true;
}

void LoRALinear::unmerge()
{
    if (!merged_) return;

    // W = W - (alpha/rank) * B * A
    core::Tensor delta = core::blas::matmul(B_->data, A_->data);
    core::Tensor scaled_delta = core::blas::scale(delta, scaling());

    float *w = W_.data.data();
    const float *d = scaled_delta.data();
    for (size_t i = 0; i < W_.data.size(); i++) {
        w[i] -= d[i];
    }

    merged_ = false;
}

std::vector<core::autograd::Node *> LoRALinear::parameters()
{
    return {A_, B_};
}

}  // namespace lora
}  // namespace serius
