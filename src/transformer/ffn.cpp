#define _USE_MATH_DEFINES
#include "ffn.h"

#include <cmath>

#include "../autograd/ops.h"
#include "../blas/blas.h"

namespace serius {
namespace model {

using core::Tensor;
using core::autograd::Node;

FFN::FFN(size_t d_model, size_t d_ff) : d_model_(d_model), d_ff_(d_ff)
{
    // Xavier initialization
    float scale1 = 1.0F / std::sqrt(static_cast<float>(d_model));
    float scale2 = 1.0F / std::sqrt(static_cast<float>(d_ff));

    Tensor w1_t({d_model, d_ff});
    w1_t.rand();
    for (size_t idx = 0; idx < w1_t.size(); ++idx) {
        w1_t[idx] = (w1_t[idx] - 0.5F) * 2.0F * scale1;
    }
    w1 = new Node(std::move(w1_t));
    w1->name = "ffn_w1";

    Tensor b1_t({d_ff});
    b1_t.zeros();
    b1 = new Node(std::move(b1_t));
    b1->name = "ffn_b1";

    Tensor w2_t({d_ff, d_model});
    w2_t.rand();
    for (size_t idx = 0; idx < w2_t.size(); ++idx) {
        w2_t[idx] = (w2_t[idx] - 0.5F) * 2.0F * scale2;
    }
    w2 = new Node(std::move(w2_t));
    w2->name = "ffn_w2";

    Tensor b2_t({d_model});
    b2_t.zeros();
    b2 = new Node(std::move(b2_t));
    b2->name = "ffn_b2";
}

FFN::~FFN()
{
    delete w1;
    delete b1;
    delete w2;
    delete b2;
}

Node *FFN::forward(Node *input) const
{
    using namespace core::autograd::ops;

    // 1. Linear 1: x * W_1 + b_1
    Node *h1 = matmul(input, w1);
    h1 = add(h1, b1);

    // 2. GELU activation
    h1 = gelu(h1);

    // 3. Linear 2: h1 * W_2 + b_2
    Node *out = matmul(h1, w2);
    out = add(out, b2);

    return out;
}

Node *FFN::forward_optimized(Node *input) const
{
    using namespace core::autograd::ops;

    // 1. Linear 1: x * W_1 (without bias)
    Node *h1 = matmul(input, w1);

    // 2. Fused Bias + GELU activation
    // Ideally this is a specific ops::fused_bias_gelu(h1, b1), but since we
    // are adding this progressively, we'll fall back to add+gelu if the op is not available
    // or we'll assume there is a fused_bias_gelu op. Let's create it in ops.h/cpp.
    // For now we assume ops::fused_bias_gelu is available or will be shortly.
    // h1 = fused_bias_gelu(h1, b1);
    // Since we need to wrap the CUDA kernel `hal::fused_bias_gelu` in an autograd op,
    // let's temporarily use standard add+gelu until we inject the op, or just do standard ops here
    // and rely on block_optimized for broader layer fusion.
    // Wait, the plan says "Integrate the fused_bias_gelu kernel... into FFN/Block logic."
    // Let's assume we map it in ops::fused_bias_gelu later.
    // Actually, I should just implement ops::fused_bias_gelu first.

    // Let's implement the fallback for now and update ops.h next
    h1 = fused_bias_gelu(h1, b1);  // We will add this to ops.h and ops.cpp

    // 3. Linear 2: h1 * W_2 + b_2
    Node *out = matmul(h1, w2);
    out = add(out, b2);

    return out;
}

}  // namespace model
}  // namespace serius
