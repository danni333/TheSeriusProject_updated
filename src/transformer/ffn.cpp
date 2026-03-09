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

}  // namespace model
}  // namespace serius
