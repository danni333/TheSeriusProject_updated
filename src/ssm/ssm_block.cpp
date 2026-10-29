#include "ssm_block.h"

#include <cmath>
#include <vector>

namespace serius {
namespace ssm {

namespace {
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
}  // namespace

MambaBlock::MambaBlock(size_t d_model, size_t d_state, size_t d_conv)
    : d_model_(d_model), d_state_(d_state), d_conv_(d_conv)
{
    A_log_ = core::Tensor({d_state});
    for (size_t i = 0; i < d_state; i++) A_log_[i] = std::log(1.0f + (float)i);
    D_ = core::Tensor({d_model});
    for (size_t i = 0; i < d_model; i++) D_[i] = 1.0f;
    in_proj_ = core::Tensor({hidden_size * 2});  // input projection placeholder allocation
}

core::Tensor MambaBlock::forward(const core::Tensor &x) const
{
    size_t seq = x.shape()[0];
    core::Tensor out({seq, d_model_});

    // state h: [d_state]
    std::vector<float> h(d_state_, 0.0f);

    for (size_t t = 0; t < seq; t++) {
        // dt from input (simplified: scalar positive)
        float dt = 0.1f;
        // A (negative for stability)
        std::vector<float> A(d_state_);
        for (size_t n = 0; n < d_state_; n++) A[n] = -std::exp(A_log_[n]);
        // B, C from input (simplified: use x[t,0..])
        std::vector<float> B(d_state_, 0.0f), C(d_state_, 0.0f);
        for (size_t n = 0; n < d_state_; n++) {
            float xn = (t * d_model_ + (n % d_model_) < x.size()) ? x[t * d_model_ + (n % d_model_)] : 0.0f;
            B[n] = xn;
            C[n] = xn;
        }
        // discretize (ZOH)
        std::vector<float> Abar(d_state_), Bbar(d_state_);
        for (size_t n = 0; n < d_state_; n++) {
            Abar[n] = std::exp(A[n] * dt);
            Bbar[n] = (Abar[n] - 1.0f) / (A[n] + 1e-6f) * B[n];
        }
        // update state
        for (size_t n = 0; n < d_state_; n++) {
            float xt = (n < d_model_) ? x.at(t, n % d_model_) : 0.0f;
            h[n] = Abar[n] * h[n] + Bbar[n] * xt;
        }
        // output
        for (size_t i = 0; i < d_model_; i++) {
            float yi = D_[i] * x.at(t, i);
            for (size_t n = 0; n < d_state_; n++) yi += C[n % d_state_] * h[n];
            out.at(t, i) = yi;
        }
    }
    return out;
}

}  // namespace ssm
}  // namespace serius