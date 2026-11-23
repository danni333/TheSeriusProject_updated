#include "ssm_block.h"

#include <cmath>
#include <vector>

namespace serius {
namespace ssm {

namespace {
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
}  // namespace

MambaBlock::MambaBlock(size_t d_model, size_t d_state, size_t d_conv)
    : d_model_(d_model), d_state_(d_state), d_conv_(d_conv)
{
    A_log_ = core::Tensor({d_state});
    for (size_t i = 0; i < d_state; i++) A_log_[i] = std::log(1.0f + (float)i);
    D_ = core::Tensor({d_model});
    for (size_t i = 0; i < d_model; i++) D_[i] = 1.0f;
    // Input projection: x[t] -> [B(d_state), C(d_state), dt(1)].
    // Row-major [d_model, 2*d_state+1]; small deterministic init so early
    // activations stay in a stable range before training.
    in_proj_ = core::Tensor({d_model_, 2 * d_state_ + 1});
    for (size_t i = 0; i < in_proj_.size(); i++) {
        in_proj_[i] = ((int)(i * 31 + 7) % 17 - 8) * 0.005f;
    }
}

core::Tensor MambaBlock::forward(const core::Tensor &x) const
{
    size_t seq = x.shape()[0];
    core::Tensor out({seq, d_model_});
    const size_t proj_dim = 2 * d_state_ + 1;

    // state h: [d_state]
    std::vector<float> h(d_state_, 0.0f);

    for (size_t t = 0; t < seq; t++) {
        // Project input: proj = x[t] @ in_proj_  (x row [d_model], W [d_model, proj_dim])
        std::vector<float> proj(proj_dim, 0.0f);
        for (size_t i = 0; i < d_model_; i++) {
            const float xi = x.at(t, i);
            if (xi == 0.0f) continue;
            const size_t row = i * proj_dim;
            for (size_t k = 0; k < proj_dim; k++) {
                proj[k] += xi * in_proj_[row + k];
            }
        }

        // dt must be positive for discretization stability (softplus).
        const float dt = softplus(proj[2 * d_state_]);
        const float *B = proj.data();
        const float *C = proj.data() + d_state_;

        // A (negative for stability)
        std::vector<float> A(d_state_);
        for (size_t n = 0; n < d_state_; n++) A[n] = -std::exp(A_log_[n]);

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
            for (size_t n = 0; n < d_state_; n++) yi += C[n] * h[n];
            out.at(t, i) = yi;
        }
    }
    return out;
}

}  // namespace ssm
}  // namespace serius
