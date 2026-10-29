#include "dnc.h"

#include <cmath>
#include <vector>

namespace serius {
namespace memory {

namespace {
float dot(const core::Tensor &a, const core::Tensor &b, size_t offA, size_t offB, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; i++) s += a[offA + i] * b[offB + i];
    return s;
}
float norm(const core::Tensor &a, size_t off, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; i++) s += a[off + i] * a[off + i];
    return std::sqrt(s + 1e-8f);
}
}  // namespace

DNCMemory::DNCMemory(size_t n_slots, size_t d_mem)
    : n_(n_slots), d_(d_mem), M_({n_slots, d_mem}), keys_({n_slots, d_mem}), usage_({n_slots})
{
    M_.zeros();
    keys_.zeros();
    usage_.zeros();
}

void DNCMemory::write(const core::Tensor &write_key, const core::Tensor &write_vector,
                      float write_strength, float free_gate)
{
    // Content addressing weights based on keys_
    std::vector<float> w(n_, 0.0f);
    bool any_used = false;
    for (size_t i = 0; i < n_; i++) {
        if (usage_[i] > 1e-6f) {
            any_used = true;
            break;
        }
    }

    if (any_used) {
        float maxs = -1e9f;
        for (size_t i = 0; i < n_; i++) {
            if (usage_[i] < 1e-6f) {
                w[i] = -1e9f;
                continue;
            }
            float nK = norm(write_key, 0, d_);
            float nM = norm(keys_, i * d_, d_);
            float sim = (nK > 1e-6f && nM > 1e-6f) ? dot(write_key, keys_, 0, i * d_, d_) / (nK * nM) : 0.0f;
            w[i] = sim * write_strength;
            if (w[i] > maxs) maxs = w[i];
        }
        float sum = 0.0f;
        for (size_t i = 0; i < n_; i++) {
            if (w[i] <= -1e8f) {
                w[i] = 0.0f;
            } else {
                w[i] = std::exp(w[i] - maxs);
                sum += w[i];
            }
        }
        for (size_t i = 0; i < n_; i++) w[i] /= (sum + 1e-8f);
    } else {
        for (size_t i = 0; i < n_; i++) w[i] = 1.0f / static_cast<float>(n_);
    }

    // Allocation: find least used slot
    size_t alloc = 0;
    float minu = usage_[0];
    for (size_t i = 1; i < n_; i++) {
        if (usage_[i] < minu) {
            minu = usage_[i];
            alloc = i;
        }
    }
    std::vector<float> aw(n_, 0.0f);
    aw[alloc] = 1.0f;

    // Combine content and allocation
    for (size_t i = 0; i < n_; i++) {
        float wi = free_gate * w[i] + (1.0f - free_gate) * aw[i];
        for (size_t j = 0; j < d_; j++) {
            M_[i * d_ + j] = (1.0f - wi) * M_[i * d_ + j] + wi * write_vector[j];
            keys_[i * d_ + j] = (1.0f - wi) * keys_[i * d_ + j] + wi * write_key[j];
        }
        usage_[i] = usage_[i] * 0.9f + wi;
    }
}

core::Tensor DNCMemory::read(const core::Tensor &read_key, float read_strength) const
{
    bool any_used = false;
    for (size_t i = 0; i < n_; i++) {
        if (usage_[i] > 1e-6f) {
            any_used = true;
            break;
        }
    }

    if (!any_used) {
        core::Tensor out({d_});
        out.zeros();
        return out;
    }

    std::vector<float> w(n_, 0.0f);
    float maxs = -1e9f;
    for (size_t i = 0; i < n_; i++) {
        if (usage_[i] < 1e-6f) {
            w[i] = -1e9f;
            continue;
        }
        float nK = norm(read_key, 0, d_);
        float nM = norm(keys_, i * d_, d_);
        float sim = (nK > 1e-6f && nM > 1e-6f) ? dot(read_key, keys_, 0, i * d_, d_) / (nK * nM) : 0.0f;
        w[i] = sim * read_strength;
        if (w[i] > maxs) maxs = w[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n_; i++) {
        if (w[i] <= -1e8f) {
            w[i] = 0.0f;
        } else {
            w[i] = std::exp(w[i] - maxs);
            sum += w[i];
        }
    }
    for (size_t i = 0; i < n_; i++) w[i] /= (sum + 1e-8f);

    core::Tensor out({d_});
    out.zeros();
    for (size_t i = 0; i < n_; i++) {
        if (w[i] > 0.0f) {
            for (size_t j = 0; j < d_; j++) out[j] += w[i] * M_[i * d_ + j];
        }
    }
    return out;
}

}  // namespace memory
}  // namespace serius