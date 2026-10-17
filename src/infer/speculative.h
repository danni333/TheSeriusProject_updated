#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "sampler.h"

namespace serius {
namespace infer {

class DraftModel
{
public:
    using ForwardFn = std::function<core::Tensor(const std::vector<int> &)>;

    DraftModel() = default;
    explicit DraftModel(ForwardFn forward_fn);

    core::Tensor forward(const std::vector<int> &sequence) const;
    bool valid() const noexcept { return static_cast<bool>(forward_fn_); }

private:
    ForwardFn forward_fn_;
};

class SpeculativeDecoder
{
public:
    using ForwardFn = std::function<core::Tensor(const std::vector<int> &)>;

    struct Metrics
    {
        std::size_t draft_steps = 0;
        std::size_t accepted_tokens = 0;
        std::size_t rejected_tokens = 0;
        float acceptance_rate = 0.0f;
    };

    SpeculativeDecoder(ForwardFn draft_model, ForwardFn target_model, Sampler sampler = Sampler(), int gamma = 4,
                       int eos_token_id = 2, unsigned seed = 0);
    SpeculativeDecoder(DraftModel draft_model, DraftModel target_model, Sampler sampler = Sampler(), int gamma = 4,
                       int eos_token_id = 2, unsigned seed = 0);

    std::vector<int> generate(const std::vector<int> &prompt, std::size_t max_tokens) const;

    void set_gamma(int gamma) { gamma_ = std::max(1, gamma); }
    int gamma() const noexcept { return gamma_; }
    const Metrics &metrics() const noexcept { return metrics_; }

private:
    static core::Tensor logits_to_last_row(const core::Tensor &logits);
    static core::Tensor softmax_distribution(const core::Tensor &logits);
    static int sample_from_distribution(const core::Tensor &probs, std::mt19937 &rng);

    ForwardFn draft_model_;
    ForwardFn target_model_;
    mutable Sampler sampler_;
    int gamma_;
    int eos_token_id_;
    mutable std::mt19937 rng_;
    mutable Metrics metrics_;
};

}  // namespace infer
}  // namespace serius
