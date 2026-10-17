#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace serius {
namespace infer {

DraftModel::DraftModel(ForwardFn forward_fn) : forward_fn_(std::move(forward_fn)) {}

core::Tensor DraftModel::forward(const std::vector<int> &sequence) const
{
    if (!forward_fn_) {
        throw std::logic_error("DraftModel::forward called with no registered model function");
    }
    return forward_fn_(sequence);
}

SpeculativeDecoder::SpeculativeDecoder(ForwardFn draft_model, ForwardFn target_model, Sampler sampler, int gamma,
                                       int eos_token_id, unsigned seed)
    : draft_model_(std::move(draft_model)),
      target_model_(std::move(target_model)),
      sampler_(std::move(sampler)),
      gamma_(std::max(1, gamma)),
      eos_token_id_(eos_token_id),
      rng_(seed == 0 ? static_cast<unsigned>(std::random_device{}()) : seed)
{
    if (!draft_model_) {
        throw std::invalid_argument("SpeculativeDecoder: draft model callback is required");
    }
    if (!target_model_) {
        throw std::invalid_argument("SpeculativeDecoder: target model callback is required");
    }
}

SpeculativeDecoder::SpeculativeDecoder(DraftModel draft_model, DraftModel target_model, Sampler sampler, int gamma,
                                       int eos_token_id, unsigned seed)
    : SpeculativeDecoder(draft_model.valid() ? ForwardFn([draft_model](const std::vector<int> &seq) {
                            return draft_model.forward(seq);
                        })
                                            : ForwardFn(),
                        target_model.valid() ? ForwardFn([target_model](const std::vector<int> &seq) {
                            return target_model.forward(seq);
                        })
                                              : ForwardFn(),
                        std::move(sampler), gamma, eos_token_id, seed)
{
}

core::Tensor SpeculativeDecoder::logits_to_last_row(const core::Tensor &logits)
{
    if (logits.ndim() == 1) {
        return logits.copy();
    }

    if (logits.ndim() < 2) {
        throw std::invalid_argument("SpeculativeDecoder expects 1D or 2D logits");
    }

    const auto &shape = logits.shape();
    const size_t vocab_size = shape.back();
    const size_t seq_len = shape[0];
    const float *src = logits.data() + (seq_len - 1) * vocab_size;
    core::Tensor last({vocab_size});
    float *dst = last.data();
    for (size_t i = 0; i < vocab_size; ++i) {
        dst[i] = src[i];
    }
    return last;
}

core::Tensor SpeculativeDecoder::softmax_distribution(const core::Tensor &logits)
{
    const float *data = logits.data();
    const size_t vocab_size = logits.size();
    if (vocab_size == 0) {
        return core::Tensor({0});
    }

    float max_value = data[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (data[i] > max_value) {
            max_value = data[i];
        }
    }

    core::Tensor probs({vocab_size});
    float *out = probs.data();
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        out[i] = std::exp(data[i] - max_value);
        sum += out[i];
    }

    if (sum <= 0.0f) {
        out[0] = 1.0f;
        for (size_t i = 1; i < vocab_size; ++i) {
            out[i] = 0.0f;
        }
        return probs;
    }

    const float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < vocab_size; ++i) {
        out[i] *= inv_sum;
    }
    return probs;
}

int SpeculativeDecoder::sample_from_distribution(const core::Tensor &probs, std::mt19937 &rng)
{
    const size_t vocab_size = probs.size();
    if (vocab_size == 0) {
        return 0;
    }

    std::vector<float> weights(vocab_size);
    const float *data = probs.data();
    float total = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        weights[i] = std::max(0.0f, data[i]);
        total += weights[i];
    }

    if (total <= 0.0f) {
        return static_cast<int>(std::max_element(data, data + vocab_size) - data);
    }

    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return static_cast<int>(dist(rng));
}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int> &prompt, std::size_t max_tokens) const
{
    if (max_tokens == 0) {
        return prompt;
    }

    std::vector<int> sequence = prompt;
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    metrics_ = Metrics{};
    while (sequence.size() < prompt.size() + max_tokens) {
        std::vector<int> draft_tokens;
        std::vector<core::Tensor> draft_probs;
        std::vector<int> draft_input = sequence;

        for (int i = 0; i < gamma_; ++i) {
            auto logits = draft_model_(draft_input);
            auto probs = softmax_distribution(logits_to_last_row(logits));
            const int token = sample_from_distribution(probs, rng_);
            draft_tokens.push_back(token);
            draft_probs.push_back(probs);
            draft_input.push_back(token);

            if (token == eos_token_id_) {
                break;
            }
        }

        if (draft_tokens.empty()) {
            break;
        }

        metrics_.draft_steps += 1;
        int accepted_count = 0;
        bool rejected = false;

        for (std::size_t i = 0; i < draft_tokens.size(); ++i) {
            std::vector<int> prefix = sequence;
            prefix.insert(prefix.end(), draft_tokens.begin(), draft_tokens.begin() + static_cast<std::ptrdiff_t>(i + 1));

            auto target_logits = target_model_(prefix);
            auto target_probs = softmax_distribution(logits_to_last_row(target_logits));
            const float target_prob = target_probs.item(static_cast<size_t>(draft_tokens[i]));
            const float draft_prob = draft_probs[i].item(static_cast<size_t>(draft_tokens[i]));

            float accept_prob = 1.0f;
            if (draft_prob > 0.0f) {
                accept_prob = std::min(1.0f, target_prob / draft_prob);
            }

            const float roll = uniform_dist(rng_);
            if (roll < accept_prob) {
                sequence.push_back(draft_tokens[i]);
                accepted_count += 1;
                metrics_.accepted_tokens += 1;
            } else {
                metrics_.rejected_tokens += 1;
                core::Tensor corrected = target_probs.copy();
                float *corrected_data = corrected.data();
                const float *draft_data = draft_probs[i].data();
                float total = 0.0f;
                for (size_t j = 0; j < corrected.size(); ++j) {
                    const float value = std::max(0.0f, corrected_data[j] - draft_data[j]);
                    corrected_data[j] = value;
                    total += value;
                }

                if (total <= 0.0f) {
                    corrected = target_probs.copy();
                } else {
                    const float inv_total = 1.0f / total;
                    for (size_t j = 0; j < corrected.size(); ++j) {
                        corrected_data[j] *= inv_total;
                    }
                }

                sequence.push_back(sample_from_distribution(corrected, rng_));
                rejected = true;
                break;
            }
        }

        if (rejected) {
            break;
        }

        if (accepted_count == static_cast<int>(draft_tokens.size())) {
            auto next_logits = target_model_(sequence);
            auto next_probs = softmax_distribution(logits_to_last_row(next_logits));
            const int bonus_token = sample_from_distribution(next_probs, rng_);
            sequence.push_back(bonus_token);
            if (bonus_token == eos_token_id_) {
                break;
            }
        }

        if (sequence.back() == eos_token_id_) {
            break;
        }
    }

    if (metrics_.draft_steps > 0) {
        metrics_.acceptance_rate = static_cast<float>(metrics_.accepted_tokens) / static_cast<float>(metrics_.draft_steps);
    }

    return sequence;
}

}  // namespace infer
}  // namespace serius
