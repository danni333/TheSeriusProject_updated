#include "generate.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "autograd/engine.h"

namespace serius {
namespace inference {

namespace {

/// Validate that logits are a contiguous (rows, vocab) matrix and return a
/// pointer to the last row. Throws on any other layout: the caller indexes
/// it as flat row-major data.
const float *last_logits_row(const core::Tensor &logits, size_t expected_rows,
                             size_t &vocab_size)
{
    const auto &shape = logits.shape();
    if (shape.size() != 2 || shape[0] != expected_rows || shape[1] == 0 ||
        !logits.is_contiguous()) {
        throw std::runtime_error(
            "generate: logits must be a contiguous (seq_len, vocab) matrix");
    }
    vocab_size = shape[1];
    return logits.data() + (expected_rows - 1) * vocab_size;
}

}  // namespace

GenerateResult generate(model::TransformerModel &model, const std::vector<int> &prompt_ids,
                        const GenerateConfig &config, StreamCallback callback)
{
    if (prompt_ids.empty()) {
        throw std::invalid_argument("generate: prompt_ids cannot be empty");
    }

    Sampler sampler(config.sampler);
    GenerateResult result;
    result.token_ids = prompt_ids;
    result.prompt_len = static_cast<int>(prompt_ids.size());

    // Parameter nodes are owned by the model and must survive graph cleanup.
    std::unordered_set<core::autograd::Node *> params;
    for (core::autograd::Node *p : model.parameters()) {
        if (p) params.insert(p);
    }

    // Autoregressive generation loop
    for (int step = 0; step < config.max_new_tokens; step++) {
        // Create input tensor from current sequence
        size_t seq_len = result.token_ids.size();
        core::Tensor input_ids({seq_len});
        float *input_data = input_ids.data();
        for (size_t i = 0; i < seq_len; i++) {
            input_data[i] = static_cast<float>(result.token_ids[i]);
        }

        // Heap-allocated: the graph stores this pointer as a child of the
        // embedding node, so it must outlive forward() and be freed with the
        // rest of the graph (a stack Node would dangle).
        auto *input_node = new core::autograd::Node(std::move(input_ids), false);

        // Forward pass through the model
        core::autograd::Node *logits_node = model.forward(input_node);

        // Extract logits for the last token position
        // logits_node->data has shape (seq_len, vocab_size)
        const core::Tensor &logits_tensor = logits_node->data;
        size_t vocab_size = 0;
        const float *src = last_logits_row(logits_tensor, seq_len, vocab_size);

        // Get logits for the last position
        core::Tensor last_logits({vocab_size});
        float *dst = last_logits.data();
        for (size_t i = 0; i < vocab_size; i++) {
            dst[i] = src[i];
        }

        // Free this step's autograd graph (params excluded) before the next
        // forward builds a fresh one.
        core::autograd::free_graph(logits_node, params);

        // Sample next token
        int next_token = sampler.sample(last_logits, result.token_ids);
        result.token_ids.push_back(next_token);
        result.generated_len++;

        // Check EOS
        bool is_done = (next_token == config.eos_token_id) || (step == config.max_new_tokens - 1);

        if (next_token == config.eos_token_id) {
            result.hit_eos = true;
        }

        // Stream callback
        if (callback) {
            bool should_continue = callback(next_token, is_done);
            if (!should_continue) break;
        }

        if (is_done) break;
    }

    return result;
}

std::vector<Beam> beam_search(model::TransformerModel &model, const std::vector<int> &prompt_ids, int beam_width,
                              int max_new_tokens, int eos_token_id, float length_penalty)
{
    if (prompt_ids.empty()) {
        throw std::invalid_argument("beam_search: prompt_ids cannot be empty");
    }
    if (beam_width <= 0) {
        beam_width = 1;
    }

    // Parameter nodes are owned by the model and must survive graph cleanup.
    std::unordered_set<core::autograd::Node *> params;
    for (core::autograd::Node *p : model.parameters()) {
        if (p) params.insert(p);
    }

    // Initialize with a single beam containing the prompt
    std::vector<Beam> beams;
    beams.push_back({prompt_ids, 0.0f, false});

    for (int step = 0; step < max_new_tokens; step++) {
        std::vector<Beam> candidates;

        for (auto &beam : beams) {
            if (beam.finished) {
                candidates.push_back(beam);
                continue;
            }

            // Create input tensor
            size_t seq_len = beam.tokens.size();
            core::Tensor input_ids({seq_len});
            float *input_data = input_ids.data();
            for (size_t i = 0; i < seq_len; i++) {
                input_data[i] = static_cast<float>(beam.tokens[i]);
            }

            auto *input_node = new core::autograd::Node(std::move(input_ids), false);
            core::autograd::Node *logits_node = model.forward(input_node);

            const core::Tensor &logits_tensor = logits_node->data;
            size_t vocab_size = 0;
            const float *src = last_logits_row(logits_tensor, seq_len, vocab_size);

            // Get logits for last position
            core::Tensor last_logits({vocab_size});
            float *dst = last_logits.data();
            for (size_t i = 0; i < vocab_size; i++) {
                dst[i] = src[i];
            }
            core::autograd::free_graph(logits_node, params);

            // Compute log-softmax
            core::Tensor log_probs = Sampler::softmax(last_logits);
            float *lp_data = log_probs.data();
            for (size_t i = 0; i < vocab_size; i++) {
                lp_data[i] = std::log(lp_data[i] + 1e-10f);
            }

            // Find top beam_width tokens (clamped to vocab size)
            const int k = std::min(beam_width, static_cast<int>(vocab_size));
            std::vector<int> indices(vocab_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                              [lp_data](int a, int b) { return lp_data[a] > lp_data[b]; });

            for (int j = 0; j < k; j++) {
                int idx = indices[j];
                Beam new_beam;
                new_beam.tokens = beam.tokens;
                new_beam.tokens.push_back(idx);
                new_beam.score = beam.score + lp_data[idx];
                new_beam.finished = (idx == eos_token_id);
                candidates.push_back(std::move(new_beam));
            }
        }

        // Keep top beam_width candidates. With length_penalty > 0 rank by
        // GNMT-style normalized score so finished (shorter) beams aren't
        // systematically disadvantaged by their smaller cumulative |logp|.
        auto rank = [length_penalty](const Beam &b) {
            if (length_penalty <= 0.0f || b.tokens.empty()) {
                return b.score;
            }
            return b.score / std::pow(static_cast<float>(b.tokens.size()), length_penalty);
        };
        std::sort(candidates.begin(), candidates.end(),
                  [&rank](const Beam &a, const Beam &b) { return rank(a) > rank(b); });

        if (static_cast<int>(candidates.size()) > beam_width) {
            candidates.resize(beam_width);
        }
        beams = std::move(candidates);

        // Check if all beams are finished
        bool all_done = true;
        for (const auto &b : beams) {
            if (!b.finished) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;
    }

    return beams;
}

}  // namespace inference
}  // namespace serius
