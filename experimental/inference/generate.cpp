#include "generate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace serius {
namespace inference {

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

    // Autoregressive generation loop
    for (int step = 0; step < config.max_new_tokens; step++) {
        // Create input tensor from current sequence
        size_t seq_len = result.token_ids.size();
        core::Tensor input_ids({seq_len});
        float *input_data = input_ids.data();
        for (size_t i = 0; i < seq_len; i++) {
            input_data[i] = static_cast<float>(result.token_ids[i]);
        }

        // Create a Node for input (non-trainable)
        core::autograd::Node input_node(std::move(input_ids), false);

        // Forward pass through the model
        core::autograd::Node *logits_node = model.forward(&input_node);

        // Extract logits for the last token position
        // logits_node->data has shape (seq_len, vocab_size)
        const core::Tensor &logits_tensor = logits_node->data;
        size_t vocab_size = logits_tensor.shape().back();

        // Get logits for the last position
        core::Tensor last_logits({vocab_size});
        const float *src = logits_tensor.data() + (seq_len - 1) * vocab_size;
        float *dst = last_logits.data();
        for (size_t i = 0; i < vocab_size; i++) {
            dst[i] = src[i];
        }

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
                              int max_new_tokens, int eos_token_id)
{
    if (prompt_ids.empty()) {
        throw std::invalid_argument("beam_search: prompt_ids cannot be empty");
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

            core::autograd::Node input_node(std::move(input_ids), false);
            core::autograd::Node *logits_node = model.forward(&input_node);

            const core::Tensor &logits_tensor = logits_node->data;
            size_t vocab_size = logits_tensor.shape().back();

            // Get logits for last position
            core::Tensor last_logits({vocab_size});
            const float *src = logits_tensor.data() + (seq_len - 1) * vocab_size;
            float *dst = last_logits.data();
            for (size_t i = 0; i < vocab_size; i++) {
                dst[i] = src[i];
            }

            // Compute log-softmax
            core::Tensor log_probs = Sampler::softmax(last_logits);
            float *lp_data = log_probs.data();
            for (size_t i = 0; i < vocab_size; i++) {
                lp_data[i] = std::log(lp_data[i] + 1e-10f);
            }

            // Find top beam_width tokens
            std::vector<int> indices(vocab_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + beam_width, indices.end(),
                              [lp_data](int a, int b) { return lp_data[a] > lp_data[b]; });

            for (int k = 0; k < beam_width; k++) {
                int idx = indices[k];
                Beam new_beam;
                new_beam.tokens = beam.tokens;
                new_beam.tokens.push_back(idx);
                new_beam.score = beam.score + lp_data[idx];
                new_beam.finished = (idx == eos_token_id);
                candidates.push_back(std::move(new_beam));
            }
        }

        // Keep top beam_width candidates by score
        std::sort(candidates.begin(), candidates.end(),
                  [](const Beam &a, const Beam &b) { return a.score > b.score; });

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
