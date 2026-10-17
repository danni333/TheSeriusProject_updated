#include "toxicity.h"

#include <cmath>
#include <random>

#include "../init/weight_init.h"

namespace serius {
namespace safety {

ToxicityClassifier::ToxicityClassifier(const Config& config)
    : config_(config), threshold_(config.threshold)
{
    // Initialize embedding with small random values
    embedding_ = Tensor({config_.vocab_size, config_.embedding_dim});
    init::embedding_init(embedding_);

    // Initialize hidden layer weights
    hidden_weight_ = Tensor({config_.embedding_dim, config_.hidden_dim});
    init::xavier_uniform(hidden_weight_, config_.embedding_dim, config_.hidden_dim);

    hidden_bias_ = Tensor({config_.hidden_dim});
    hidden_bias_.fill(0.0f);

    // Initialize output layer weights
    output_weight_ = Tensor({config_.hidden_dim, 1});
    init::xavier_uniform(output_weight_, config_.hidden_dim, 1);

    output_bias_ = Tensor({1});
    output_bias_.fill(0.0f);
}

ToxicityClassifier::~ToxicityClassifier()
{
}

float ToxicityClassifier::compute_embedding_mean(const std::vector<int>& tokens) const
{
    if (tokens.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    size_t valid_count = 0;

    for (int token : tokens) {
        if (token >= 0 && static_cast<size_t>(token) < config_.vocab_size) {
            // Compute mean of embedding vector for this token
            float token_sum = 0.0f;
            for (size_t i = 0; i < config_.embedding_dim; ++i) {
                token_sum += embedding_.at({static_cast<size_t>(token), i});
            }
            sum += token_sum / config_.embedding_dim;
            valid_count++;
        }
    }

    return valid_count > 0 ? (sum / valid_count) : 0.0f;
}

float ToxicityClassifier::score(const std::vector<int>& tokens) const
{
    if (tokens.empty()) {
        return 0.0f;
    }

    // Step 1: Compute mean embedding
    float embedding_mean = compute_embedding_mean(tokens);

    // Step 2: Hidden layer: linear(embedding_mean) + bias -> relu
    float hidden = 0.0f;
    for (size_t i = 0; i < config_.hidden_dim; ++i) {
        float h = embedding_mean * hidden_weight_.at(0, i) + hidden_bias_.at({i});
        hidden += std::max(0.0f, h);  // ReLU
    }
    hidden /= config_.hidden_dim;  // Average

    // Step 3: Output layer: linear(hidden) + bias -> sigmoid
    float output = hidden * output_weight_.at(0, 0) + output_bias_.at({0});
    float sigmoid = 1.0f / (1.0f + std::exp(-output));

    return sigmoid;
}

bool ToxicityClassifier::is_toxic(const std::vector<int>& tokens) const
{
    return score(tokens) > threshold_;
}

}  // namespace safety
}  // namespace serius
