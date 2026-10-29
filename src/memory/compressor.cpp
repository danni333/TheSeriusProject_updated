#include "compressor.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

#include <nlohmann/json.hpp>

namespace serius {
namespace memory {

namespace {

float dot_product(const float *a, const float *b, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

float vector_norm(const float *a, size_t n)
{
    return std::sqrt(dot_product(a, a, n));
}

void softmax_inplace(float *scores, size_t n)
{
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        max_score = std::max(max_score, scores[i]);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        scores[i] = std::exp(scores[i] - max_score);
        sum += scores[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            scores[i] /= sum;
        }
    }
}

}  // namespace

StateCompressor::StateCompressor(size_t d_model, size_t summary_tokens, unsigned seed)
    : d_model_(d_model), summary_tokens_(summary_tokens)
{
    // Initialize learned summary query vectors with small random values.
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    summary_queries_ = core::Tensor({summary_tokens, d_model});
    for (size_t i = 0; i < summary_queries_.size(); ++i) {
        summary_queries_[i] = dist(rng);
    }
}

CompressionResult StateCompressor::compress(const core::Tensor &hidden_states)
{
    CompressionResult result;

    if (hidden_states.ndim() != 2) {
        result.compressed = hidden_states.copy();
        result.information_ratio = 1.0f;
        return result;
    }

    size_t seq_len = hidden_states.shape()[0];
    result.compressed = core::Tensor({summary_tokens_, d_model_});
    result.compressed.zeros();

    // Cross-attention style pooling:
    // For each summary query, compute attention weights over the sequence
    // and produce a weighted sum.
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_model_));

    for (size_t q = 0; q < summary_tokens_; ++q) {
        const float *query = summary_queries_.data() + q * d_model_;

        // Compute scores.
        std::vector<float> scores(seq_len);
        for (size_t t = 0; t < seq_len; ++t) {
            const float *key = hidden_states.data() + t * d_model_;
            scores[t] = dot_product(query, key, d_model_) * scale;
        }
        softmax_inplace(scores.data(), seq_len);

        // Weighted sum.
        float *out_row = result.compressed.data() + q * d_model_;
        for (size_t t = 0; t < seq_len; ++t) {
            const float *key_row = hidden_states.data() + t * d_model_;
            for (size_t d = 0; d < d_model_; ++d) {
                out_row[d] += scores[t] * key_row[d];
            }
        }
    }

    result.information_ratio = static_cast<float>(summary_tokens_) / static_cast<float>(seq_len);
    return result;
}

core::Tensor StateCompressor::build_context(const std::vector<std::string> &session_paths)
{
    std::vector<core::Tensor> memories;
    for (const auto &path : session_paths) {
        core::Tensor loaded = load_memory(path);
        if (loaded.size() > 0) {
            memories.push_back(std::move(loaded));
        }
    }

    if (memories.empty()) {
        std::vector<size_t> empty_shape = {0, d_model_};
        return core::Tensor(empty_shape);
    }

    // Concatenate along dimension 0.
    size_t total_rows = 0;
    for (const auto &t : memories) {
        total_rows += t.shape()[0];
    }

    core::Tensor context({total_rows, d_model_});
    size_t offset = 0;
    for (const auto &t : memories) {
        for (size_t i = 0; i < t.shape()[0]; ++i) {
            for (size_t d = 0; d < d_model_; ++d) {
                context.at(offset + i, d) = t.at(i, d);
            }
        }
        offset += t.shape()[0];
    }
    return context;
}

bool StateCompressor::save_memory(const std::string &path, const core::Tensor &state) const
{
    // Save as simple binary format: shape[0], shape[1], then raw floats.
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    if (state.ndim() != 2) {
        return false;
    }

    uint32_t rows = static_cast<uint32_t>(state.shape()[0]);
    uint32_t cols = static_cast<uint32_t>(state.shape()[1]);
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char *>(state.data()), rows * cols * sizeof(float));
    return true;
}

core::Tensor StateCompressor::load_memory(const std::string &path) const
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::vector<size_t> empty_shape = {0, d_model_};
        return core::Tensor(empty_shape);
    }

    uint32_t rows = 0, cols = 0;
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    if (!in || rows == 0 || cols == 0) {
        std::vector<size_t> empty_shape = {0, d_model_};
        return core::Tensor(empty_shape);
    }

    core::Tensor state({rows, cols});
    in.read(reinterpret_cast<char *>(state.data()), rows * cols * sizeof(float));
    if (!in) {
        // Partial read: return what we have.
    }
    return state;
}

}  // namespace memory
}  // namespace serius