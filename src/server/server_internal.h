#pragma once

// Internal helpers shared between server.cpp (HTTP routes) and
// server_generation.cpp (prompt building, generation, streaming).
// Not part of the public API — do not include from outside src/server/.

#include <httplib.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "../infer/llama_inference.h"
#include "../infer/sampler.h"
#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"
#include "../transformer/model.h"
#include "metrics.h"

namespace serius {
namespace server {

struct GenerationResult
{
    std::string text;
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
};

std::string extract_user_prompt(const nlohmann::json &body);
std::string build_chat_prompt(const nlohmann::json &body, bool chatml);

/// True when tokenized `prompt` + `max_new` exceeds the model's declared
/// context window (max_position_embeddings). False when the window is
/// unknown (0). Lets routes reject oversized requests with a 400 instead of
/// growing the KV cache without bound.
bool prompt_exceeds_context(const model::LLaMAModel &model, BPETokenizer &tokenizer, const std::string &prompt,
                            size_t max_new);

core::Tensor prefill_llama(const model::LLaMAModel &model, const std::vector<TokenId> &ids, infer::KVCache &cache,
                           std::vector<TokenId> *kv_ids);

GenerationResult generate_llama(const model::LLaMAModel &model, BPETokenizer &tokenizer, const std::string &prompt,
                                size_t max_new_tokens, const infer::Sampler &sampler, infer::KVCache &cache,
                                InferenceMetrics *metrics, std::vector<TokenId> *kv_ids = nullptr);

void stream_llama(const model::LLaMAModel &model, BPETokenizer &tokenizer, const std::string &prompt,
                  size_t max_new_tokens, const infer::Sampler &sampler, infer::KVCache &cache,
                  const std::string &req_id, const std::string &model_name, httplib::DataSink &sink,
                  InferenceMetrics *metrics, std::vector<TokenId> *kv_ids = nullptr);

GenerationResult generate_transformer(const model::TransformerModel &model, BPETokenizer &tokenizer,
                                      const std::string &prompt, size_t max_new_tokens, const infer::Sampler &sampler,
                                      InferenceMetrics *metrics);

void stream_transformer(const model::TransformerModel &model, BPETokenizer &tokenizer, const std::string &prompt,
                        size_t max_new_tokens, const infer::Sampler &sampler, const std::string &req_id,
                        const std::string &model_name, httplib::DataSink &sink, InferenceMetrics *metrics);

GenerationResult fallback_generation(const std::string &prompt);

void stream_fallback(const std::string &content, const std::string &req_id, const std::string &model_name,
                     httplib::DataSink &sink);

}  // namespace server
}  // namespace serius
