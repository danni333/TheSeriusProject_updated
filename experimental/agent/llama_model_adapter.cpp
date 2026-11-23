#include "llama_model_adapter.h"

#include <algorithm>

namespace serius {
namespace agent {

LlamaAgentModel::LlamaAgentModel(const model::LLaMAModel &llm,
                                 const BPETokenizer &tokenizer,
                                 uint64_t seed)
    : llm_(llm),
      tokenizer_(tokenizer),
      sampler_(defaults::kTemperature, defaults::kTopK,
               defaults::kTopP, seed)
{
}

std::string LlamaAgentModel::generate(const std::string &prompt,
                                      const SamplingParams &params)
{
    const float temperature = params.temperature > 0.0f ? params.temperature : 1.0f;
    sampler_.set_temperature(temperature);
    const int max_tokens = params.max_tokens > 0 ? params.max_tokens : 128;

    std::vector<TokenId> ids = tokenizer_.encode(prompt);
    if (ids.empty() || ids.front() != tokenizer_.bos_id()) {
        ids.insert(ids.begin(), tokenizer_.bos_id());
    }

    cache_.clear();
    core::Tensor input({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) {
        input[i] = static_cast<float>(ids[i]);
    }

    // Prefill: logits for the last prompt position only.
    core::Tensor logits = infer::llama_forward_inference_last(llm_, input, &cache_);
    if (logits.ndim() == 0 || logits.size() == 0) {
        return {};
    }

    const size_t vocab_size = logits.shape().back();
    auto last_row = [&](const core::Tensor &lg) {
        core::Tensor row({vocab_size});
        const float *src = lg.data() + (lg.shape()[0] - 1) * vocab_size;
        std::copy(src, src + vocab_size, row.data());
        return row;
    };

    int next_id = sampler_.sample(last_row(logits));

    std::string output;
    std::vector<int> recent_ids;
    recent_ids.reserve(ids.size() + static_cast<size_t>(max_tokens));
    for (TokenId tid : ids) {
        recent_ids.push_back(static_cast<int>(tid));
    }

    for (int step = 0; step < max_tokens; ++step) {
        const std::string tok_str = tokenizer_.decode({static_cast<TokenId>(next_id)});
        if (next_id == static_cast<int>(tokenizer_.eos_id()) ||
            tok_str == "<|im_end|>" ||
            std::find(params.stop_tokens.begin(), params.stop_tokens.end(), tok_str) !=
                params.stop_tokens.end()) {
            break;
        }

        output += tok_str;
        ids.push_back(static_cast<TokenId>(next_id));
        recent_ids.push_back(next_id);

        const size_t pos = ids.size() - 1;
        logits = infer::llama_forward_single_token(llm_, next_id, pos, &cache_);
        if (logits.ndim() == 0 || logits.size() == 0) {
            break;
        }
        next_id = sampler_.sample(last_row(logits), recent_ids, 1.0f);
    }

    return output;
}

}  // namespace agent
}  // namespace serius
