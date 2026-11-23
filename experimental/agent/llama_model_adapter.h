#pragma once

#include "model_interface.h"

#include "../infer/llama_inference.h"
#include "../infer/sampler.h"
#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"

namespace serius {
namespace agent {

/// agent::Model adapter backed by a real LLaMAModel + BPETokenizer.
///
/// generate() runs the production inference path: KV-cache prefill via
/// llama_forward_inference_last, then single-token decode with
/// llama_forward_single_token and infer::Sampler. This lets the whole
/// agent stack (ReAct, Planner, Supervisor, DelphiConsensus) drive an
/// actual loaded model instead of a mock.
///
/// The adapter is single-threaded: it owns one KVCache that is cleared at
/// the start of every generate() call. Do not share one adapter across
/// threads; create one per worker.
class LlamaAgentModel final : public Model
{
   public:
    /// @param llm       Loaded LLaMA model (weights read-only).
    /// @param tokenizer Tokenizer used for encode/decode and EOS/BOS ids.
    /// @param seed       RNG seed for the internal sampler.
    LlamaAgentModel(const model::LLaMAModel &llm,
                    const BPETokenizer &tokenizer,
                    uint64_t seed = 0);

    std::string generate(const std::string &prompt,
                         const SamplingParams &params = {}) override;

   private:
    const model::LLaMAModel &llm_;
    const BPETokenizer &tokenizer_;
    infer::Sampler sampler_;
    infer::KVCache cache_;
};

}  // namespace agent
}  // namespace serius
