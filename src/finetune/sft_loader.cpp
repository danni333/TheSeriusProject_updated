#include "sft_loader.h"

#include <stdexcept>
#include <string>

namespace serius {
namespace finetune {

SFTSample SFTDataLoader::prepare(const SFTExample &example) const
{
    if (example.instruction.empty()) {
        throw std::invalid_argument("SFTDataLoader::prepare: instruction cannot be empty.");
    }
    if (example.output.empty()) {
        throw std::invalid_argument("SFTDataLoader::prepare: output cannot be empty.");
    }

    std::string prompt = "[SYSTEM] Sei un assistente utile.\n[USER] " + example.instruction;
    if (!example.input.empty()) {
        prompt += " " + example.input;
    }
    prompt += "\n[ASSISTANT] ";

    std::vector<TokenId> prompt_ids = tokenizer_.encode(prompt);
    std::vector<TokenId> response_ids = tokenizer_.encode(example.output);
    if (prompt_ids.empty() || response_ids.empty()) {
        throw std::runtime_error("SFTDataLoader::prepare: tokenizer returned an empty prompt or response.");
    }

    std::vector<TokenId> sequence;
    sequence.reserve(prompt_ids.size() + response_ids.size() + 1);
    sequence.insert(sequence.end(), prompt_ids.begin(), prompt_ids.end());
    sequence.insert(sequence.end(), response_ids.begin(), response_ids.end());
    sequence.push_back(BPETokenizer::EOS_ID);

    SFTSample sample;
    sample.input_ids.assign(sequence.begin(), sequence.end() - 1);
    sample.target_ids.assign(sequence.begin() + 1, sequence.end());
    sample.loss_mask.resize(sample.target_ids.size(), false);

    for (size_t i = 0; i < sample.loss_mask.size(); ++i) {
        sample.loss_mask[i] = i + 1 >= prompt_ids.size();
    }

    return sample;
}

}  // namespace finetune
}  // namespace serius
