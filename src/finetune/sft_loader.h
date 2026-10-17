#pragma once

#include <string>
#include <vector>

#include "../tokenizer/bpe.h"

namespace serius {
namespace finetune {

struct SFTExample
{
    std::string instruction;
    std::string input;
    std::string output;
};

struct SFTSample
{
    std::vector<TokenId> input_ids;
    std::vector<TokenId> target_ids;  // One token ahead of input_ids
    std::vector<bool> loss_mask;      // Aligned with target_ids; true for response tokens and EOS
};

class SFTDataLoader
{
   public:
    explicit SFTDataLoader(const BPETokenizer &tokenizer) : tokenizer_(tokenizer) {}

    SFTSample prepare(const SFTExample &example) const;

   private:
    const BPETokenizer &tokenizer_;
};

}  // namespace finetune
}  // namespace serius
