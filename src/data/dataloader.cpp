#include "dataloader.h"

#include <fstream>
#include <sstream>

namespace serius {

DataLoader::DataLoader(std::shared_ptr<BPETokenizer> tokenizer, size_t batch_size, size_t seq_len, size_t num_workers)
    : tokenizer_(std::move(tokenizer)), batch_size_(batch_size), seq_len_(seq_len), num_workers_(num_workers)
{
}

void DataLoader::load_files(const std::vector<std::string> &file_paths)
{
    tokens_.clear();
    for (const auto &path : file_paths) {
        std::ifstream file(path);
        if (!file.is_open()) continue;
        std::ostringstream ss;
        ss << file.rdbuf();
        auto file_tokens = tokenizer_->encode(ss.str());
        tokens_.insert(tokens_.end(), file_tokens.begin(), file_tokens.end());
    }
    cursor_ = 0;
}

std::optional<Batch> DataLoader::next_batch()
{
    if (stopped_) return std::nullopt;

    // We need at least seq_len + 1 tokens for input/target pair
    size_t tokens_needed_per_sample = seq_len_ + 1;

    // Check if we have enough tokens for at least one sample
    if (cursor_ + tokens_needed_per_sample > tokens_.size()) {
        return std::nullopt;
    }

    // Determine actual batch size (might be smaller at the end)
    size_t actual_batch = 0;
    for (size_t b = 0; b < batch_size_; b++) {
        size_t start = cursor_ + b * tokens_needed_per_sample;
        if (start + tokens_needed_per_sample <= tokens_.size()) {
            actual_batch++;
        }
    }

    if (actual_batch == 0) return std::nullopt;

    Batch batch(actual_batch, seq_len_);

    for (size_t b = 0; b < actual_batch; b++) {
        size_t start = cursor_ + b * tokens_needed_per_sample;
        size_t available = std::min(tokens_needed_per_sample, tokens_.size() - start);
        size_t input_len = std::min(seq_len_, available > 0 ? available - 1 : 0);

        for (size_t s = 0; s < seq_len_; s++) {
            if (s < input_len) {
                batch.input_ids.at(b, s) = static_cast<float>(tokens_[start + s]);
                batch.target_ids.at(b, s) = static_cast<float>(tokens_[start + s + 1]);
                batch.padding_mask.at(b, s) = 1.0f;
            } else {
                batch.input_ids.at(b, s) = static_cast<float>(BPETokenizer::PAD_ID);
                batch.target_ids.at(b, s) = static_cast<float>(BPETokenizer::PAD_ID);
                batch.padding_mask.at(b, s) = 0.0f;
            }
        }
    }

    cursor_ += actual_batch * tokens_needed_per_sample;
    return batch;
}

void DataLoader::reset()
{
    cursor_ = 0;
    stopped_ = false;
}

void DataLoader::stop()
{
    stopped_ = true;
}

}  // namespace serius
