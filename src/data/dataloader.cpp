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
    if (stopped_ || tokens_.empty() || cursor_ >= tokens_.size()) {
        return std::nullopt;
    }

    const size_t sample_stride = seq_len_ + 1;
    Batch batch(batch_size_, seq_len_);

    for (size_t b = 0; b < batch_size_; ++b) {
        const size_t start = cursor_ + b * sample_stride;

        for (size_t s = 0; s < seq_len_; ++s) {
            const size_t input_index = start + s;
            const size_t target_index = start + s + 1;

            const bool has_input = input_index < tokens_.size();
            const bool has_target = target_index < tokens_.size();

            if (has_input) {
                batch.input_ids.at(b, s) = static_cast<float>(tokens_[input_index]);
                batch.padding_mask.at(b, s) = 1.0f;
            } else {
                batch.input_ids.at(b, s) = static_cast<float>(BPETokenizer::PAD_ID);
                batch.padding_mask.at(b, s) = 0.0f;
            }

            if (has_target) {
                batch.target_ids.at(b, s) = static_cast<float>(tokens_[target_index]);
            } else {
                batch.target_ids.at(b, s) = static_cast<float>(BPETokenizer::PAD_ID);
            }
        }
    }

    cursor_ = std::min(tokens_.size(), cursor_ + batch_size_ * sample_stride);
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
