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
    if (tokens_.empty() || cursor_ >= tokens_.size()) return std::nullopt;

    size_t tokens_needed_per_sample = seq_len_ + 1;

    Batch batch(batch_size_, seq_len_);

    for (size_t b = 0; b < batch_size_; b++) {
        size_t start = cursor_ + b * tokens_needed_per_sample;
        size_t available = (start < tokens_.size()) ? (tokens_.size() - start) : 0;
        size_t input_len = std::min(seq_len_, available);

        for (size_t s = 0; s < seq_len_; s++) {
            if (s < input_len) {
                batch.input_ids.at(b, s) = static_cast<float>(tokens_[start + s]);
                batch.target_ids.at(b, s) = (s + 1 < available)
                    ? static_cast<float>(tokens_[start + s + 1])
                    : static_cast<float>(BPETokenizer::PAD_ID);
                batch.padding_mask.at(b, s) = 1.0f;
            } else {
                batch.input_ids.at(b, s) = static_cast<float>(BPETokenizer::PAD_ID);
                batch.target_ids.at(b, s) = static_cast<float>(BPETokenizer::PAD_ID);
                batch.padding_mask.at(b, s) = 0.0f;
            }
        }
    }

    cursor_ += batch_size_ * tokens_needed_per_sample;
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