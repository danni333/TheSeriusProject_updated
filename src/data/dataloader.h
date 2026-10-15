#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "../tokenizer/bpe.h"

namespace serius {

/// A single training batch containing input_ids, target_ids, and padding mask.
struct Batch
{
    core::Tensor input_ids;    // (batch_size, seq_len)
    core::Tensor target_ids;   // (batch_size, seq_len)
    core::Tensor padding_mask; // (batch_size, seq_len) - 1.0 for real tokens, 0.0 for padding

    Batch(size_t batch_size, size_t seq_len)
        : input_ids({batch_size, seq_len}),
          target_ids({batch_size, seq_len}),
          padding_mask({batch_size, seq_len})
    {
    }

    Batch(Batch &&other) noexcept = default;
    Batch &operator=(Batch &&other) noexcept = default;
};

/// Simple DataLoader that reads text files, tokenizes them, and produces batches.
class DataLoader
{
   public:
    DataLoader(std::shared_ptr<BPETokenizer> tokenizer, size_t batch_size, size_t seq_len, size_t num_workers = 1);
    ~DataLoader() = default;

    /// Load and tokenize text files.
    void load_files(const std::vector<std::string> &file_paths);

    /// Get the next batch. Returns nullopt when all data has been consumed.
    std::optional<Batch> next_batch();

    /// Reset the cursor to the beginning.
    void reset();

    /// Signal to stop (for multi-threaded scenarios).
    void stop();

   private:
    std::shared_ptr<BPETokenizer> tokenizer_;
    size_t batch_size_;
    size_t seq_len_;
    size_t num_workers_;
    std::vector<TokenId> tokens_;  // all tokens from loaded files
    size_t cursor_ = 0;
    std::atomic<bool> stopped_{false};
};

}  // namespace serius