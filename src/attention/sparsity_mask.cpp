#include "sparsity_mask.h"

#include <cmath>
#include <random>

namespace serius {
namespace attention {

Tensor generate_block_mask(size_t seq_len, size_t block_size,
                          SparsityPattern pattern, size_t window_size)
{
    if (seq_len % block_size != 0) {
        // Adjust block_size to divide seq_len
        block_size = 1;
    }

    size_t num_blocks = seq_len / block_size;
    Tensor block_mask({num_blocks, num_blocks});

    switch (pattern) {
        case SparsityPattern::CAUSAL:
            block_mask = causal_block_mask(num_blocks);
            break;
        case SparsityPattern::SLIDING_WINDOW:
            block_mask = sliding_window_block_mask(num_blocks, window_size / block_size);
            break;
        case SparsityPattern::RANDOM:
            block_mask = random_block_mask(num_blocks, 0.5f);
            break;
        case SparsityPattern::LONGFORMER:
            block_mask = longformer_block_mask(num_blocks, window_size / block_size, {0});
            break;
        case SparsityPattern::BIGBIRD:
            // Combination of random + global + local
            block_mask = causal_block_mask(num_blocks);  // Simplified: use causal
            break;
        default:
            block_mask = causal_block_mask(num_blocks);
            break;
    }

    return block_mask;
}

Tensor causal_block_mask(size_t num_blocks)
{
    Tensor mask({num_blocks, num_blocks});
    mask.fill(0.0f);

    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mask.at({i, j}) = 1.0f;
        }
    }

    return mask;
}

Tensor sliding_window_block_mask(size_t num_blocks, size_t window_blocks)
{
    Tensor mask({num_blocks, num_blocks});
    mask.fill(0.0f);

    if (window_blocks == 0) {
        window_blocks = num_blocks;
    }

    for (size_t i = 0; i < num_blocks; ++i) {
        size_t start = (i >= window_blocks - 1) ? (i - window_blocks + 1) : 0;
        for (size_t j = start; j <= i; ++j) {
            mask.at({i, j}) = 1.0f;
        }
    }

    return mask;
}

Tensor random_block_mask(size_t num_blocks, float sparsity_ratio)
{
    Tensor mask({num_blocks, num_blocks});
    mask.fill(0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < num_blocks; ++i) {
        for (size_t j = 0; j <= i; ++j) {  // Causal
            if (dist(gen) > sparsity_ratio) {
                mask.at({i, j}) = 1.0f;
            }
        }
    }

    return mask;
}

Tensor longformer_block_mask(size_t num_blocks, size_t window_blocks,
                             const std::vector<size_t>& global_blocks)
{
    Tensor mask({num_blocks, num_blocks});
    mask.fill(0.0f);

    std::vector<bool> is_global(num_blocks, false);
    for (size_t idx : global_blocks) {
        if (idx < num_blocks) {
            is_global[idx] = true;
        }
    }

    for (size_t i = 0; i < num_blocks; ++i) {
        bool row_global = is_global[i];

        for (size_t j = 0; j < num_blocks; ++j) {
            bool col_global = is_global[j];

            // Row i can attend to column j if:
            // - Row i is global (attends to everything)
            // - Column j is global (global tokens are visible to all)
            // - Column j is within the sliding window
            bool allowed = row_global || col_global ||
                           (j >= (i >= window_blocks - 1 ? i - window_blocks + 1 : 0) && j <= i);

            if (allowed) {
                mask.at({i, j}) = 1.0f;
            }
        }
    }

    return mask;
}

float calculate_sparsity_ratio(const Tensor& block_mask)
{
    size_t total = block_mask.size();
    size_t active = 0;

    for (size_t i = 0; i < total; ++i) {
        if (block_mask.item(i) > 0.5f) {
            active++;
        }
    }

    return 1.0f - static_cast<float>(active) / static_cast<float>(total);
}

Tensor block_mask_to_full_mask(const Tensor& block_mask, size_t block_size)
{
    size_t num_blocks = block_mask.shape()[0];
    size_t seq_len = num_blocks * block_size;

    Tensor full_mask({seq_len, seq_len});
    full_mask.fill(0.0f);

    for (size_t block_i = 0; block_i < num_blocks; ++block_i) {
        for (size_t block_j = 0; block_j < num_blocks; ++block_j) {
            if (block_mask.at({block_i, block_j}) > 0.5f) {
                // Fill the corresponding block in the full mask
                for (size_t i = 0; i < block_size; ++i) {
                    for (size_t j = 0; j < block_size; ++j) {
                        size_t full_i = block_i * block_size + i;
                        size_t full_j = block_j * block_size + j;
                        if (full_i < seq_len && full_j < seq_len) {
                            full_mask.at({full_i, full_j}) = 0.0f;  // 0 = allowed (attention)
                        }
                    }
                }
            } else {
                // Fill with -inf (blocked)
                for (size_t i = 0; i < block_size; ++i) {
                    for (size_t j = 0; j < block_size; ++j) {
                        size_t full_i = block_i * block_size + i;
                        size_t full_j = block_j * block_size + j;
                        if (full_i < seq_len && full_j < seq_len) {
                            full_mask.at({full_i, full_j}) = -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }
        }
    }

    return full_mask;
}

}  // namespace attention
}  // namespace serius
