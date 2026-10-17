#include "sliding_window.h"

#include <cmath>
#include <vector>

namespace serius {
namespace attention {

Tensor sliding_window_mask(size_t seq_len, size_t window_size)
{
    if (!is_valid_window(seq_len, window_size)) {
        window_size = seq_len;  // fallback to full attention
    }

    Tensor mask({seq_len, seq_len});
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    mask.fill(NEG_INF);

    for (size_t i = 0; i < seq_len; ++i) {
        size_t start = (i >= window_size - 1) ? (i - window_size + 1) : 0;
        for (size_t j = start; j <= i; ++j) {
            mask.at(i, j) = 0.0f;
        }
    }

    return mask;
}

Tensor global_local_mask(size_t seq_len, size_t window_size,
                         const std::vector<size_t>& global_indices)
{
    if (!is_valid_window(seq_len, window_size)) {
        window_size = seq_len;
    }

    Tensor mask({seq_len, seq_len});
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    mask.fill(NEG_INF);

    // Create a set for fast lookup of global indices
    std::vector<bool> is_global(seq_len, false);
    for (size_t idx : global_indices) {
        if (idx < seq_len) {
            is_global[idx] = true;
        }
    }

    for (size_t i = 0; i < seq_len; ++i) {
        bool is_row_global = is_global[i];

        for (size_t j = 0; j < seq_len; ++j) {
            bool is_col_global = is_global[j];

            // Row i can attend to column j if:
            // - Row i is global (attends to everything)
            // - Column j is global (global tokens are visible to all)
            // - Column j is within the sliding window of row i
            bool allowed = is_row_global || is_col_global ||
                           (j >= (i >= window_size - 1 ? i - window_size + 1 : 0) && j <= i);

            if (allowed) {
                mask.at(i, j) = 0.0f;
            }
        }
    }

    return mask;
}

bool is_valid_window(size_t seq_len, size_t window_size)
{
    return window_size > 0 && window_size <= seq_len;
}

}  // namespace attention
}  // namespace serius
