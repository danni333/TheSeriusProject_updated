#pragma once

#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace model {

using core::Tensor;

/**
 * @brief Manages a pool of physical memory blocks for KV caches.
 */
class BlockManager
{
   public:
    BlockManager(size_t num_blocks, size_t block_size, size_t num_heads, size_t head_dim,
                 core::Tensor::DataLocation location = core::Tensor::DataLocation::CPU);
    ~BlockManager() = default;

    // Allocate a block and return its ID
    int allocate();

    // Free a block by ID
    void free(int block_id);

    // Get number of available blocks
    size_t num_free_blocks() const { return free_blocks_.size(); }

    // Access to actual physical memory
    Tensor &get_k_pool() { return k_pool_; }
    Tensor &get_v_pool() { return v_pool_; }

    size_t block_size() const { return block_size_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

   private:
    size_t num_blocks_;
    size_t block_size_;  // tokens per block
    size_t num_heads_;
    size_t head_dim_;

    Tensor k_pool_;  // [num_blocks, num_heads, block_size, head_dim]
    Tensor v_pool_;  // [num_blocks, num_heads, block_size, head_dim]

    std::stack<int> free_blocks_;
};

/**
 * @brief A KV cache implementation using paged memory (non-contiguous blocks).
 */
class PagedKVCache
{
   public:
    PagedKVCache(std::shared_ptr<BlockManager> block_manager);
    ~PagedKVCache();

    /**
     * @brief Update the cache with new tokens.
     * @param key: [num_heads, new_len, head_dim]
     * @param value: [num_heads, new_len, head_dim]
     */
    void update(const Tensor &key, const Tensor &value);

    /**
     * @brief Get the block table for this sequence.
     * @return List of physical block IDs.
     */
    const std::vector<int> &get_block_table() const { return block_table_; }

    /**
     * @brief Gather all blocks into a contiguous tensor.
     * @return [num_heads, current_len, head_dim]
     */
    Tensor gather_keys() const;
    Tensor gather_values() const;

    size_t current_len() const { return current_len_; }
    size_t num_blocks() const { return block_table_.size(); }

    void reset();

   private:
    std::shared_ptr<BlockManager> block_manager_;
    std::vector<int> block_table_;
    size_t current_len_;
};

}  // namespace model
}  // namespace serius
