#include "paged_kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace serius {
namespace model {

BlockManager::BlockManager(size_t num_blocks, size_t block_size, size_t num_heads, size_t head_dim,
                           core::Tensor::DataLocation location)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_heads_(num_heads),
      head_dim_(head_dim),
      k_pool_({num_blocks, num_heads, block_size, head_dim}),
      v_pool_({num_blocks, num_heads, block_size, head_dim})
{
    for (int i = (int)num_blocks - 1; i >= 0; --i) {
        free_blocks_.push(i);
    }

    if (location == core::Tensor::DataLocation::GPU) {
        k_pool_.syncToDevice();
        v_pool_.syncToDevice();
    }
}

int BlockManager::allocate()
{
    if (free_blocks_.empty()) {
        throw std::runtime_error("BlockManager execution error: No free blocks available in pool.");
    }
    int block_id = free_blocks_.top();
    free_blocks_.pop();
    return block_id;
}

void BlockManager::free(int block_id)
{
    free_blocks_.push(block_id);
}

PagedKVCache::PagedKVCache(std::shared_ptr<BlockManager> block_manager)
    : block_manager_(std::move(block_manager)), current_len_(0)
{
}

PagedKVCache::~PagedKVCache()
{
    reset();
}

void PagedKVCache::reset()
{
    for (int block_id : block_table_) {
        block_manager_->free(block_id);
    }
    block_table_.clear();
    current_len_ = 0;
}

void PagedKVCache::update(const Tensor &key, const Tensor &value)
{
    // key, value shape: [num_heads, new_len, head_dim] or [1, num_heads, new_len, head_dim]
    size_t num_heads = 0;
    size_t new_len = 0;
    size_t head_dim = 0;
    size_t start_dim = 0;

    if (key.ndim() == 4 && key.shape()[0] == 1) {
        start_dim = 1;
    } else if (key.ndim() == 3) {
        start_dim = 0;
    } else {
        throw std::runtime_error("PagedKVCache update error: Input must be 3D or 4D with batch=1.");
    }

    num_heads = key.shape()[start_dim];
    new_len = key.shape()[start_dim + 1];
    head_dim = key.shape()[start_dim + 2];

    if (num_heads != block_manager_->num_heads() || head_dim != block_manager_->head_dim()) {
        throw std::runtime_error("PagedKVCache update error: Input dimension mismatch with BlockManager.");
    }

    size_t block_size = block_manager_->block_size();

    for (size_t i = 0; i < new_len; ++i) {
        size_t logical_block_idx = current_len_ / block_size;
        size_t block_offset = current_len_ % block_size;

        if (logical_block_idx >= block_table_.size()) {
            block_table_.push_back(block_manager_->allocate());
        }

        int physical_block_id = block_table_[logical_block_idx];

        for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (size_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
                float k_val, v_val;
                if (start_dim == 1) {
                    k_val = key.item({0, head_idx, i, dim_idx});
                    v_val = value.item({0, head_idx, i, dim_idx});
                } else {
                    k_val = key.item({head_idx, i, dim_idx});
                    v_val = value.item({head_idx, i, dim_idx});
                }

                block_manager_->get_k_pool().set_item({(size_t)physical_block_id, head_idx, block_offset, dim_idx},
                                                      k_val);
                block_manager_->get_v_pool().set_item({(size_t)physical_block_id, head_idx, block_offset, dim_idx},
                                                      v_val);
            }
        }
        current_len_++;
    }
}

Tensor PagedKVCache::gather_keys() const
{
    if (current_len_ == 0) {
        return Tensor({block_manager_->num_heads(), 0, block_manager_->head_dim()});
    }

    Tensor gathered_keys({block_manager_->num_heads(), current_len_, block_manager_->head_dim()});
    size_t num_blocks_in_cache = block_manager_->num_heads();
    size_t dim_size = block_manager_->head_dim();
    size_t blk_size = block_manager_->block_size();

    for (size_t iIdx = 0; iIdx < current_len_; ++iIdx) {
        size_t block_idx = iIdx / blk_size;
        size_t offset_val = iIdx % blk_size;
        int physical_id = block_table_[block_idx];

        for (size_t head_idx = 0; head_idx < num_blocks_in_cache; ++head_idx) {
            for (size_t head_dim_idx = 0; head_dim_idx < dim_size; ++head_dim_idx) {
                float k_val =
                    block_manager_->get_k_pool().item({(size_t)physical_id, head_idx, offset_val, head_dim_idx});
                gathered_keys.set_item({head_idx, iIdx, head_dim_idx}, k_val);
            }
        }
    }
    return gathered_keys;
}

Tensor PagedKVCache::gather_values() const
{
    if (current_len_ == 0) {
        return Tensor({block_manager_->num_heads(), 0, block_manager_->head_dim()});
    }

    Tensor gathered_values({block_manager_->num_heads(), current_len_, block_manager_->head_dim()});
    size_t num_blocks_in_cache = block_manager_->num_heads();
    size_t dim_size = block_manager_->head_dim();
    size_t blk_size = block_manager_->block_size();

    for (size_t iIdx = 0; iIdx < current_len_; ++iIdx) {
        size_t block_idx = iIdx / blk_size;
        size_t offset_val = iIdx % blk_size;
        int physical_id = block_table_[block_idx];

        for (size_t head_idx = 0; head_idx < num_blocks_in_cache; ++head_idx) {
            for (size_t head_dim_idx = 0; head_dim_idx < dim_size; ++head_dim_idx) {
                float v_val =
                    block_manager_->get_v_pool().item({(size_t)physical_id, head_idx, offset_val, head_dim_idx});
                gathered_values.set_item({head_idx, iIdx, head_dim_idx}, v_val);
            }
        }
    }
    return gathered_values;
}

}  // namespace model
}  // namespace serius
