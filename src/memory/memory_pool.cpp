#include "memory_pool.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace serius {
namespace core {
namespace memory {

// --- Block ---

MemoryPool::Block::Block(size_t alloc_size) : size(alloc_size)
{
#ifdef _MSC_VER
    data = static_cast<uint8_t *>(_aligned_malloc(size, MemoryPool::ALIGNMENT));

    if (data == nullptr) {
        throw std::bad_alloc();
    }
#else
    if (posix_memalign(reinterpret_cast<void **>(&data), MemoryPool::ALIGNMENT, size) != 0) {
        throw std::bad_alloc();
    }
#endif
}

MemoryPool::Block::~Block()
{
#ifdef _MSC_VER
    _aligned_free(data);
#else
    free(data);
#endif
}

// --- MemoryPool ---

MemoryPool::MemoryPool(size_t initial_size) : current_block_idx_(0), current_offset_(0)
{
    blocks_.push_back(std::make_unique<Block>(initial_size));
}

MemoryPool::~MemoryPool() = default;

size_t MemoryPool::align_up(size_t val, size_t alignment)
{
    return (val + alignment - 1) & ~(alignment - 1);
}

void *MemoryPool::allocate(size_t size)
{
    if (size == 0) {
        return nullptr;
    }

    size_t aligned_size = align_up(size, ALIGNMENT);

    allocation_count_++;

    // Reuse a freed block first (first-fit, split the remainder when it is
    // large enough to hold another aligned allocation).
    for (size_t i = 0; i < free_list_.size(); ++i) {
        FreeBlock &fb = free_list_[i];
        if (fb.size < aligned_size) continue;
        void *ptr = fb.ptr;
        size_t remainder = fb.size - aligned_size;
        free_list_bytes_ -= fb.size;
        if (remainder >= ALIGNMENT) {
            fb.ptr = static_cast<uint8_t *>(fb.ptr) + aligned_size;
            fb.size = remainder;
            free_list_bytes_ += remainder;
        } else {
            free_list_.erase(free_list_.begin() + static_cast<std::ptrdiff_t>(i));
        }
        peak_used_ = std::max(peak_used_, used());
        return ptr;
    }

    if (current_offset_ + aligned_size <= blocks_[current_block_idx_]->size) {
        void *ptr = blocks_[current_block_idx_]->data + current_offset_;
        current_offset_ += aligned_size;
        peak_used_ = std::max(peak_used_, used());
        return ptr;
    }

    // Need more space
    if (current_block_idx_ + 1 < blocks_.size()) {
        if (blocks_[current_block_idx_ + 1]->size >= aligned_size) {
            current_block_idx_++;
            current_offset_ = aligned_size;
            peak_used_ = std::max(peak_used_, used());
            return blocks_[current_block_idx_]->data;
        }
    }

    // Grow with at least the required size or double the last block size
    size_t last_size = blocks_.back()->size;
    size_t new_size = std::max(last_size * 2, align_up(size, ALIGNMENT));
    grow(new_size);

    void *ptr = blocks_[current_block_idx_]->data;
    current_offset_ = aligned_size;
    peak_used_ = std::max(peak_used_, used());
    return ptr;
}

void MemoryPool::grow(size_t min_size)
{
    blocks_.push_back(std::make_unique<Block>(min_size));
    current_block_idx_ = blocks_.size() - 1;
    current_offset_ = 0;
}

void MemoryPool::deallocate(void *ptr, size_t size)
{
    if (ptr == nullptr || size == 0) {
        return;
    }
    size_t aligned_size = align_up(size, ALIGNMENT);
    free_list_.push_back(FreeBlock{ptr, aligned_size});
    free_list_bytes_ += aligned_size;
    deallocation_count_++;
}

void MemoryPool::reset()
{
    current_block_idx_ = 0;
    current_offset_ = 0;
    free_list_.clear();
    free_list_bytes_ = 0;
}

size_t MemoryPool::capacity() const
{
    size_t total = 0;
    for (const auto &block : blocks_) {
        total += block->size;
    }
    return total;
}

size_t MemoryPool::used() const
{
    size_t total = 0;
    for (size_t i = 0; i < current_block_idx_; ++i) {
        total += blocks_[i]->size;
    }
    total += current_offset_;
    // Subtract bytes sitting in the free list (they were bump-allocated but
    // are available for reuse).
    total = total > free_list_bytes_ ? total - free_list_bytes_ : 0;
    return total;
}

size_t MemoryPool::peak_used() const
{
    return peak_used_;
}

size_t MemoryPool::allocation_count() const
{
    return allocation_count_;
}

void MemoryPool::print_stats() const
{
    constexpr double kMB = 1024.0 * 1024.0;
    std::cout << "[MemoryPool Stats]\n"
              << "  Capacity:         " << static_cast<double>(capacity()) / kMB << " MB\n"
              << "  Currently Used:   " << static_cast<double>(used()) / kMB << " MB\n"
              << "  Peak Used:        " << static_cast<double>(peak_used_) / kMB << " MB\n"
              << "  Allocation Count: " << allocation_count_ << "\n"
              << "  Blocks Allocated: " << blocks_.size() << "\n";
}

// --- ScopedArena ---

ScopedArena::ScopedArena(MemoryPool &pool)
    : pool_(pool), saved_block_idx_(pool.current_block_idx_), saved_offset_(pool.current_offset_)
{
}

ScopedArena::~ScopedArena()
{
    pool_.current_block_idx_ = saved_block_idx_;
    pool_.current_offset_ = saved_offset_;
    // Rewinding the bump pointer invalidates any free-list entries that
    // point into the rewound region — drop the whole list to be safe.
    pool_.free_list_.clear();
    pool_.free_list_bytes_ = 0;
}

}  // namespace memory
}  // namespace core
}  // namespace serius
