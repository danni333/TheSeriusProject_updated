#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace serius {
namespace core {
namespace memory {

class MemoryPool
{
   public:
    static constexpr size_t kDefaultPoolSize = 128ULL * 1024ULL * 1024ULL;
    explicit MemoryPool(size_t initial_size = kDefaultPoolSize);
    ~MemoryPool();

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    void *allocate(size_t size);
    /// Return a previously allocated block to the pool's free list so it can
    /// be reused by a later allocate() of equal-or-smaller size. Arena
    /// semantics still apply: reset() and ~ScopedArena() rewind the bump
    /// pointer and invalidate the free list.
    void deallocate(void *ptr, size_t size);
    void reset();

    size_t capacity() const;
    size_t used() const;
    size_t peak_used() const;
    size_t allocation_count() const;
    void print_stats() const;

    static constexpr size_t ALIGNMENT = 64;

    /// Get a thread-local memory pool (64MB per thread)
    static MemoryPool &get_thread_local()
    {
        thread_local MemoryPool tl_pool(64ULL * 1024ULL * 1024ULL);
        return tl_pool;
    }

   private:
    struct Block
    {
        uint8_t *data;
        size_t size;

        explicit Block(size_t alloc_size);
        ~Block();
    };

    std::vector<std::unique_ptr<Block>> blocks_;
    size_t current_block_idx_;
    size_t current_offset_;

    struct FreeBlock
    {
        void *ptr;
        size_t size;  // aligned size
    };
    std::vector<FreeBlock> free_list_;
    size_t free_list_bytes_ = 0;

    size_t peak_used_{0};
    size_t allocation_count_{0};
    size_t deallocation_count_{0};

    static size_t align_up(size_t val, size_t alignment);
    void grow(size_t min_size);

    friend class ScopedArena;
};

class ScopedArena
{
   public:
    explicit ScopedArena(MemoryPool &pool);
    ~ScopedArena();

    ScopedArena(const ScopedArena &) = delete;
    ScopedArena &operator=(const ScopedArena &) = delete;

   private:
    MemoryPool &pool_;
    size_t saved_block_idx_;
    size_t saved_offset_;
};

}  // namespace memory
}  // namespace core
}  // namespace serius
