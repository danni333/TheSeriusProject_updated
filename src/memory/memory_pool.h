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
    void reset();

    size_t capacity() const;
    size_t used() const;
    size_t peak_used() const;
    size_t allocation_count() const;
    void print_stats() const;

    static constexpr size_t ALIGNMENT = 64;

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

    size_t peak_used_{0};
    size_t allocation_count_{0};

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
