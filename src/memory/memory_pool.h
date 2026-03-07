#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


namespace serius {
namespace memory {

class MemoryPool
{
   public:
    explicit MemoryPool(size_t initial_size = 1024 * 1024 * 128);  // Default 128MB
    ~MemoryPool();

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    void *allocate(size_t size);
    void reset();

    size_t capacity() const;
    size_t used() const;

    static constexpr size_t ALIGNMENT = 64;

   private:
    struct Block
    {
        uint8_t *data;
        size_t size;

        explicit Block(size_t s);
        ~Block();
    };

    std::vector<std::unique_ptr<Block>> blocks_;
    size_t current_block_idx_;
    size_t current_offset_;

    static size_t align_up(size_t x, size_t alignment);
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
}  // namespace serius
