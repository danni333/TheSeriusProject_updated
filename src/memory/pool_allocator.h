#pragma once

#include <cstddef>
#include <new>
#include <utility>

#include "memory_pool.h"


namespace serius {
namespace memory {

template <typename T>
class PoolAllocator
{
   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::false_type;

    explicit PoolAllocator(MemoryPool &pool) noexcept : pool_(&pool) {}

    template <typename U>
    PoolAllocator(const PoolAllocator<U> &other) noexcept : pool_(other.pool())
    {
    }

    T *allocate(std::size_t n)
    {
        if (n == 0) return nullptr;
        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }
        void *ptr = pool_->allocate(n * sizeof(T));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(ptr);
    }

    void deallocate(T *p, std::size_t n) noexcept
    {
        // Real freeing happens when the pool is reset or destroyed.
        (void)p;
        (void)n;
    }

    MemoryPool *pool() const noexcept { return pool_; }

    template <typename U>
    bool operator==(const PoolAllocator<U> &other) const noexcept
    {
        return pool_ == other.pool();
    }

    template <typename U>
    bool operator!=(const PoolAllocator<U> &other) const noexcept
    {
        return pool_ != other.pool();
    }

   private:
    MemoryPool *pool_;
};

}  // namespace memory
}  // namespace serius
