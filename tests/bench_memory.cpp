#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../src/memory/memory_pool.h"


using namespace serius;
using namespace std::chrono;

void run_memory_benchmark(size_t alloc_count, size_t alloc_size)
{
    std::cout << "Allocating " << alloc_count << " blocks of " << alloc_size << " bytes\n";

    std::vector<void *> raw_pointers(alloc_count);

    // --- Standard Malloc Benchmark ---
    auto start_malloc = high_resolution_clock::now();
    for (size_t i = 0; i < alloc_count; ++i) {
        raw_pointers[i] = std::malloc(alloc_size);
    }
    // Free
    for (size_t i = 0; i < alloc_count; ++i) {
        std::free(raw_pointers[i]);
    }
    auto end_malloc = high_resolution_clock::now();
    duration<double, std::milli> time_malloc = end_malloc - start_malloc;

    // --- Memory Pool Benchmark ---
    memory::MemoryPool pool(alloc_count * alloc_size + 1024 * 1024);  // Large enough to prevent resize

    auto start_pool = high_resolution_clock::now();
    for (size_t i = 0; i < alloc_count; ++i) {
        raw_pointers[i] = pool.allocate(alloc_size);
    }
    pool.reset();  // Instant free
    auto end_pool = high_resolution_clock::now();
    duration<double, std::milli> time_pool = end_pool - start_pool;

    // --- Results ---
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  std::malloc/free : " << time_malloc.count() << " ms\n";
    std::cout << "  MemoryPool       : " << time_pool.count() << " ms\n";
    std::cout << "  Speedup          : " << time_malloc.count() / time_pool.count() << "x\n";
    std::cout << std::string(45, '-') << "\n";
}

int main()
{
    std::cout << "--- MEMORY ALLOCATION BENCHMARK ---\n\n";

    // Test small allocations (High overhead for malloc)
    run_memory_benchmark(1000000, 64);

    // Test medium allocations
    run_memory_benchmark(100000, 1024);

    // Test large allocations
    run_memory_benchmark(1000, 1024 * 1024);

    return 0;
}
