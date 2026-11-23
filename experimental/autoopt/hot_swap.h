#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../tensor/tensor.h"
#include "sandbox.h"

namespace serius {
namespace autoopt {

/// Runtime hot-swapping and rollback manager for compute kernels.
/// Enables seamless zero-downtime kernel replacement with instant rollback capability.
class HotSwapManager
{
   public:
    HotSwapManager() = default;

    /// Register the initial (baseline) version of a named kernel.
    void register_kernel(const std::string &name, KernelFn fn);

    /// Execute the currently active version of a kernel.
    core::Tensor execute(const std::string &name, const core::Tensor &input) const;

    /// Hot-swap the active kernel with an optimized candidate.
    bool hot_swap(const std::string &name, KernelFn new_fn);

    /// Hot-swap from a compiled shared library (.dll/.so)
    /// Loads the library and extracts the kernel function with symbol 'kernel_entry'
    bool hot_swap_from_library(const std::string &name, const std::string &library_path);

    /// Roll back to the preceding version of the kernel.
    bool rollback(const std::string &name);

    /// Get the current active version number of a kernel (1-indexed).
    size_t get_version(const std::string &name) const;

    /// Check if a kernel is registered.
    bool has_kernel(const std::string &name) const;

   private:
    mutable std::mutex mutex_;
    std::map<std::string, std::vector<KernelFn>> registry_;
    std::map<std::string, void*> loaded_libraries_;  // Track loaded libraries for cleanup

    /// Load shared library and return handle
    void* load_library(const std::string &path) const;

    /// Get function symbol from loaded library
    KernelFn get_function_symbol(void* library_handle, const std::string &symbol_name) const;

    /// Unload library
    void unload_library(void* handle) const;
};

}  // namespace autoopt
}  // namespace serius
