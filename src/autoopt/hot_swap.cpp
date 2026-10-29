#include "hot_swap.h"

#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace serius {
namespace autoopt {

void* HotSwapManager::load_library(const std::string &path) const
{
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

KernelFn HotSwapManager::get_function_symbol(void* library_handle, const std::string &symbol_name) const
{
    if (!library_handle) {
        return nullptr;
    }

#ifdef _WIN32
    FARPROC proc = GetProcAddress(static_cast<HMODULE>(library_handle), symbol_name.c_str());
    if (!proc) {
        return nullptr;
    }
    // Windows expects C-style function pointer
    using KernelRaw = core::Tensor(*)(const core::Tensor&);
    return reinterpret_cast<KernelRaw>(proc);
#else
    void* sym = dlsym(library_handle, symbol_name.c_str());
    if (!sym) {
        return nullptr;
    }
    using KernelRaw = core::Tensor(*)(const core::Tensor&);
    return reinterpret_cast<KernelRaw>(sym);
#endif
}

void HotSwapManager::unload_library(void* handle) const
{
    if (!handle) return;

#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void HotSwapManager::register_kernel(const std::string &name, KernelFn fn)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fn) return;
    registry_[name] = {std::move(fn)};
}

core::Tensor HotSwapManager::execute(const std::string &name, const core::Tensor &input) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(name);
    if (it == registry_.end() || it->second.empty()) {
        throw std::runtime_error("HotSwapManager: Kernel not found: " + name);
    }
    return it->second.back()(input);
}

bool HotSwapManager::hot_swap(const std::string &name, KernelFn new_fn)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!new_fn) return false;

    auto it = registry_.find(name);
    if (it == registry_.end()) {
        registry_[name] = {std::move(new_fn)};
    } else {
        it->second.push_back(std::move(new_fn));
    }
    return true;
}

bool HotSwapManager::hot_swap_from_library(const std::string &name, const std::string &library_path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Load the library
    void* lib_handle = load_library(library_path);
    if (!lib_handle) {
        return false;
    }

    // Get the kernel function (expecting symbol 'kernel_entry')
    KernelFn kernel_fn = get_function_symbol(lib_handle, "kernel_entry");
    if (!kernel_fn) {
        unload_library(lib_handle);
        return false;
    }

    // Register the new kernel
    auto it = registry_.find(name);
    if (it == registry_.end()) {
        registry_[name] = {std::move(kernel_fn)};
    } else {
        it->second.push_back(std::move(kernel_fn));
    }

    // Track the loaded library for later cleanup
    loaded_libraries_[name + "_v" + std::to_string(registry_[name].size())] = lib_handle;

    return true;
}

bool HotSwapManager::rollback(const std::string &name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(name);
    if (it == registry_.end() || it->second.size() <= 1) {
        return false;  // Cannot rollback initial baseline version
    }

    it->second.pop_back();
    return true;
}

size_t HotSwapManager::get_version(const std::string &name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(name);
    if (it == registry_.end()) return 0;
    return it->second.size();
}

bool HotSwapManager::has_kernel(const std::string &name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(name);
    return it != registry_.end() && !it->second.empty();
}

}  // namespace autoopt
}  // namespace serius
