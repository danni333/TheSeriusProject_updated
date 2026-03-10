#include "plugin_manager.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace serius::hub {

PluginManager::PluginManager() {}

PluginManager::~PluginManager() {
    // Unload all plugins
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto& [plugin_id, instance] : loaded_plugins_) {
        if (instance->plugin) {
            instance->plugin->shutdown();
        }
        if (instance->library_handle) {
            unload_library(instance->library_handle);
        }
        stop_monitoring(plugin_id);
    }
}

bool PluginManager::load_plugin(const std::string& plugin_path, const PluginContext& context) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    
    // Validate plugin
    PluginFactory factory;
    if (!factory.validate_plugin(plugin_path)) {
        spdlog::error("Plugin validation failed: {}", plugin_path);
        return false;
    }
    
    // Extract metadata
    PluginMetadata metadata = factory.extract_metadata(plugin_path);
    if (metadata.id.empty()) {
        spdlog::error("Failed to extract metadata from plugin: {}", plugin_path);
        return false;
    }
    
    // Check if already loaded
    if (loaded_plugins_.find(metadata.id) != loaded_plugins_.end()) {
        spdlog::warn("Plugin already loaded: {}", metadata.id);
        return false;
    }
    
    // Create sandbox
    if (!create_sandbox(context)) {
        spdlog::error("Failed to create sandbox for plugin: {}", metadata.id);
        return false;
    }
    
    // Load library
    void* handle = load_library(plugin_path);
    if (!handle) {
        spdlog::error("Failed to load plugin library: {}", plugin_path);
        return false;
    }
    
    // Get plugin factory function
    typedef IPlugin* (*create_plugin_func_t)();
    create_plugin_func_t create_plugin_func = 
        reinterpret_cast<create_plugin_func_t>(get_symbol(handle, "create_plugin"));
    
    if (!create_plugin_func) {
        unload_library(handle);
        spdlog::error("Plugin missing create_plugin function: {}", plugin_path);
        return false;
    }
    
    // Create plugin instance
    std::unique_ptr<IPlugin> plugin(create_plugin_func());
    if (!plugin) {
        unload_library(handle);
        spdlog::error("Failed to create plugin instance: {}", metadata.id);
        return false;
    }
    
    // Initialize plugin
    if (!plugin->initialize(context)) {
        unload_library(handle);
        spdlog::error("Plugin initialization failed: {}", metadata.id);
        return false;
    }
    
    // Store plugin instance
    auto instance = std::make_unique<PluginInstance>();
    instance->plugin = std::move(plugin);
    instance->metadata = metadata;
    instance->state = PluginState::LOADED;
    instance->context = context;
    instance->library_handle = handle;
    instance->should_monitor = true;
    
    loaded_plugins_[metadata.id] = std::move(instance);
    
    // Start resource monitoring
    start_monitoring(metadata.id);
    
    spdlog::info("Plugin loaded successfully: {}", metadata.id);
    return true;
}

bool PluginManager::create_sandbox(const PluginContext& context) {
    // For now, implement basic sandboxing
    // In production, this would use containers, chroot, or namespaces
    
    SandboxConfig config;
    config.allowed_paths = context.allowed_paths;
    config.environment_vars = context.environment_vars;
    config.max_memory_mb = 512;  // 512MB limit
    config.max_cpu_percent = 50.0;  // 50% CPU limit
    config.max_file_descriptors = 100;
    config.network_access = false;  // No network access by default
    
    return setup_resource_limits(context.plugin_id, config);
}

bool PluginManager::setup_resource_limits(const std::string& plugin_id, const SandboxConfig& config) {
#ifndef _WIN32
    // Set memory limit
    struct rlimit mem_limit;
    mem_limit.rlim_cur = config.max_memory_mb * 1024 * 1024;  // Convert to bytes
    mem_limit.rlim_max = mem_limit.rlim_cur;
    if (setrlimit(RLIMIT_AS, &mem_limit) != 0) {
        spdlog::warn("Failed to set memory limit for plugin: {}", plugin_id);
    }
    
    // Set file descriptor limit
    struct rlimit fd_limit;
    fd_limit.rlim_cur = config.max_file_descriptors;
    fd_limit.rlim_max = config.max_file_descriptors;
    if (setrlimit(RLIMIT_NOFILE, &fd_limit) != 0) {
        spdlog::warn("Failed to set file descriptor limit for plugin: {}", plugin_id);
    }
#endif
    
    return true;
}

void PluginManager::start_monitoring(const std::string& plugin_id) {
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return;
    
    auto& instance = it->second;
    instance->monitoring_thread = std::thread([this, plugin_id]() {
        monitor_plugin_resources(plugin_id);
    });
}

void PluginManager::monitor_plugin_resources(const std::string& plugin_id) {
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return;
    
    auto& instance = it->second;
    
    while (instance->should_monitor) {
        // Monitor memory usage
        size_t memory_usage = get_process_memory_usage();
        instance->memory_usage = memory_usage;
        
        // Monitor CPU usage
        double cpu_usage = get_process_cpu_usage();
        instance->cpu_usage = cpu_usage;
        
        // Check if plugin exceeded limits
        if (memory_usage > 512 * 1024 * 1024) {  // 512MB
            spdlog::warn("Plugin {} exceeded memory limit: {} MB", 
                        plugin_id, memory_usage / (1024 * 1024));
        }
        
        if (cpu_usage > 50.0) {  // 50%
            spdlog::warn("Plugin {} exceeded CPU limit: {}%", plugin_id, cpu_usage);
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

size_t PluginManager::get_process_memory_usage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss * 1024;  // Convert KB to bytes
    }
#endif
    return 0;
}

double PluginManager::get_process_cpu_usage() {
    // Simplified CPU usage calculation
    // In production, this would use more accurate timing
    static auto last_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - last_time).count();
    
    last_time = current_time;
    
    // Return a dummy value for now
    return duration > 0 ? 25.0 : 0.0;
}

void* PluginManager::load_library(const std::string& path) {
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

void PluginManager::unload_library(void* handle) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void* PluginManager::get_symbol(void* handle, const std::string& symbol) {
#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(handle), symbol.c_str());
#else
    return dlsym(handle, symbol.c_str());
#endif
}

} // namespace serius::hub
