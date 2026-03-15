#include "plugin_manager.h"

#include <algorithm>
#include <spdlog/spdlog.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace serius::hub {

PluginManager::PluginManager() {}

PluginManager::~PluginManager()
{
    // Unload all plugins
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    for (auto &[plugin_id, instance] : loaded_plugins_) {
        if (instance->plugin) {
            instance->plugin->shutdown();
        }
        if (instance->library_handle) {
            unload_library(instance->library_handle);
        }
        stop_monitoring(plugin_id);
    }
}

bool PluginManager::load_plugin(const std::string &plugin_path, const PluginContext &context)
{
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
#ifdef _WIN32
    void *handle = (HMODULE)LoadLibraryA(plugin_path.c_str());
#else
    void *handle = dlopen(plugin_path.c_str(), RTLD_LAZY);
#endif
    if (!handle) {
        spdlog::error("Failed to load plugin library: {}", plugin_path);
        return false;
    }

    // Get plugin factory function
    typedef IPlugin *(*create_plugin_func_t)();
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
#ifdef _WIN32
    instance->library_handle = (HMODULE)handle;
#else
    instance->library_handle = handle;
#endif
    instance->should_monitor = true;

    loaded_plugins_[metadata.id] = std::move(instance);

    // Start resource monitoring
    start_monitoring(metadata.id);

    spdlog::info("Plugin loaded successfully: {}", metadata.id);
    return true;
}

bool PluginManager::create_sandbox(const PluginContext &context)
{
    // For now, implement basic sandboxing
    // In production, this would use containers, chroot, or namespaces

    SandboxConfig config;
    config.allowed_paths = context.allowed_paths;
    config.environment_vars = context.environment_vars;
    config.max_memory_mb = 512;     // 512MB limit
    config.max_cpu_percent = 50.0;  // 50% CPU limit
    config.max_file_descriptors = 100;
    config.network_access = false;  // No network access by default

    return setup_resource_limits(context.plugin_id, config);
}

bool PluginManager::setup_resource_limits(const std::string &plugin_id, const SandboxConfig &config)
{
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

void PluginManager::start_monitoring(const std::string &plugin_id)
{
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return;

    auto &instance = it->second;
    instance->monitoring_thread = std::thread([this, plugin_id]() { monitor_plugin_resources(plugin_id); });
}

void PluginManager::monitor_plugin_resources(const std::string &plugin_id)
{
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return;

    auto &instance = it->second;

    while (instance->should_monitor) {
        // Monitor memory usage
        size_t memory_usage = get_process_memory_usage();
        instance->memory_usage = memory_usage;

        // Monitor CPU usage
        double cpu_usage = get_process_cpu_usage();
        instance->cpu_usage = cpu_usage;

        // Check if plugin exceeded limits
        if (memory_usage > 512 * 1024 * 1024) {  // 512MB
            spdlog::warn("Plugin {} exceeded memory limit: {} MB", plugin_id, memory_usage / (1024 * 1024));
        }

        if (cpu_usage > 50.0) {  // 50%
            spdlog::warn("Plugin {} exceeded CPU limit: {}%", plugin_id, cpu_usage);
        }

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

size_t PluginManager::get_process_memory_usage()
{
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

double PluginManager::get_process_cpu_usage()
{
    static auto last_process_time = std::chrono::steady_clock::now();
    static double last_cpu_time = 0.0;
    
    auto current_time = std::chrono::steady_clock::now();
    double cpu_time = 0.0;

#ifdef _WIN32
    FILETIME createTime, exitTime, kernelTime, userTime;
    if (GetProcessTimes(GetCurrentProcess(), &createTime, &exitTime, &kernelTime, &userTime)) {
        ULARGE_INTEGER kernel, user;
        kernel.LowPart = kernelTime.dwLowDateTime;
        kernel.HighPart = kernelTime.dwHighDateTime;
        user.LowPart = userTime.dwLowDateTime;
        user.HighPart = userTime.dwHighDateTime;
        // Convert 100ns units to microseconds
        cpu_time = static_cast<double>(kernel.QuadPart + user.QuadPart) / 10.0;
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        cpu_time = static_cast<double>(usage.ru_utime.tv_sec * 1000000 + usage.ru_utime.tv_usec +
                                      usage.ru_stime.tv_sec * 1000000 + usage.ru_stime.tv_usec);
    }
#endif

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_process_time).count();
    double usage_pct = 0.0;
    if (duration > 0) {
        usage_pct = ((cpu_time - last_cpu_time) / duration) * 100.0;
    }

    last_process_time = current_time;
    last_cpu_time = cpu_time;

    return (std::max)(0.0, (std::min)(100.0, usage_pct));
}

bool PluginManager::unload_plugin(const std::string &plugin_id)
{
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return false;

    auto &instance = it->second;
    if (instance->plugin) {
        instance->plugin->shutdown();
    }
    if (instance->library_handle) {
        unload_library(instance->library_handle);
    }
    stop_monitoring(plugin_id);
    loaded_plugins_.erase(it);
    return true;
}

bool PluginManager::reload_plugin(const std::string &plugin_id)
{
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return false;

    std::string path = it->second->metadata.entry_point;
    PluginContext context = it->second->context;

    if (!unload_plugin(plugin_id)) return false;
    return load_plugin(path, context);
}

std::vector<PluginMetadata> PluginManager::discover_plugins(const std::string &search_path)
{
    std::vector<PluginMetadata> plugins;
    if (!std::filesystem::exists(search_path)) return plugins;

    for (const auto &entry : std::filesystem::directory_iterator(search_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
#ifdef _WIN32
            if (ext == ".dll")
#else
            if (ext == ".so")
#endif
            {
                PluginFactory factory;
                if (factory.validate_plugin(entry.path().string())) {
                    plugins.push_back(factory.extract_metadata(entry.path().string()));
                }
            }
        }
    }
    return plugins;
}

PluginMetadata PluginManager::get_plugin_metadata(const std::string &plugin_id)
{
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    auto it = loaded_plugins_.find(plugin_id);
    if (it != loaded_plugins_.end()) {
        return it->second->metadata;
    }
    return PluginMetadata{};
}

std::vector<std::string> PluginManager::list_loaded_plugins()
{
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    std::vector<std::string> plugin_ids;
    for (const auto &[id, _] : loaded_plugins_) {
        plugin_ids.push_back(id);
    }
    return plugin_ids;
}

nlohmann::json PluginManager::execute_plugin(const std::string &plugin_id, const nlohmann::json &input,
                                             const PluginContext &context)
{
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end() || !it->second->plugin) {
        return {{"error", "Plugin not loaded"}};
    }

    if (!check_plugin_permissions(plugin_id, "execute")) {
        return {{"error", "Permission denied"}};
    }

    try {
        return it->second->plugin->process(input);
    }
    catch (const std::exception &e) {
        return {{"error", e.what()}};
    }
}

bool PluginManager::destroy_sandbox(const std::string &plugin_id)
{
    spdlog::info("Destroying sandbox for plugin: {}", plugin_id);
    // In a real implementation, this would clean up cgroups, namespaces, etc.
    return true;
}

bool PluginManager::check_plugin_permissions(const std::string &plugin_id, const std::string &operation)
{
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return false;

    // Basic permission check: allowed_paths validation for IO operations
    if (operation == "io_read" || operation == "io_write") {
        return !it->second->context.allowed_paths.empty();
    }

    // Default to allow for other operations for now
    return true;
}

nlohmann::json PluginManager::get_plugin_metrics(const std::string &plugin_id)
{
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    auto it = loaded_plugins_.find(plugin_id);
    if (it == loaded_plugins_.end()) return {};

    return {{"memory_usage", (size_t)it->second->memory_usage}, {"cpu_usage", (double)it->second->cpu_usage}};
}

nlohmann::json PluginManager::get_system_metrics()
{
    return collect_system_metrics();
}

void PluginManager::stop_monitoring(const std::string &plugin_id)
{
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    auto it = loaded_plugins_.find(plugin_id);
    if (it != loaded_plugins_.end()) {
        it->second->should_monitor = false;
        if (it->second->monitoring_thread.joinable()) {
            it->second->monitoring_thread.detach();  // Simple for now
        }
    }
}

nlohmann::json PluginManager::collect_system_metrics()
{
    return {{"total_memory_usage", get_process_memory_usage()}, {"total_cpu_usage", get_process_cpu_usage()}};
}

// PluginFactory Implementation

std::unique_ptr<IPlugin> PluginFactory::create_plugin(const PluginMetadata &metadata)
{
    std::string path = metadata.entry_point;
    if (path.empty()) return nullptr;

#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path.c_str());
#else
    void *handle = dlopen(path.c_str(), RTLD_LAZY);
#endif

    if (!handle) return nullptr;

    typedef IPlugin *(*create_plugin_func_t)();
#ifdef _WIN32
    create_plugin_func_t create_plugin_func = (create_plugin_func_t)GetProcAddress(handle, "create_plugin");
#else
    create_plugin_func_t create_plugin_func = (create_plugin_func_t)dlsym(handle, "create_plugin");
#endif

    if (!create_plugin_func) {
#ifdef _WIN32
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        return nullptr;
    }

    return std::unique_ptr<IPlugin>(create_plugin_func());
}

bool PluginFactory::validate_plugin(const std::string &plugin_path)
{
    return std::filesystem::exists(plugin_path);
}

PluginMetadata PluginFactory::extract_metadata(const std::string &plugin_path)
{
    PluginMetadata metadata;
    metadata.entry_point = plugin_path;
    metadata.id = std::filesystem::path(plugin_path).stem().string();
    metadata.name = metadata.id;
    metadata.version = "1.0.0";
    return metadata;
}

void *PluginManager::load_library(const std::string &path)
{
#ifdef _WIN32
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_LAZY);
#endif
}

void PluginManager::unload_library(void *handle)
{
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void *PluginManager::get_symbol(void *handle, const std::string &symbol)
{
#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(handle), symbol.c_str());
#else
    return dlsym(handle, symbol.c_str());
#endif
}

}  // namespace serius::hub
