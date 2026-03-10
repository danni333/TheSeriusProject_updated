#pragma once

#include "plugin_interface.h"
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace serius::hub {

class PluginManager : public IPluginManager {
public:
    PluginManager();
    ~PluginManager();

    // Plugin lifecycle
    bool load_plugin(const std::string& plugin_path, const PluginContext& context) override;
    bool unload_plugin(const std::string& plugin_id) override;
    bool reload_plugin(const std::string& plugin_id) override;

    // Plugin discovery
    std::vector<PluginMetadata> discover_plugins(const std::string& search_path) override;
    PluginMetadata get_plugin_metadata(const std::string& plugin_id) override;
    std::vector<std::string> list_loaded_plugins() override;

    // Plugin execution
    nlohmann::json execute_plugin(const std::string& plugin_id, 
                                const nlohmann::json& input,
                                const PluginContext& context) override;

    // Security and sandboxing
    bool create_sandbox(const PluginContext& context) override;
    bool destroy_sandbox(const std::string& plugin_id) override;
    bool check_plugin_permissions(const std::string& plugin_id, 
                               const std::string& operation) override;

    // Resource monitoring
    nlohmann::json get_plugin_metrics(const std::string& plugin_id) override;
    nlohmann::json get_system_metrics() override;

private:
    struct PluginInstance {
        std::unique_ptr<IPlugin> plugin;
        PluginMetadata metadata;
        PluginState state;
        PluginContext context;
        
#ifdef _WIN32
        HMODULE library_handle;
#else
        void* library_handle;
#endif
        
        std::thread monitoring_thread;
        std::atomic<bool> should_monitor;
        std::atomic<size_t> memory_usage;
        std::atomic<double> cpu_usage;
    };

    std::map<std::string, std::unique_ptr<PluginInstance>> loaded_plugins_;
    std::mutex plugins_mutex_;
    
    // Sandbox implementation
    struct SandboxConfig {
        std::vector<std::string> allowed_paths;
        std::map<std::string, std::string> environment_vars;
        size_t max_memory_mb;
        double max_cpu_percent;
        int max_file_descriptors;
        bool network_access;
    };

    // Resource monitoring
    void start_monitoring(const std::string& plugin_id);
    void stop_monitoring(const std::string& plugin_id);
    void monitor_plugin_resources(const std::string& plugin_id);
    
    // Plugin loading utilities
    void* load_library(const std::string& path);
    void unload_library(void* handle);
    void* get_symbol(void* handle, const std::string& symbol);
    
    // Sandbox utilities
    bool create_container_sandbox(const PluginContext& context);
    bool create_chroot_sandbox(const PluginContext& context);
    bool create_namespace_sandbox(const PluginContext& context);
    bool setup_resource_limits(const std::string& plugin_id, const SandboxConfig& config);
    
    // Security utilities
    bool validate_plugin_signature(const std::string& plugin_path);
    bool check_plugin_dependencies(const PluginMetadata& metadata);
    bool scan_plugin_malware(const std::string& plugin_path);
    
    // Metrics collection
    size_t get_process_memory_usage();
    double get_process_cpu_usage();
    nlohmann::json collect_system_metrics();
};

// Plugin factory implementation
class PluginFactory : public IPluginFactory {
public:
    std::unique_ptr<IPlugin> create_plugin(const PluginMetadata& metadata) override;
    bool validate_plugin(const std::string& plugin_path) override;
    PluginMetadata extract_metadata(const std::string& plugin_path) override;

private:
    bool validate_plugin_structure(const std::string& plugin_path);
    bool check_plugin_compatibility(const PluginMetadata& metadata);
};

} // namespace serius::hub
