#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>

namespace serius::hub {

// Plugin capability flags
enum class PluginCapability {
    INFERENCE = 1 << 0,
    TRAINING = 1 << 1,
    PREPROCESSING = 1 << 2,
    POSTPROCESSING = 1 << 3,
    CUSTOM_KERNELS = 1 << 4,
    MODEL_FORMATS = 1 << 5,
    SECURITY_HOOKS = 1 << 6
};

// Plugin lifecycle states
enum class PluginState {
    UNLOADED,
    LOADING,
    LOADED,
    ACTIVE,
    ERROR,
    UNLOADING
};

// Plugin metadata structure
struct PluginMetadata {
    std::string id;
    std::string name;
    std::string version;
    std::string author;
    std::string description;
    std::string license;
    std::vector<std::string> dependencies;
    uint32_t capabilities;
    std::string entry_point;  // Path to plugin library
    std::string config_schema;  // JSON schema for plugin config
    std::string created_at;
    std::string updated_at;
};

// Plugin context for sandboxing
struct PluginContext {
    std::string plugin_id;
    std::string user_id;
    std::string session_id;
    nlohmann::json config;
    void* sandbox_handle;
    std::vector<std::string> allowed_paths;
    std::map<std::string, std::string> environment_vars;
};

// Base plugin interface
class IPlugin {
public:
    virtual ~IPlugin() = default;
    
    // Lifecycle
    virtual bool initialize(const PluginContext& context) = 0;
    virtual bool shutdown() = 0;
    virtual PluginState get_state() const = 0;
    
    // Capabilities
    virtual bool has_capability(PluginCapability cap) const = 0;
    virtual uint32_t get_capabilities() const = 0;
    
    // Core operations
    virtual nlohmann::json process(const nlohmann::json& input) = 0;
    virtual bool validate_input(const nlohmann::json& input) = 0;
    virtual nlohmann::json get_status() = 0;
    
    // Configuration
    virtual bool update_config(const nlohmann::json& config) = 0;
    virtual nlohmann::json get_config() = 0;
    
    // Resource management
    virtual size_t get_memory_usage() = 0;
    virtual double get_cpu_usage() = 0;
    virtual void cleanup_resources() = 0;
};

// Plugin factory interface
class IPluginFactory {
public:
    virtual ~IPluginFactory() = default;
    virtual std::unique_ptr<IPlugin> create_plugin(const PluginMetadata& metadata) = 0;
    virtual bool validate_plugin(const std::string& plugin_path) = 0;
    virtual PluginMetadata extract_metadata(const std::string& plugin_path) = 0;
};

// Plugin manager interface
class IPluginManager {
public:
    virtual ~IPluginManager() = default;
    
    // Plugin lifecycle
    virtual bool load_plugin(const std::string& plugin_path, const PluginContext& context) = 0;
    virtual bool unload_plugin(const std::string& plugin_id) = 0;
    virtual bool reload_plugin(const std::string& plugin_id) = 0;
    
    // Plugin discovery
    virtual std::vector<PluginMetadata> discover_plugins(const std::string& search_path) = 0;
    virtual PluginMetadata get_plugin_metadata(const std::string& plugin_id) = 0;
    virtual std::vector<std::string> list_loaded_plugins() = 0;
    
    // Plugin execution
    virtual nlohmann::json execute_plugin(const std::string& plugin_id, 
                                        const nlohmann::json& input,
                                        const PluginContext& context) = 0;
    
    // Security and sandboxing
    virtual bool create_sandbox(const PluginContext& context) = 0;
    virtual bool destroy_sandbox(const std::string& plugin_id) = 0;
    virtual bool check_plugin_permissions(const std::string& plugin_id, 
                                         const std::string& operation) = 0;
    
    // Resource monitoring
    virtual nlohmann::json get_plugin_metrics(const std::string& plugin_id) = 0;
    virtual nlohmann::json get_system_metrics() = 0;
};

// Plugin SDK utilities
namespace plugin_sdk {

// Helper functions for plugin development
std::string generate_plugin_id(const std::string& name, const std::string& version);
bool validate_plugin_schema(const nlohmann::json& config, const std::string& schema);
nlohmann::json create_error_response(const std::string& error_code, const std::string& message);
nlohmann::json create_success_response(const nlohmann::json& data);

// Logging utilities
void log_plugin_info(const std::string& plugin_id, const std::string& message);
void log_plugin_error(const std::string& plugin_id, const std::string& message);
void log_plugin_warning(const std::string& plugin_id, const std::string& message);

// Resource monitoring helpers
size_t get_current_memory_usage();
double get_current_cpu_usage();
bool check_resource_limits(size_t memory_mb, double cpu_percent);

} // namespace plugin_sdk

} // namespace serius::hub
