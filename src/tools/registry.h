#pragma once

#include <functional>
#include <map>
#include <string>

#include "nlohmann/json.hpp"

namespace serius {
namespace tools {

using json = nlohmann::json;

/// Definition of a tool/function.
struct ToolDefinition {
    std::string name;
    std::string description;
    json parameters_schema;  // JSON Schema for parameters
    std::function<json(const json&)> handler;

    ToolDefinition() = default;
    ToolDefinition(const std::string& n, const std::string& d,
                  const json& ps, std::function<json(const json&)> h)
        : name(n), description(d), parameters_schema(ps), handler(h) {}
};

/// Registry for tools/functions.
class ToolRegistry
{
public:
    ToolRegistry() = default;
    ~ToolRegistry() = default;

    /// Register a new tool.
    /// @param tool Tool definition
    void register_tool(const ToolDefinition& tool);

    /// Unregister a tool by name.
    /// @param name Tool name
    void unregister_tool(const std::string& name);

    /// Execute a tool.
    /// @param name Tool name
    /// @param arguments Tool arguments (JSON)
    /// @return Tool result (JSON)
    json execute(const std::string& name, const json& arguments);

    /// Get tool definition by name.
    /// @param name Tool name
    /// @return Tool definition (empty if not found)
    ToolDefinition get_tool(const std::string& name) const;

    /// Get all registered tool names.
    /// @return List of tool names
    std::vector<std::string> list_tools() const;

    /// Check if a tool is registered.
    /// @param name Tool name
    /// @return true if registered
    bool has_tool(const std::string& name) const;

    /// Clear all tools.
    void clear();

private:
    std::map<std::string, ToolDefinition> tools_;

    /// Validate arguments against schema.
    /// @param arguments Arguments to validate
    /// @param schema JSON Schema
    /// @return true if valid
    bool validate_schema(const json& arguments, const json& schema) const;
};

}  // namespace tools
}  // namespace serius
