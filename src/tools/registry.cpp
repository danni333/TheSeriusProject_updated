#include "registry.h"

namespace serius {
namespace tools {

void ToolRegistry::register_tool(const ToolDefinition& tool)
{
    tools_[tool.name] = tool;
}

void ToolRegistry::unregister_tool(const std::string& name)
{
    tools_.erase(name);
}

json ToolRegistry::execute(const std::string& name, const json& arguments)
{
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        return json{{"error", "Tool not found: " + name}};
    }

    const ToolDefinition& tool = it->second;

    // Validate arguments
    if (!validate_schema(arguments, tool.parameters_schema)) {
        return json{{"error", "Invalid arguments for tool: " + name}};
    }

    // Execute handler
    try {
        return tool.handler(arguments);
    } catch (const std::exception& e) {
        return json{{"error", std::string("Tool execution failed: ") + e.what()}};
    }
}

ToolDefinition ToolRegistry::get_tool(const std::string& name) const
{
    auto it = tools_.find(name);
    if (it != tools_.end()) {
        return it->second;
    }
    return ToolDefinition{};
}

std::vector<std::string> ToolRegistry::list_tools() const
{
    std::vector<std::string> names;
    for (const auto& [name, _] : tools_) {
        names.push_back(name);
    }
    return names;
}

bool ToolRegistry::has_tool(const std::string& name) const
{
    return tools_.find(name) != tools_.end();
}

void ToolRegistry::clear()
{
    tools_.clear();
}

bool ToolRegistry::validate_schema(const json& arguments, const json& schema) const
{
    // Simplified validation: check if required fields are present
    // In production, this would use a full JSON Schema validator
    if (!schema.contains("properties")) {
        return true;  // No schema = no validation
    }

    const auto& properties = schema["properties"];
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        const std::string& key = it.key();
        if (!arguments.contains(key)) {
            return false;
        }
    }

    return true;
}

}  // namespace tools
}  // namespace serius
