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

json ToolRegistry::execute(const std::string& name, const json& arguments) const
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
    // null counts as "no arguments" (same as {}); anything else non-object
    // (array, scalar) is malformed.
    if (!arguments.is_object() && !arguments.is_null()) {
        return false;
    }
    if (!schema.is_object() || !schema.contains("properties")) {
        return true;  // No schema = no validation
    }

    // Only the fields listed under "required" must be present; every other
    // declared property is optional (standard JSON Schema semantics).
    if (schema.contains("required") && schema["required"].is_array()) {
        for (const auto& req : schema["required"]) {
            if (req.is_string() && !arguments.contains(req.get<std::string>())) {
                return false;
            }
        }
    }

    // Type-check provided arguments against the declared property schemas.
    const auto& properties = schema["properties"];
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
        if (!properties.contains(it.key())) {
            continue;  // Undeclared argument: allowed (schemas are not closed)
        }
        const json& prop = properties[it.key()];
        if (!prop.is_object() || !prop.contains("type")) {
            continue;
        }
        const std::string& t = prop["type"].is_string() ? prop["type"].get<std::string>() : "";
        const json& v = it.value();
        bool ok = true;
        if (t == "string") ok = v.is_string();
        else if (t == "integer") ok = v.is_number_integer();
        else if (t == "number") ok = v.is_number();
        else if (t == "boolean") ok = v.is_boolean();
        else if (t == "object") ok = v.is_object();
        else if (t == "array") ok = v.is_array();
        else if (t == "null") ok = v.is_null();
        if (!ok) {
            return false;
        }
    }

    return true;
}

}  // namespace tools
}  // namespace serius
