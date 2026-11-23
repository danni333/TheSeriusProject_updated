#include "registry.h"

#include <exception>
#include <regex>

namespace serius {
namespace tools {

namespace {

bool type_matches(const json &v, const std::string &t)
{
    if (t == "string")
        return v.is_string();
    if (t == "integer")
        return v.is_number_integer();
    if (t == "number")
        return v.is_number();
    if (t == "boolean")
        return v.is_boolean();
    if (t == "object")
        return v.is_object();
    if (t == "array")
        return v.is_array();
    if (t == "null")
        return v.is_null();
    return true;  // Unknown type keyword: don't reject on type alone.
}

/// Validate a single JSON value against a JSON Schema fragment.
/// Supports: type (string or array), enum, minimum/maximum, minLength/
/// maxLength, pattern, minItems/maxItems, items, nested properties/required,
/// additionalProperties:false.
bool validate_value(const json &v, const json &schema)
{
    if (!schema.is_object())
        return true;

    if (schema.contains("type")) {
        const json &t = schema["type"];
        bool ok = false;
        if (t.is_string()) {
            ok = type_matches(v, t.get<std::string>());
        } else if (t.is_array()) {
            for (const auto &tt : t) {
                if (tt.is_string() && type_matches(v, tt.get<std::string>())) {
                    ok = true;
                    break;
                }
            }
        }
        if (!ok)
            return false;
    }

    if (schema.contains("enum") && schema["enum"].is_array()) {
        bool found = false;
        for (const auto &e : schema["enum"]) {
            if (e == v) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }

    if (v.is_number()) {
        if (schema.contains("minimum") && schema["minimum"].is_number() &&
            v.get<double>() < schema["minimum"].get<double>()) {
            return false;
        }
        if (schema.contains("maximum") && schema["maximum"].is_number() &&
            v.get<double>() > schema["maximum"].get<double>()) {
            return false;
        }
    }

    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        if (schema.contains("minLength") && s.size() < schema["minLength"].get<size_t>())
            return false;
        if (schema.contains("maxLength") && s.size() > schema["maxLength"].get<size_t>())
            return false;
        if (schema.contains("pattern") && schema["pattern"].is_string()) {
            try {
                if (!std::regex_search(s, std::regex(schema["pattern"].get<std::string>()))) {
                    return false;
                }
            } catch (const std::regex_error &) {
                return false;  // Invalid pattern in the schema itself.
            }
        }
    }

    if (v.is_array()) {
        if (schema.contains("minItems") && v.size() < schema["minItems"].get<size_t>())
            return false;
        if (schema.contains("maxItems") && v.size() > schema["maxItems"].get<size_t>())
            return false;
        if (schema.contains("items") && schema["items"].is_object()) {
            for (const auto &el : v) {
                if (!validate_value(el, schema["items"]))
                    return false;
            }
        }
    }

    if (v.is_object() && schema.contains("properties") && schema["properties"].is_object()) {
        const json &props = schema["properties"];
        const bool closed = schema.contains("additionalProperties") &&
                            schema["additionalProperties"].is_boolean() &&
                            !schema["additionalProperties"].get<bool>();
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (props.contains(it.key())) {
                if (!validate_value(it.value(), props[it.key()]))
                    return false;
            } else if (closed) {
                return false;
            }
        }
        if (schema.contains("required") && schema["required"].is_array()) {
            for (const auto &req : schema["required"]) {
                if (req.is_string() && !v.contains(req.get<std::string>()))
                    return false;
            }
        }
    }

    return true;
}

}  // namespace

void ToolRegistry::register_tool(const ToolDefinition &tool)
{
    tools_[tool.name] = tool;
}

void ToolRegistry::unregister_tool(const std::string &name)
{
    tools_.erase(name);
}

json ToolRegistry::execute(const std::string &name, const json &arguments) const
{
    auto it = tools_.find(name);
    if (it == tools_.end()) {
        return json{{"error", "Tool not found: " + name}};
    }

    const ToolDefinition &tool = it->second;

    // Validate arguments
    if (!validate_schema(arguments, tool.parameters_schema)) {
        return json{{"error", "Invalid arguments for tool: " + name}};
    }

    // Execute handler
    try {
        return tool.handler(arguments);
    }
    catch (const std::exception &e) {
        return json{{"error", std::string("Tool execution failed: ") + e.what()}};
    }
}

ToolDefinition ToolRegistry::get_tool(const std::string &name) const
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
    for (const auto &[name, _] : tools_) {
        names.push_back(name);
    }
    return names;
}

bool ToolRegistry::has_tool(const std::string &name) const
{
    return tools_.find(name) != tools_.end();
}

void ToolRegistry::clear()
{
    tools_.clear();
}

bool ToolRegistry::validate_schema(const json &arguments, const json &schema) const
{
    // null counts as "no arguments" (same as {}); anything else non-object
    // (array, scalar) is malformed.
    if (!arguments.is_object() && !arguments.is_null()) {
        return false;
    }
    static const json empty_args = json::object();
    const json &args = arguments.is_null() ? empty_args : arguments;

    if (!schema.is_object()) {
        return true;  // No schema = no validation
    }

    // Schemas without "properties" may still carry value-level keywords.
    if (!schema.contains("properties")) {
        return validate_value(args, schema);
    }

    const auto &properties = schema["properties"];
    const bool closed = schema.contains("additionalProperties") &&
                        schema["additionalProperties"].is_boolean() &&
                        !schema["additionalProperties"].get<bool>();

    // Only the fields listed under "required" must be present; every other
    // declared property is optional (standard JSON Schema semantics).
    if (schema.contains("required") && schema["required"].is_array()) {
        for (const auto &req : schema["required"]) {
            if (req.is_string() && !args.contains(req.get<std::string>())) {
                return false;
            }
        }
    }

    // Validate each provided argument against its declared property schema.
    for (auto it = args.begin(); it != args.end(); ++it) {
        if (!properties.is_object() || !properties.contains(it.key())) {
            if (closed) {
                return false;  // additionalProperties: false
            }
            continue;  // Undeclared argument: allowed (schemas are not closed by default)
        }
        if (!validate_value(it.value(), properties[it.key()])) {
            return false;
        }
    }

    return true;
}

}  // namespace tools
}  // namespace serius
