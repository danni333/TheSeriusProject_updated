#include "client.h"

#include <sstream>

namespace serius {
namespace mcp {

using json = nlohmann::json;

MCPClient::MCPClient(Transport &transport) : transport_(transport) {}

MCPClient::~MCPClient() = default;

json MCPClient::send_request(const std::string &method, const json &params)
{
    return transport_.send_request(method, params);
}

std::vector<MCPTool> MCPClient::list_tools()
{
    std::vector<MCPTool> tools;
    json resp = send_request("tools/list", json::object());
    if (!resp.contains("result") || !resp["result"].contains("tools")) {
        return tools;
    }
    for (const auto &t : resp["result"]["tools"]) {
        MCPTool tool;
        tool.name = t.value("name", std::string{});
        tool.description = t.value("description", std::string{});
        if (t.contains("inputSchema")) {
            tool.input_schema = t["inputSchema"];
        } else {
            tool.input_schema = t.value("input_schema", json::object());
        }
        tools.push_back(std::move(tool));
    }
    return tools;
}

json MCPClient::call_tool(const std::string &name, const json &arguments)
{
    json params;
    params["name"] = name;
    params["arguments"] = arguments;
    return send_request("tools/call", params);
}

std::string MCPClient::format_tools_for_prompt(const std::vector<MCPTool> &tools)
{
    std::ostringstream oss;
    oss << "Available tools:\n";
    for (const auto &t : tools) {
        oss << "- " << t.name << ": " << t.description
            << " | Args: " << t.input_schema.dump() << "\n";
    }
    return oss.str();
}

}  // namespace mcp
}  // namespace serius
