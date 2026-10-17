#pragma once

#include "transport.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace serius {
namespace mcp {

using json = nlohmann::json;

/// A tool exposed by an MCP server.
struct MCPTool
{
    std::string name;
    std::string description;
    json input_schema;
};

/// MCP client: JSON-RPC 2.0 over a Transport, with tool discovery.
class MCPClient
{
   public:
    explicit MCPClient(Transport &transport);
    ~MCPClient();

    MCPClient(const MCPClient &) = delete;
    MCPClient &operator=(const MCPClient &) = delete;

    /// List available tools from the server.
    std::vector<MCPTool> list_tools();

    /// Call a tool by name with JSON arguments.
    json call_tool(const std::string &name, const json &arguments);

    /// Format discovered tools into a prompt-friendly description for the model.
    static std::string format_tools_for_prompt(const std::vector<MCPTool> &tools);

   private:
    Transport &transport_;
    int next_id_ = 1;

    json send_request(const std::string &method, const json &params = json::object());
};

}  // namespace mcp
}  // namespace serius
