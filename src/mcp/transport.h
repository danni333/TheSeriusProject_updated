#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace serius {
namespace mcp {

using json = nlohmann::json;

/// Transport abstraction for MCP: stdio or HTTP+SSE.
class Transport
{
   public:
    virtual ~Transport() = default;

    /// Send a JSON-RPC request and return the parsed response.
    virtual json send_request(const std::string &method,
                              const json &params = json::object()) = 0;
};

}  // namespace mcp
}  // namespace serius
