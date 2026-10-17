#include "transport_http.h"

#include <nlohmann/json.hpp>

#include <string>

namespace serius {
namespace mcp {

using json = nlohmann::json;

HttpTransport::HttpTransport(const std::string &endpoint) : endpoint_(endpoint) {}

HttpTransport::~HttpTransport() = default;

json HttpTransport::send_request(const std::string &method,
                                 const json &params)
{
    (void)method;
    (void)params;
    (void)endpoint_;
    return json{{"jsonrpc", "2.0"},
                {"id", nullptr},
                {"error", {{"code", -32001}, {"message", "HTTP transport not implemented"}}},
            };
}

}  // namespace mcp
}  // namespace serius
