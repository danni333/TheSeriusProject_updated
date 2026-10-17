#include "transport_stdio.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <string>

namespace serius {
namespace mcp {

using json = nlohmann::json;

StdioTransport::StdioTransport(const std::string &command) : command_(command)
{
    // Best-effort: open a read pipe from the subprocess. Writing back to the
    // same subprocess requires bidirectional pipes; for simplicity we leave
    // in_/out_ null until a full bidirectional pipe implementation is added.
    (void)command_;
}

StdioTransport::~StdioTransport()
{
    if (in_) {
        std::fclose(in_);
    }
    if (out_) {
        std::fclose(out_);
    }
}

json StdioTransport::send_request(const std::string &method,
                                  const json &params)
{
    (void)method;
    (void)params;
    return json{{"jsonrpc", "2.0"},
                {"id", nullptr},
                {"error", {{"code", -32000}, {"message", "stdio transport not connected"}}}};
}

}  // namespace mcp
}  // namespace serius
