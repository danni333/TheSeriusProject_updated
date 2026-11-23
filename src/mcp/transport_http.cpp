#include "transport_http.h"

#include <httplib.h>

#include <atomic>
#include <nlohmann/json.hpp>
#include <string>

#include "../common/defaults.h"

namespace serius {
namespace mcp {

using json = nlohmann::json;

namespace {

std::atomic<int> g_next_http_id{1};

struct ParsedUrl
{
    std::string host;
    int port = 0;
    std::string path = "/";
    bool ssl = false;
};

ParsedUrl parse_url(const std::string &url)
{
    ParsedUrl out;
    std::string rest = url;
    const auto scheme_end = rest.find("://");
    if (scheme_end != std::string::npos) {
        out.ssl = rest.substr(0, scheme_end) == "https";
        rest = rest.substr(scheme_end + 3);
    }
    const auto slash = rest.find('/');
    std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
    if (slash != std::string::npos) {
        out.path = rest.substr(slash);
    }
    const auto colon = hostport.rfind(':');
    if (colon != std::string::npos) {
        out.host = hostport.substr(0, colon);
        try {
            out.port = std::stoi(hostport.substr(colon + 1));
        }
        catch (...) {
            out.port = -1;  // malformed port: rejected by the port <= 0 check
        }
    } else {
        out.host = hostport;
    }
    if (out.port == 0) out.port = out.ssl ? 443 : 80;
    if (out.port < 0) out.host.clear();  // signal invalid URL to the caller
    return out;
}

}  // namespace

HttpTransport::HttpTransport(const std::string &endpoint) : endpoint_(endpoint) {}

HttpTransport::~HttpTransport() = default;

json HttpTransport::send_request(const std::string &method, const json &params)
{
    const ParsedUrl url = parse_url(endpoint_);
    if (url.host.empty()) {
        return json{{"jsonrpc", "2.0"},
                    {"id", nullptr},
                    {"error", {{"code", -32600}, {"message", "invalid MCP HTTP endpoint: " + endpoint_}}}};
    }
    if (url.ssl) {
        return json{
            {"jsonrpc", "2.0"},
            {"id", nullptr},
            {"error", {{"code", -32004}, {"message", "HTTPS endpoints not supported (build without OpenSSL)"}}}};
    }

    const int id = g_next_http_id++;
    json req;
    req["jsonrpc"] = "2.0";
    req["id"] = id;
    req["method"] = method;
    req["params"] = params;

    httplib::Client cli(url.host, url.port);
    cli.set_connection_timeout(serius::defaults::kMcpConnectTimeoutSec);
    cli.set_read_timeout(serius::defaults::kMcpReadTimeoutSec);

    httplib::Headers headers = {{"Content-Type", "application/json"}, {"Accept", "application/json"}};
    auto res = cli.Post(url.path, headers, req.dump(), "application/json");
    if (!res) {
        return json{{"jsonrpc", "2.0"},
                    {"id", id},
                    {"error", {{"code", -32005}, {"message", "HTTP request failed (network error)"}}}};
    }
    try {
        json resp = json::parse(res->body);
        if (resp.value("id", -1) != id && !resp.contains("id")) {
            resp["id"] = id;
        }
        return resp;
    }
    catch (...) {
        json err;
        err["jsonrpc"] = "2.0";
        err["id"] = id;
        err["error"]["code"] = res->status;
        err["error"]["message"] = "non-JSON response (HTTP " + std::to_string(res->status) + ")";
        return err;
    }
}

}  // namespace mcp
}  // namespace serius
