#include "guard.h"

#include <nlohmann/json.hpp>

#include <iostream>

namespace serius {
namespace mcp {

using json = nlohmann::json;

namespace {
const std::map<std::string, RiskLevel> kDefaultRisks = {
    {"read_file", RiskLevel::SAFE},
    {"search", RiskLevel::SAFE},
    {"list_files", RiskLevel::SAFE},
    {"write_file", RiskLevel::MODERATE},
    {"edit_file", RiskLevel::MODERATE},
    {"run_command", RiskLevel::DANGEROUS},
    {"send_email", RiskLevel::DANGEROUS},
    {"delete_file", RiskLevel::DANGEROUS},
    {"format_disk", RiskLevel::DANGEROUS},
};
}  // namespace

std::map<std::string, RiskLevel> MCPGuard::tool_risk_ = kDefaultRisks;

MCPGuard::MCPGuard()
{
    tool_risk_ = kDefaultRisks;
}

RiskLevel MCPGuard::risk_level(const std::string &tool)
{
    auto it = tool_risk_.find(tool);
    if (it == tool_risk_.end()) {
        return RiskLevel::MODERATE;  // unknown tools default to moderate
    }
    return it->second;
}

json MCPGuard::guarded_call(MCPClient &mcp, const std::string &tool,
                            const json &args, ApprovalCallback approval)
{
    RiskLevel risk = risk_level(tool);

    if (risk == RiskLevel::DANGEROUS) {
        bool approved = false;
        if (approval) {
            approved = approval(tool, args);
        }
        if (!approved) {
            return json{{"error", "Operazione rifiutata dall'utente"},
                        {"tool", tool},
                        {"risk", "dangerous"}};
        }
    }

    if (risk >= RiskLevel::MODERATE) {
        std::cout << "[AUDIT] " << tool << " args=" << args.dump() << std::endl;
    }

    return mcp.call_tool(tool, args);
}

void MCPGuard::set_risk(const std::string &tool, RiskLevel level)
{
    tool_risk_[tool] = level;
}

}  // namespace mcp
}  // namespace serius
