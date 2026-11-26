#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

#include "../mcp/client.h"

namespace serius {
namespace mcp {

using json = nlohmann::json;

enum class RiskLevel { SAFE, MODERATE, DANGEROUS };

/// MCPGuard intercepts tool calls, classifies risk, and enforces human approval
/// for dangerous operations.
class MCPGuard
{
   public:
    using ApprovalCallback = std::function<bool(const std::string &tool, const json &args)>;

    MCPGuard();

    /// Classify the risk level of a tool.
    static RiskLevel risk_level(const std::string &tool);

    /// Call a tool through the guard. Dangerous tools require approval.
    json guarded_call(MCPClient &mcp, const std::string &tool, const json &args, ApprovalCallback approval = nullptr);

    /// Set a custom risk level for a tool.
    void set_risk(const std::string &tool, RiskLevel level);

   private:
    // Shared risk table across instances (API compatibility: risk_level is
    // static). Guarded by risk_mutex_; initialized once from kDefaultRisks —
    // constructors must NOT reset it (that would wipe other instances'
    // custom set_risk() entries).
    static std::map<std::string, RiskLevel> tool_risk_;
    static std::mutex risk_mutex_;
};

}  // namespace mcp
}  // namespace serius
