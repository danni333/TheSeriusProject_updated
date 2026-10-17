#pragma once

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace serius {
namespace mcp {

/// Audit trail: logs every tool action with a timestamp.
class Audit
{
   public:
    explicit Audit(const std::string &log_path = "mcp_audit.log");
    ~Audit();

    /// Log an action.
    void log(const std::string &tool, const std::string &args,
             const std::string &outcome);

   private:
    std::ofstream log_file_;

    static std::string timestamp();
};

}  // namespace mcp
}  // namespace serius
