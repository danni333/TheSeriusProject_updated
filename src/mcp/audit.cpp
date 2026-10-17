#include "audit.h"

#include <iostream>

namespace serius {
namespace mcp {

Audit::Audit(const std::string &log_path)
{
    log_file_.open(log_path, std::ios::app);
    if (!log_file_) {
        std::cerr << "[WARN] Could not open audit log: " << log_path << std::endl;
    }
}

Audit::~Audit()
{
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

std::string Audit::timestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_MSC_VER)
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void Audit::log(const std::string &tool, const std::string &args,
                const std::string &outcome)
{
    std::string line = timestamp() + " | " + tool + " | args=" + args +
                       " | outcome=" + outcome + "\n";
    if (log_file_.is_open()) {
        log_file_ << line;
        log_file_.flush();
    }
    std::cout << line;
}

}  // namespace mcp
}  // namespace serius
