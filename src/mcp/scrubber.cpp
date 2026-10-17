#include "scrubber.h"

#include <algorithm>
#include <cctype>

namespace serius {
namespace mcp {

Scrubber::Scrubber()
{
    keywords_.push_back("api_key");
    keywords_.push_back("apikey");
    keywords_.push_back("token");
    keywords_.push_back("secret");
    keywords_.push_back("password");
    keywords_.push_back("passwd");
    keywords_.push_back("pwd");

    complex_.emplace_back(R"(\b[A-Za-z0-9+/]{40,}={0,2}\b)");
    complex_.emplace_back(R"(sk-[a-zA-Z0-9]{32,})");
    complex_.emplace_back(R"(ghp_[a-zA-Z0-9]{32,})");
}

namespace {

std::string to_lower_copy(const std::string &s)
{
    std::string r = s;
    for (auto &c : r) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return r;
}

std::string replace_ci(std::string text, const std::string &kw,
                       const std::string &replacement)
{
    std::string low = to_lower_copy(text);
    std::string low_kw = to_lower_copy(kw);
    size_t pos = 0;
    while ((pos = low.find(low_kw, pos)) != std::string::npos) {
        text.replace(pos, kw.size(), replacement);
        low.replace(pos, kw.size(), replacement);
        pos += replacement.size();
    }
    return text;
}

}  // namespace

std::string Scrubber::scrub(const std::string &text) const
{
    std::string result = text;
    for (const auto &kw : keywords_) {
        result = replace_ci(result, kw, "[REDACTED]");
    }
    for (const auto &p : complex_) {
        result = std::regex_replace(result, p, "[REDACTED]");
    }
    return result;
}

void Scrubber::add_pattern(const std::string &pattern)
{
    complex_.emplace_back(pattern);
}

}  // namespace mcp
}  // namespace serius
