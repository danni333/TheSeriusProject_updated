#pragma once

#include <cctype>
#include <string>

namespace serius {
namespace common {

inline std::string to_lower(std::string s)
{
    for (auto &c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

}  // namespace common
}  // namespace serius
