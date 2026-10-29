#include "args_parser.h"

#include <cstdlib>
#include <iostream>

namespace serius {
namespace cli {

ArgsParser::ArgsParser(const std::string &program_name, const std::string &description)
    : program_name_(program_name), description_(description)
{
}

void ArgsParser::add_flag(const Flag &flag)
{
    flags_.push_back(flag);
}

const ArgsParser::Flag *ArgsParser::find_flag(const std::string &name) const
{
    for (const auto &flag : flags_) {
        if (flag.name == name || flag.short_name == name) {
            return &flag;
        }
    }
    return nullptr;
}

bool ArgsParser::parse(int argc, char **argv)
{
    positionals_.clear();
    error_.clear();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Check if it's a flag (starts with -)
        if (arg.size() > 0 && arg[0] == '-') {
            std::string flag_name = arg;

            // Handle --flag or -f
            if (flag_name.size() > 1 && flag_name[1] == '-') {
                flag_name = flag_name.substr(2);  // Remove --
            } else {
                flag_name = flag_name.substr(1);  // Remove -
            }

            const Flag *flag = find_flag(flag_name);
            if (!flag) {
                error_ = "Unknown flag: " + arg;
                return false;
            }

            std::string value;
            if (flag->has_value) {
                if (i + 1 >= argc) {
                    error_ = "Flag " + arg + " requires a value";
                    return false;
                }
                value = argv[++i];
            } else {
                value = "true";  // Boolean flag
            }

            // Store the value (using flag's canonical name)
            positionals_.push_back("--" + flag->name);
            positionals_.push_back(value);
        } else {
            // Positional argument
            positionals_.push_back(arg);
        }
    }

    return true;
}

bool ArgsParser::has(const std::string &name) const
{
    const Flag *flag = find_flag(name);
    if (!flag) {
        return false;
    }

    // Check if flag is in positionals
    for (size_t i = 0; i < positionals_.size(); ++i) {
        if (positionals_[i] == "--" + flag->name || positionals_[i] == "-" + flag->short_name) {
            return true;
        }
    }

    return false;
}

std::string ArgsParser::get(const std::string &name, const std::string &default_val) const
{
    const Flag *flag = find_flag(name);
    if (!flag) {
        return default_val;
    }

    // Find the flag in positionals
    for (size_t i = 0; i < positionals_.size(); ++i) {
        if (positionals_[i] == "--" + flag->name || positionals_[i] == "-" + flag->short_name) {
            if (i + 1 < positionals_.size()) {
                return positionals_[i + 1];
            }
        }
    }

    return flag->default_value.empty() ? default_val : flag->default_value;
}

int ArgsParser::get_int(const std::string &name, int default_val) const
{
    std::string val = get(name);
    if (val.empty()) {
        return default_val;
    }
    return std::atoi(val.c_str());
}

float ArgsParser::get_float(const std::string &name, float default_val) const
{
    std::string val = get(name);
    if (val.empty()) {
        return default_val;
    }
    return static_cast<float>(std::atof(val.c_str()));
}

void ArgsParser::print_help() const
{
    std::cout << "Usage: " << program_name_ << " [OPTIONS]\n\n";
    std::cout << description_ << "\n\n";
    std::cout << "Options:\n";

    for (const auto &flag : flags_) {
        std::cout << "  ";
        if (!flag.short_name.empty()) {
            std::cout << "-" << flag.short_name << ", ";
        }
        std::cout << "--" << flag.name;

        if (flag.has_value) {
            std::cout << " <value>";
        }

        if (!flag.default_value.empty()) {
            std::cout << " (default: " << flag.default_value << ")";
        }

        std::cout << "\n      " << flag.description << "\n";
    }
}

}  // namespace cli
}  // namespace serius
