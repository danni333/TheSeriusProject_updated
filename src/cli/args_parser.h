#pragma once

#include <functional>
#include <string>
#include <vector>

namespace serius {
namespace cli {

/// Simple command-line argument parser without external dependencies.
class ArgsParser
{
   public:
    struct Flag
    {
        std::string name;
        std::string short_name;
        std::string description;
        bool has_value = false;
        std::string default_value;
    };

    ArgsParser(const std::string &program_name, const std::string &description);

    /// Add a flag to the parser.
    void add_flag(const Flag &flag);

    /// Parse command-line arguments.
    /// Returns true on success, false on error.
    bool parse(int argc, char **argv);

    /// Check if a flag was provided.
    bool has(const std::string &name) const;

    /// Get the value of a flag (as string).
    std::string get(const std::string &name, const std::string &default_val = "") const;

    /// Get the value as int.
    int get_int(const std::string &name, int default_val = 0) const;

    /// Get the value as float.
    float get_float(const std::string &name, float default_val = 0.0f) const;

    /// Get positional arguments (non-flag arguments).
    const std::vector<std::string> &positionals() const { return positionals_; }

    /// Print help message.
    void print_help() const;

    /// Get error message if parsing failed.
    const std::string &error() const { return error_; }

   private:
    std::string program_name_;
    std::string description_;
    std::vector<Flag> flags_;
    std::vector<std::string> positionals_;
    std::string error_;

    /// Find flag by name or short name.
    const Flag *find_flag(const std::string &name) const;
};

}  // namespace cli
}  // namespace serius
