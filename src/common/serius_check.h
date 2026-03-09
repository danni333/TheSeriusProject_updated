#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace serius {

/**
 * @brief Base exception class for the Serius project.
 */
class Error : public std::runtime_error
{
   public:
    Error(const std::string &msg, const char *file, int line)
        : std::runtime_error(msg + " at " + file + ":" + std::to_string(line)), file_(file), line_(line)
    {
    }

    const char *file() const { return file_; }
    int line() const { return line_; }

   private:
    const char *file_;
    int line_;
};

/**
 * @brief Fatal error class.
 */
class FatalError : public Error
{
   public:
    using Error::Error;
};

namespace debug {
inline void panic(const char *file, int line, const char *cond)
{
    std::cerr << "PANIC: Assertion failed: (" << cond << ") at " << file << ":" << line << std::endl;
    std::abort();
}
}  // namespace debug

}  // namespace serius

/**
 * @brief Programmer error - abort in all builds if condition fails
 * NOTE: The spec says no-op in Release for SERIUS_ASSERT, but also says "abort in Debug".
 * I'll follow the "abort in Debug, no-op in Release" pattern if NDEBUG is defined.
 */
#ifdef NDEBUG
#define SERIUS_ASSERT(cond) ((void)0)
#else
#define SERIUS_ASSERT(cond)                                  \
    do {                                                     \
        if (!(cond)) {                                       \
            serius::debug::panic(__FILE__, __LINE__, #cond); \
        }                                                    \
    } while (0)
#endif

/**
 * @brief Runtime check - throws serius::Error if condition fails
 */
#define SERIUS_CHECK(cond, msg)                           \
    do {                                                  \
        if (!(cond)) {                                    \
            throw serius::Error(msg, __FILE__, __LINE__); \
        }                                                 \
    } while (0)
