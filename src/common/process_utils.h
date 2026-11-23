#pragma once

// Cross-platform process spawn WITHOUT a shell: the argument vector is passed
// to the OS verbatim, so arguments containing quotes, '$', backticks or
// spaces can never be reinterpreted as shell syntax.

#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
#endif

namespace serius::common {

#ifdef _WIN32
namespace detail {
inline std::string quote_windows_arg(const std::string &arg)
{
    std::string out = "\"";
    size_t backslashes = 0;
    for (char c : arg) {
        if (c == '\\') {
            ++backslashes;
        } else if (c == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out += '"';
            backslashes = 0;
        } else {
            out.append(backslashes, '\\');
            out += c;
            backslashes = 0;
        }
    }
    out.append(backslashes * 2, '\\');
    out += '"';
    return out;
}
}  // namespace detail
#endif

/// Run `args[0]` with the given arguments. No shell is involved.
/// When `stdout_out`/`stderr_out` are non-null the corresponding stream is
/// captured into it (the other stream is inherited). Returns the child's exit
/// code, or -1 if the process could not be started.
inline int run_process(const std::vector<std::string> &args, std::string *stdout_out = nullptr,
                       std::string *stderr_out = nullptr)
{
    if (args.empty()) { return -1; }
#ifdef _WIN32
    std::string cmdline;
    for (const auto &a : args) {
        if (!cmdline.empty()) { cmdline += ' '; }
        cmdline += detail::quote_windows_arg(a);
    }

    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_out = nullptr, write_out = nullptr;
    HANDLE read_err = nullptr, write_err = nullptr;
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    const bool cap_out = stdout_out != nullptr;
    const bool cap_err = stderr_out != nullptr;
    if (cap_out) {
        if (!CreatePipe(&read_out, &write_out, &sa, 0)) { return -1; }
        SetHandleInformation(read_out, HANDLE_FLAG_INHERIT, 0);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = write_out;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    if (cap_err) {
        if (!CreatePipe(&read_err, &write_err, &sa, 0)) {
            if (read_out) CloseHandle(read_out);
            if (write_out) CloseHandle(write_out);
            return -1;
        }
        SetHandleInformation(read_err, HANDLE_FLAG_INHERIT, 0);
        if (cap_out) {
            si.hStdError = write_err;
        } else {
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
            si.hStdError = write_err;
            si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        }
    }

    PROCESS_INFORMATION pi{};
    std::vector<char> buf(cmdline.begin(), cmdline.end());
    buf.push_back('\0');
    const BOOL ok =
        CreateProcessA(nullptr, buf.data(), nullptr, nullptr,
                       (cap_out || cap_err) ? TRUE : FALSE, 0, nullptr, nullptr, &si, &pi);
    if (write_out) CloseHandle(write_out);
    if (write_err) CloseHandle(write_err);
    if (!ok) {
        if (read_out) CloseHandle(read_out);
        if (read_err) CloseHandle(read_err);
        return -1;
    }
    // Drain both pipes concurrently: a child that fills one pipe while we
    // block reading the other would deadlock (compiler diagnostics on stderr
    // routinely exceed the pipe buffer).
    std::thread drain_out, drain_err;
    if (cap_out) {
        drain_out = std::thread([read_out, stdout_out]() {
            char buffer[8192];
            DWORD n = 0;
            while (ReadFile(read_out, buffer, sizeof(buffer), &n, nullptr) && n > 0) {
                stdout_out->append(buffer, n);
            }
            CloseHandle(read_out);
        });
    }
    if (cap_err) {
        drain_err = std::thread([read_err, stderr_out]() {
            char buffer[8192];
            DWORD n = 0;
            while (ReadFile(read_err, buffer, sizeof(buffer), &n, nullptr) && n > 0) {
                stderr_out->append(buffer, n);
            }
            CloseHandle(read_err);
        });
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    if (drain_out.joinable()) { drain_out.join(); }
    if (drain_err.joinable()) { drain_err.join(); }
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
#else
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &a : args) { argv.push_back(const_cast<char *>(a.c_str())); }
    argv.push_back(nullptr);

    int fds_out[2] = {-1, -1};
    int fds_err[2] = {-1, -1};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    auto make_pipe = [](int fds[2]) {
        if (pipe(fds) != 0) { return false; }
        fcntl(fds[0], F_SETFD, FD_CLOEXEC);
        fcntl(fds[1], F_SETFD, FD_CLOEXEC);
        return true;
    };
    if (stdout_out && !make_pipe(fds_out)) {
        posix_spawn_file_actions_destroy(&actions);
        return -1;
    }
    if (stderr_out && !make_pipe(fds_err)) {
        if (stdout_out) { close(fds_out[0]); close(fds_out[1]); }
        posix_spawn_file_actions_destroy(&actions);
        return -1;
    }
    if (stdout_out) { posix_spawn_file_actions_adddup2(&actions, fds_out[1], STDOUT_FILENO); }
    if (stderr_out) { posix_spawn_file_actions_adddup2(&actions, fds_err[1], STDERR_FILENO); }

    pid_t pid = 0;
    const int rc = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (stdout_out) close(fds_out[1]);
    if (stderr_out) close(fds_err[1]);
    if (rc != 0) {
        if (stdout_out) close(fds_out[0]);
        if (stderr_out) close(fds_err[0]);
        return -1;
    }
    // Drain both pipes concurrently to avoid the child blocking on a full
    // pipe while we read the other stream.
    auto drain = [](int fd, std::string *dst) {
        char buffer[8192];
        for (;;) {
            const ssize_t n = read(fd, buffer, sizeof(buffer));
            if (n > 0) {
                dst->append(buffer, static_cast<size_t>(n));
            } else if (n < 0 && errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        close(fd);
    };
    std::thread drain_out, drain_err;
    if (stdout_out) { drain_out = std::thread(drain, fds_out[0], stdout_out); }
    if (stderr_out) { drain_err = std::thread(drain, fds_err[0], stderr_out); }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            if (drain_out.joinable()) drain_out.join();
            if (drain_err.joinable()) drain_err.join();
            return -1;
        }
    }
    if (drain_out.joinable()) { drain_out.join(); }
    if (drain_err.joinable()) { drain_err.join(); }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

}  // namespace serius::common
