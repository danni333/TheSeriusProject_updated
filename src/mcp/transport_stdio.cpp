#include "transport_stdio.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace serius {
namespace mcp {

using json = nlohmann::json;

namespace {

std::atomic<int> g_next_stdio_id{1};

}  // namespace

StdioTransport::StdioTransport(const std::string &command) : command_(command)
{
    spawn();
}

StdioTransport::~StdioTransport()
{
    if (in_) {
        std::fclose(in_);
        in_ = nullptr;
    }
    if (out_) {
        std::fclose(out_);
        out_ = nullptr;
    }
#if defined(_WIN32)
    if (process_) {
        auto h = static_cast<HANDLE>(process_);
        TerminateProcess(h, 1);
        CloseHandle(h);
        process_ = nullptr;
    }
#else
    if (pid_ > 0) {
        kill(static_cast<pid_t>(pid_), SIGTERM);
        waitpid(static_cast<pid_t>(pid_), nullptr, 0);
        pid_ = -1;
    }
#endif
}

void StdioTransport::spawn()
{
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE child_in_read = nullptr, child_in_write = nullptr;
    HANDLE child_out_read = nullptr, child_out_write = nullptr;
    if (!CreatePipe(&child_in_read, &child_in_write, &sa, 0) ||
        !CreatePipe(&child_out_read, &child_out_write, &sa, 0)) {
        return;
    }
    // The child ends (child_in_read, child_out_write) must stay inheritable
    // for STARTF_USESTDHANDLES; the parent ends must not leak into the child.
    SetHandleInformation(child_in_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(child_out_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = child_in_read;
    si.hStdOutput = child_out_write;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION pi{};
    std::string cmdline = command_;  // CreateProcess needs a mutable buffer
    if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si,
                        &pi)) {
        CloseHandle(child_in_read);
        CloseHandle(child_in_write);
        CloseHandle(child_out_read);
        CloseHandle(child_out_write);
        return;
    }
    process_ = pi.hProcess;
    CloseHandle(pi.hThread);
    // Parent keeps the opposite ends; child ends must be closed here.
    CloseHandle(child_in_read);
    CloseHandle(child_out_write);

    const int in_fd = _open_osfhandle(reinterpret_cast<intptr_t>(child_in_write), _O_WRONLY | _O_TEXT);
    const int out_fd = _open_osfhandle(reinterpret_cast<intptr_t>(child_out_read), _O_RDONLY | _O_TEXT);
    in_ = in_fd >= 0 ? _fdopen(in_fd, "w") : nullptr;
    out_ = out_fd >= 0 ? _fdopen(out_fd, "r") : nullptr;
    if (in_) std::setvbuf(in_, nullptr, _IONBF, 0);
    if (out_) std::setvbuf(out_, nullptr, _IONBF, 0);
#else
    int in_pipe[2] = {-1, -1}, out_pipe[2] = {-1, -1};
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
        return;
    }
    pid_ = fork();
    if (pid_ == 0) {
        // Child: stdin <- in_pipe[0], stdout -> out_pipe[1]
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        execl("/bin/sh", "sh", "-c", command_.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    if (pid_ < 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        pid_ = -1;
        return;
    }
    close(in_pipe[0]);
    close(out_pipe[1]);
    in_ = fdopen(in_pipe[1], "w");
    out_ = fdopen(out_pipe[0], "r");
    if (in_) std::setvbuf(in_, nullptr, _IONBF, 0);
    if (out_) std::setvbuf(out_, nullptr, _IONBF, 0);
#endif
}

json StdioTransport::send_request(const std::string &method, const json &params)
{
    if (!in_ || !out_) {
        return json{{"jsonrpc", "2.0"},
                    {"id", nullptr},
                    {"error", {{"code", -32000}, {"message", "stdio transport not connected"}}}};
    }

    const int id = g_next_stdio_id++;
    json req;
    req["jsonrpc"] = "2.0";
    req["id"] = id;
    req["method"] = method;
    req["params"] = params;

    const std::string line = req.dump() + "\n";
    if (std::fwrite(line.data(), 1, line.size(), in_) != line.size() || std::fflush(in_) != 0) {
        return json{{"jsonrpc", "2.0"},
                    {"id", id},
                    {"error", {{"code", -32002}, {"message", "failed to write request to subprocess"}}}};
    }

    // MCP stdio uses newline-delimited JSON. Skip notifications (no "id")
    // and keep reading until the response matching our id arrives.
    char buf[65536];
    while (std::fgets(buf, sizeof(buf), out_) != nullptr) {
        json resp;
        try {
            resp = json::parse(buf);
        }
        catch (...) {
            continue;  // tolerate non-JSON noise on stdout
        }
        if (resp.value("id", -1) == id) {
            return resp;
        }
    }

    return json{{"jsonrpc", "2.0"},
                {"id", id},
                {"error", {{"code", -32003}, {"message", "subprocess closed stdout (crashed or exited)"}}}};
}

}  // namespace mcp
}  // namespace serius
