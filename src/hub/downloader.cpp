#include "downloader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
extern char **environ;
#endif

namespace serius {
namespace hub {

namespace {

constexpr size_t kMaxRepoSegment = 96;

bool is_repo_char(char c)
{
    return std::isalnum((unsigned char)c) || c == '.' || c == '_' || c == '-';
}

bool is_valid_repo_segment(const std::string &s)
{
    if (s.empty() || s.size() > kMaxRepoSegment) return false;
    if (!std::isalnum((unsigned char)s.front())) return false;
    if (s.find("..") != std::string::npos) return false;
    return std::all_of(s.begin(), s.end(), is_repo_char);
}

bool is_valid_token(const std::string &token)
{
    return std::none_of(token.begin(), token.end(), [](char c) { return std::iscntrl((unsigned char)c) || c == ' '; });
}

// Percent-encode everything except RFC 3986 unreserved characters and '/'.
std::string url_encode_path(const std::string &s)
{
    static const char *kHex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') {
            out += (char)c;
        } else {
            out += '%';
            out += kHex[c >> 4];
            out += kHex[c & 0xF];
        }
    }
    return out;
}

#ifdef _WIN32
// Quote one argument following the MSVCRT CommandLineToArgv rules.
std::string quote_windows_arg(const std::string &arg)
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
#endif

// Run a program directly (no shell) with an explicit argument vector.
// When `out` is non-null the child's stdout is captured into it.
// Returns the child's exit code, or -1 if it could not be run.
int run_process(const std::vector<std::string> &args, std::string *out)
{
    if (args.empty()) return -1;
#ifdef _WIN32
    std::string cmdline;
    for (const auto &a : args) {
        if (!cmdline.empty()) cmdline += ' ';
        cmdline += quote_windows_arg(a);
    }

    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    if (out) {
        if (!CreatePipe(&read_end, &write_end, &sa, 0)) return -1;
        SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = write_end;
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    }
    PROCESS_INFORMATION pi{};
    std::vector<char> buf(cmdline.begin(), cmdline.end());
    buf.push_back('\0');
    const BOOL ok =
        CreateProcessA(nullptr, buf.data(), nullptr, nullptr, out ? TRUE : FALSE, 0, nullptr, nullptr, &si, &pi);
    if (write_end) CloseHandle(write_end);
    if (!ok) {
        if (read_end) CloseHandle(read_end);
        return -1;
    }
    if (out) {
        char buffer[8192];
        DWORD n = 0;
        while (ReadFile(read_end, buffer, sizeof(buffer), &n, nullptr) && n > 0) out->append(buffer, n);
        CloseHandle(read_end);
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)code;
#else
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);

    int fds[2] = {-1, -1};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    if (out) {
        if (pipe(fds) != 0) {
            posix_spawn_file_actions_destroy(&actions);
            return -1;
        }
        fcntl(fds[0], F_SETFD, FD_CLOEXEC);
        fcntl(fds[1], F_SETFD, FD_CLOEXEC);
        posix_spawn_file_actions_adddup2(&actions, fds[1], STDOUT_FILENO);
    }

    pid_t pid = 0;
    const int rc = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (out) close(fds[1]);
    if (rc != 0) {
        if (out) close(fds[0]);
        return -1;
    }
    if (out) {
        char buffer[8192];
        for (;;) {
            const ssize_t n = read(fds[0], buffer, sizeof(buffer));
            if (n > 0) {
                out->append(buffer, (size_t)n);
            } else if (n < 0 && errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        close(fds[0]);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

// Base curl invocation: -q (must be first) ignores any .curlrc.
std::vector<std::string> curl_args(const std::string &token)
{
    std::vector<std::string> args = {"curl", "-q"};
    if (!token.empty()) {
        args.push_back("-H");
        args.push_back("Authorization: Bearer " + token);
    }
    return args;
}

std::string http_get(const std::string &url, const std::string &token)
{
    if (!is_valid_token(token)) return "";
    auto args = curl_args(token);
    args.insert(args.end(), {"-s", "-X", "GET", "--url", url});
    std::string response;
    if (run_process(args, &response) != 0) return "";
    return response;
}

std::string http_head(const std::string &url, const std::string &token)
{
    if (!is_valid_token(token)) return "";
    auto args = curl_args(token);
    args.insert(args.end(), {"-sIL", "--url", url});
    std::string response;
    if (run_process(args, &response) != 0) return "";
    return response;
}

std::string trim(const std::string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool iequals(const std::string &a, const std::string &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
}

// Best-effort total size from response headers (last content-length wins,
// since -L follows redirects). Returns 0 when unknown (chunked encoding).
size_t content_length_of(const std::string &headers)
{
    size_t total = 0;
    std::istringstream hs(headers);
    std::string line;
    while (std::getline(hs, line)) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (iequals(trim(line.substr(0, colon)), "content-length")) {
            total = std::strtoull(trim(line.substr(colon + 1)).c_str(), nullptr, 10);
        }
    }
    return total;
}

size_t file_size_of(const std::string &path)
{
    std::error_code ec;
    return std::filesystem::exists(path, ec) ? (size_t)std::filesystem::file_size(path, ec) : 0;
}

}  // namespace

bool ModelDownloader::is_valid_repo_id(const std::string &repo_id)
{
    const size_t slash = repo_id.find('/');
    if (slash == std::string::npos || repo_id.find('/', slash + 1) != std::string::npos) return false;
    return is_valid_repo_segment(repo_id.substr(0, slash)) && is_valid_repo_segment(repo_id.substr(slash + 1));
}

bool ModelDownloader::is_safe_repo_path(const std::string &path)
{
    if (path.empty() || path.size() > 1024 || path.front() == '/') return false;
    for (char c : path) {
        if (std::iscntrl((unsigned char)c) || c == '\\' || c == ':') return false;
    }
    std::istringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/')) {
        if (part.empty() || part == "." || part == "..") return false;
    }
    return path.back() != '/';
}

ModelDownloader::ModelDownloader()
{
}

std::string ModelDownloader::build_hf_url(const std::string &repo_id, const std::string &filename)
{
    // Convert repo_id to URL format: replace "/" with "/" in URL
    // HuggingFace URL pattern: https://huggingface.co/{repo_id}/resolve/main/{filename}
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + url_encode_path(filename);
}

std::string ModelDownloader::fetch_model_info(const std::string &repo_id, const std::string &token)
{
    if (!is_valid_repo_id(repo_id)) return "";
    return http_get("https://huggingface.co/api/models/" + repo_id, token);
}

bool ModelDownloader::download_file(const std::string &url, const std::string &path, ProgressFn progress,
                                    const std::string &token)
{
    if (path.empty() || !is_valid_token(token)) return false;
    if (url.rfind("https://", 0) != 0 && url.rfind("http://", 0) != 0) return false;

    // Create output directory if needed
    std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) std::filesystem::create_directories(output_path.parent_path());

    // Use curl for download: -L follows redirects (HF resolve → CDN),
    // --retry covers transient failures, -C - resumes partial downloads.
    // Arguments are passed as a vector; no shell ever parses them.
    auto args = curl_args(token);
    args.insert(args.end(), {"-sSL", "--retry", "3", "-C", "-", "-o", path, "--url", url});

    // Best-effort total size via HEAD (HF resolve redirects to the CDN).
    const size_t total = content_length_of(http_head(url, token));

    // Run curl asynchronously so the progress callback can report the real
    // number of bytes written so far instead of a fake 0/0.
    auto fut = std::async(std::launch::async, [&args]() { return run_process(args, nullptr); });

    if (progress) {
        while (fut.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
            progress(file_size_of(path), total);
        }
    }

    const int result = fut.get();

    if (progress) {
        progress(file_size_of(path), total);
    }

    return (result == 0);
}

bool ModelDownloader::download_hf(const std::string &repo_id,
                                 const std::string &output_dir,
                                 const std::string &token,
                                 ProgressFn progress)
{
    if (!is_valid_repo_id(repo_id)) return false;

    // Get the real file list from the HF tree API and keep the files needed
    // to load a model: configs, tokenizer assets and safetensors shards.
    // GGUF/optimizer/training files are skipped (use resolve_hf_gguf for
    // quantized downloads).
    std::vector<std::string> wanted;
    const auto remote = list_hf_files(repo_id, token);
    if (remote.empty()) {
        // API unreachable or empty repo: fall back to the common names so a
        // transient listing failure doesn't break imports entirely.
        wanted = {"config.json", "tokenizer.json", "model.safetensors", "vocab.json"};
    } else {
        auto wanted_name = [](const std::string &p) {
            const size_t slash = p.find_last_of('/');
            const std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
            static const char *kExact[] = {
                "config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "merges.txt",
                "special_tokens_map.json", "added_tokens.json",
                "sentencepiece.model", "tokenizer.model"};
            for (const char *k : kExact) {
                if (base == k) return true;
            }
            const size_t dot = base.find_last_of('.');
            if (dot == std::string::npos) return false;
            const std::string ext = base.substr(dot);
            return ext == ".safetensors" ||
                   (ext == ".json" && base.find("model.safetensors.index") != std::string::npos);
        };
        for (const auto &f : remote) {
            if (wanted_name(f.path)) wanted.push_back(f.path);
        }
    }

    bool downloaded_any = false;
    for (const auto &filename : wanted) {
        std::string url = build_hf_url(repo_id, filename);
        std::string rel = filename;
        // Keep the repo's directory layout for nested files.
        for (char &c : rel) { if (c == '\\') c = '/'; }
        std::string output_path = output_dir + "/" + rel;
        if (std::filesystem::path(output_path).has_parent_path()) {
            std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
        }
        // Individual missing files are tolerated; success requires at least
        // one file to actually land on disk.
        if (download_file(url, output_path, progress, token)) {
            downloaded_any = true;
        }
    }

    return downloaded_any;
}

std::vector<HubFile> ModelDownloader::list_hf_files(const std::string &repo_id,
                                                    const std::string &token)
{
    // GET /api/models/{repo}/tree/main?recursive=true → [{type,path,size,...}]
    if (!is_valid_repo_id(repo_id)) return {};
    const std::string response =
        http_get("https://huggingface.co/api/models/" + repo_id + "/tree/main?recursive=true", token);
    if (response.empty()) return {};

    std::vector<HubFile> files;
    try {
        auto j = nlohmann::json::parse(response);
        if (!j.is_array()) return {};
        for (const auto &entry : j) {
            if (!entry.is_object()) continue;
            if (entry.value("type", "") != "file") continue;
            HubFile f;
            f.path = entry.value("path", "");
            f.size = entry.value("size", (size_t)0);
            if (is_safe_repo_path(f.path)) files.push_back(std::move(f));
        }
    } catch (const std::exception &e) {
        std::cerr << "[Hub] Failed to parse file listing response: " << e.what() << "\n";
        return {};
    }
    return files;
}

bool ModelDownloader::download_hf_file(const std::string &repo_id, const std::string &filename,
                                       const std::string &output_path, const std::string &token,
                                       ProgressFn progress)
{
    if (!is_valid_repo_id(repo_id) || !is_safe_repo_path(filename)) return false;
    std::string url = build_hf_url(repo_id, filename);
    return download_file(url, output_path, progress, token);
}

bool ModelDownloader::resolve_hf_gguf(const std::string &repo_id, const std::string &quant_tag,
                                      std::string &local_path, std::string &chosen_name,
                                      const std::string &token)
{
    if (!is_valid_repo_id(repo_id)) return false;

    auto norm = [](std::string s) {
        for (auto &c : s) {
            c = (char)std::tolower((unsigned char)c);
            if (c == '-') c = '_';
        }
        return s;
    };
    const std::string qn = norm(quant_tag);

    std::vector<HubFile> ggufs;
    for (const auto &f : list_hf_files(repo_id, token)) {
        const std::string lp = norm(f.path);
        if (lp.size() >= 5 && lp.compare(lp.size() - 5, 5, ".gguf") == 0) {
            ggufs.push_back(f);
        }
    }
    if (ggufs.empty()) {
        return false;
    }

    const HubFile *pick = nullptr;
    if (!qn.empty()) {
        for (const auto &f : ggufs) {
            if (norm(f.path).find(qn) != std::string::npos) { pick = &f; break; }
        }
        if (!pick) {
            return false;
        }
    } else {
        static const char *kPrefs[] = {"q4_k_m", "q4_0", "q4_1", "q8_0"};
        for (const char *p : kPrefs) {
            for (const auto &f : ggufs) {
                if (norm(f.path).find(p) != std::string::npos) { pick = &f; break; }
            }
            if (pick) break;
        }
        if (!pick) {
            pick = &*std::min_element(ggufs.begin(), ggufs.end(),
                                      [](const HubFile &a, const HubFile &b) {
                                          return a.size < b.size;
                                      });
        }
    }

    const size_t slash = repo_id.find_last_of('/');
    const std::string base = (slash == std::string::npos) ? repo_id : repo_id.substr(slash + 1);
    local_path = "models/" + base + "/" + pick->path;
    chosen_name = pick->path;

    const bool complete = std::filesystem::exists(local_path) &&
                          (pick->size == 0 ||
                           std::filesystem::file_size(local_path) == pick->size);
    if (!complete && !download_hf_file(repo_id, pick->path, local_path, token)) {
        return false;
    }
    return true;
}

bool ModelDownloader::download_url(const std::string &url,
                                  const std::string &output_path,
                                  ProgressFn progress)
{
    return download_file(url, output_path, progress);
}

}  // namespace hub
}  // namespace serius
