#include "cli/cmd_doctor.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "cli/args_parser.h"
#include "hal/device.h"
#include "io/gguf_loader.h"
#include "io/hf_llama_loader.h"
#include "transformer/llama_model.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#elif defined(__linux__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifndef SERIUS_VERSION
#define SERIUS_VERSION "unknown"
#endif

namespace serius {
namespace cli {

namespace {

struct CheckEntry
{
    std::string level;  // "ok" | "warn" | "info"
    std::string label;
    std::string detail;
};

struct Section
{
    std::string title;
    std::vector<CheckEntry> entries;
};

struct Report
{
    std::vector<Section> sections;
    std::string version = SERIUS_VERSION;

    void add(const std::string &section, const std::string &level, const std::string &label, const std::string &detail)
    {
        for (auto &s : sections) {
            if (s.title == section) {
                s.entries.push_back({level, label, detail});
                return;
            }
        }
        sections.push_back({section, {{level, label, detail}}});
    }
};

void print_icon(const std::string &level)
{
    if (level == "ok") {
        std::cout << "✅ ";
    } else if (level == "warn") {
        std::cout << "⚠️  ";
    } else {
        std::cout << "ℹ️  ";
    }
}

void render_text(const Report &report)
{
    std::cout << "Serius System Diagnostics\n";
    std::cout << "========================\n\n";
    std::cout << "Version: " << report.version << "\n";

    for (const auto &section : report.sections) {
        std::cout << "\n" << section.title << ":\n";
        for (const auto &e : section.entries) {
            print_icon(e.level);
            std::cout << e.label;
            if (!e.detail.empty()) {
                std::cout << ": " << e.detail;
            }
            std::cout << "\n";
        }
    }
}

void render_json(const Report &report)
{
    nlohmann::json root;
    root["version"] = report.version;
    nlohmann::json sections = nlohmann::json::object();
    for (const auto &section : report.sections) {
        nlohmann::json entries = nlohmann::json::array();
        for (const auto &e : section.entries) {
            entries.push_back({{"level", e.level}, {"label", e.label}, {"detail", e.detail}});
        }
        sections[section.title] = entries;
    }
    root["sections"] = sections;
    std::cout << root.dump(2) << "\n";
}

std::string device_type_name(hal::DeviceType t)
{
    switch (t) {
        case hal::DeviceType::CUDA:
            return "CUDA";
        case hal::DeviceType::VULKAN:
            return "Vulkan";
        case hal::DeviceType::METAL:
            return "Metal";
        default:
            return "unknown";
    }
}

/// Total physical RAM in bytes; 0 when unavailable.
uint64_t total_ram_bytes()
{
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return ms.ullTotalPhys;
#elif defined(__linux__)
    struct sysinfo si{};
    if (sysinfo(&si) == 0) return static_cast<uint64_t>(si.totalram) * si.mem_unit;
#elif defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) return mem;
#endif
    return 0;
}

/// Available physical RAM in bytes; 0 when unavailable.
uint64_t available_ram_bytes()
{
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return ms.ullAvailPhys;
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            return std::strtoull(line.c_str() + 13, nullptr, 10) * 1024;
        }
    }
#elif defined(__APPLE__)
#endif
    return 0;
}

uint64_t free_disk_bytes(const std::filesystem::path &dir)
{
    std::error_code ec;
    const auto space = std::filesystem::space(dir, ec);
    if (ec) {
        return 0;
    }
    return static_cast<uint64_t>(space.available);
}

std::string format_bytes(uint64_t bytes)
{
    if (bytes == 0) return "unknown";
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double MiB = 1024.0 * 1024.0;
    char buf[64];
    if (bytes >= GiB) {
        std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / GiB);
    } else {
        std::snprintf(buf, sizeof(buf), "%.0f MiB", static_cast<double>(bytes) / MiB);
    }
    return buf;
}

std::string format_params(double params)
{
    if (params >= 1e9) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1fB", params / 1e9);
        return buf;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0fM", params / 1e6);
    return buf;
}

/// Rough parameter count for a dense LLaMA-style model. MoE experts are not
/// counted (the estimate then reflects the dense-equivalent size only).
double estimate_llama_params(size_t hidden, size_t layers, size_t heads, size_t kv_heads, size_t inter, size_t vocab)
{
    if (hidden == 0 || heads == 0 || layers == 0 || vocab == 0) {
        return 0.0;
    }
    const double kv = kv_heads > 0 ? static_cast<double>(kv_heads) : static_cast<double>(heads);
    const double head_dim = static_cast<double>(hidden) / static_cast<double>(heads);
    // Per layer: q (h*h) + k (h*kv*d) + v (h*kv*d) + o (h*h) + SwiGLU ffn (3*h*inter)
    const double attn = static_cast<double>(hidden) *
                        (static_cast<double>(heads) * head_dim + 2.0 * kv * head_dim + static_cast<double>(hidden));
    const double ffn = 3.0 * static_cast<double>(hidden) * static_cast<double>(inter);
    const double embed = 2.0 * static_cast<double>(vocab) * static_cast<double>(hidden);
    return embed + static_cast<double>(layers) * (attn + ffn);
}

/// True when the TCP port can be bound on localhost (i.e. likely free).
bool port_bindable(uint16_t port)
{
#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return false;
    }
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        WSACleanup();
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    const bool ok = bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    closesocket(s);
    WSACleanup();
    return ok;
#else
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    const bool ok = ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return ok;
#endif
}

void inspect_gguf(const std::filesystem::path &file, uint64_t vram_total, Report &report)
{
    const std::string path = file.string();
    const uint64_t size = std::filesystem::file_size(file);
    report.add("Model", "ok", "File", file.filename().string() + " (" + format_bytes(size) + ")");

    const io::GGUFMeta meta = io::load_gguf_metadata(path);
    if (!meta.valid) {
        report.add("Model", "warn", "GGUF", "not a valid GGUF file (bad magic or version)");
        return;
    }
    report.add("Model", "ok", "GGUF",
               "version " + std::to_string(meta.version) + ", " + std::to_string(meta.n_tensors) + " tensors, " +
                   std::to_string(meta.n_kv) + " metadata entries");

    const std::string arch = io::gguf_architecture(path);
    if (!arch.empty()) {
        report.add("Model", "info", "Architecture", arch);
    }

    const auto mc = io::gguf_to_model_config(path);
    if (mc.vocab_size > 0 && mc.d_model > 0 && mc.n_layers > 0) {
        std::string dims = "vocab=" + std::to_string(mc.vocab_size) + ", hidden=" + std::to_string(mc.d_model) +
                           ", layers=" + std::to_string(mc.n_layers) + ", heads=" + std::to_string(mc.n_heads) + "/" +
                           std::to_string(mc.n_kv_heads) + " (q/kv), ffn=" + std::to_string(mc.d_ff) +
                           ", ctx=" + std::to_string(mc.max_seq_len);
        report.add("Model", "info", "Dimensions", dims);
    }

    // Exact parameter/byte accounting straight from the tensor infos.
    const io::GGUFWeightStats stats = io::gguf_weight_stats(path);
    if (stats.valid && stats.total_params > 0) {
        const double bpw = stats.total_bytes > 0
                               ? static_cast<double>(stats.total_bytes) * 8.0 /
                                     static_cast<double>(stats.total_params)
                               : 0.0;
        std::string bpw_label = "unknown quantization";
        if (bpw > 0) {
            if (bpw < 3.5) {
                bpw_label = "very aggressive quant (IQ/Q2-class)";
            } else if (bpw < 6.5) {
                bpw_label = "Q4-class quantization";
            } else if (bpw < 8.5) {
                bpw_label = "Q5/Q6-class quantization";
            } else {
                bpw_label = "Q8/FP16-class weights";
            }
        }
        std::string detail = "~" + format_params(static_cast<double>(stats.total_params)) +
                             " params (exact, from tensor shapes), payload " + format_bytes(stats.total_bytes);
        if (bpw > 0) {
            detail += ", ~" + std::to_string(static_cast<int>(bpw * 10) / 10.0) + " bits/weight → " + bpw_label;
        }
        report.add("Model", "info", "Weights", detail);
    } else if (stats.valid) {
        report.add("Model", "warn", "Weights", "tensor infos unreadable — stats unavailable");
    }

    // Fit check against VRAM then RAM
    if (vram_total > 0 && size <= static_cast<uint64_t>(vram_total * 0.8)) {
        report.add("Model", "ok", "Fit", "file fits in GPU VRAM (" + format_bytes(vram_total) + ")");
    } else {
        const uint64_t avail = available_ram_bytes();
        if (avail > 0 && size <= static_cast<uint64_t>(avail * 0.8)) {
            report.add("Model", "ok", "Fit",
                       "exceeds VRAM budget — will run on CPU RAM (" + format_bytes(avail) + " available)");
        } else if (avail > 0) {
            report.add("Model", "warn", "Fit",
                       "file may not fit in available memory (" + format_bytes(avail) + " free RAM)");
        }
    }
}

void inspect_model_path(const std::string &model_arg, uint64_t vram_total, Report &report)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p(model_arg);

    if (!fs::exists(p, ec)) {
        report.add("Model", "warn", "Path", model_arg + " — does not exist");
        return;
    }

    if (fs::is_regular_file(p)) {
        const std::string ext = p.extension().string();
        if (ext == ".gguf") {
            inspect_gguf(p, vram_total, report);
        } else {
            report.add("Model", "warn", "File",
                       p.filename().string() + " — unsupported format (expected .gguf or model dir)");
        }
        return;
    }

    if (fs::is_directory(p)) {
        report.add("Model", "ok", "Path", model_arg + " (directory)");

        // JSON config (LLaMA-style)
        const fs::path cfg = p / "config.json";
        if (fs::exists(cfg, ec)) {
            const auto lc = io::load_llama_config(cfg.string());
            if (lc.hidden_size > 0 && lc.num_hidden_layers > 0) {
                std::string dims =
                    "vocab=" + std::to_string(lc.vocab_size) + ", hidden=" + std::to_string(lc.hidden_size) +
                    ", layers=" + std::to_string(lc.num_hidden_layers) +
                    ", heads=" + std::to_string(lc.num_attention_heads) + "/" + std::to_string(lc.num_key_value_heads) +
                    " (q/kv), ffn=" + std::to_string(lc.intermediate_size) +
                    ", ctx=" + std::to_string(lc.max_position_embeddings);
                report.add("Model", "info", "Config", dims);
                const double params =
                    estimate_llama_params(lc.hidden_size, lc.num_hidden_layers, lc.num_attention_heads,
                                          lc.num_key_value_heads, lc.intermediate_size, lc.vocab_size);
                if (params > 0) {
                    report.add("Model", "info", "Estimate",
                               "~" + format_params(params) + " params (dense approx, from config.json)");
                }
            }
        } else {
            report.add("Model", "info", "Config", "no config.json — will rely on GGUF metadata");
        }

        // GGUF files inside (weights may be sharded)
        uint64_t total_gguf = 0;
        int n_gguf = 0;
        std::vector<std::pair<std::string, uint64_t>> ggufs;
        for (const auto &entry : fs::directory_iterator(p, ec)) {
            if (!entry.is_regular_file(ec)) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.size() >= 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
                const uint64_t sz = entry.file_size(ec);
                ggufs.push_back({name, sz});
                total_gguf += sz;
                ++n_gguf;
            }
        }
        if (n_gguf == 0) {
            report.add("Model", "warn", "Weights", "no .gguf files found in directory");
        } else {
            for (const auto &[name, sz] : ggufs) {
                report.add("Model", "ok", "Weights", name + " (" + format_bytes(sz) + ")");
            }
            if (n_gguf > 1) {
                report.add("Model", "info", "Weights",
                           std::to_string(n_gguf) + " GGUF files, " + format_bytes(total_gguf) + " total");
            }

            // Inspect the first GGUF for metadata details
            const fs::path first = p / ggufs.front().first;
            const io::GGUFMeta meta = io::load_gguf_metadata(first.string());
            if (meta.valid) {
                const std::string arch = io::gguf_architecture(first.string());
                std::string detail =
                    "v" + std::to_string(meta.version) + ", " + std::to_string(meta.n_tensors) + " tensors";
                if (!arch.empty()) {
                    detail += ", arch=" + arch;
                }
                report.add("Model", "info", "GGUF meta", detail);
            }

            // Exact weight stats aggregated across all GGUF files (covers shards)
            uint64_t params_sum = 0;
            uint64_t bytes_sum = 0;
            for (const auto &[name, sz] : ggufs) {
                const io::GGUFWeightStats st = io::gguf_weight_stats((p / name).string());
                if (st.valid) {
                    params_sum += st.total_params;
                    bytes_sum += st.total_bytes;
                }
            }
            if (params_sum > 0) {
                const double bpw =
                    bytes_sum > 0 ? static_cast<double>(bytes_sum) * 8.0 / static_cast<double>(params_sum) : 0.0;
                std::string detail = "~" + format_params(static_cast<double>(params_sum)) +
                                     " params (exact, from tensor shapes)";
                if (bpw > 0) {
                    detail += ", ~" + std::to_string(static_cast<int>(bpw * 10) / 10.0) + " bits/weight";
                }
                report.add("Model", "info", "Weights payload", detail);
            }

            if (vram_total > 0 && total_gguf <= static_cast<uint64_t>(vram_total * 0.8)) {
                report.add("Model", "ok", "Fit", "weights fit in GPU VRAM (" + format_bytes(vram_total) + ")");
            } else {
                const uint64_t avail = available_ram_bytes();
                if (avail > 0 && total_gguf <= static_cast<uint64_t>(avail * 0.8)) {
                    report.add("Model", "ok", "Fit",
                               "exceeds VRAM budget — will run on CPU RAM (" + format_bytes(avail) + " available)");
                } else if (avail > 0) {
                    report.add("Model", "warn", "Fit",
                               "weights may not fit in available memory (" + format_bytes(avail) + " free RAM)");
                }
            }
        }
        return;
    }

    report.add("Model", "warn", "Path", model_arg + " — not a file or directory");
}

}  // namespace

int cmd_doctor(int argc, char **argv)
{
    ArgsParser parser("serius doctor", "System diagnostics and deployment readiness check");

    parser.add_flag({"help", "h", "Show this help message", false, ""});
    parser.add_flag({"model", "m", "Inspect a model file (.gguf) or directory (config.json + weights)", true, ""});
    parser.add_flag({"port", "", "Check if a TCP port is free for the server (e.g. 8080)", true, ""});
    parser.add_flag({"json", "", "Emit the report as JSON", false, ""});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    const bool json_out = parser.has("json");
    Report report;

    // CPU
    const unsigned cores = std::thread::hardware_concurrency();
    report.add("System", "ok", "CPU", std::to_string(cores ? cores : 0) + " logical cores");

    // RAM
    const uint64_t total = total_ram_bytes();
    const uint64_t avail = available_ram_bytes();
    if (total > 0) {
        std::string detail = format_bytes(total);
        if (avail > 0) detail += " total, " + format_bytes(avail) + " available";
        report.add("System", "ok", "RAM", detail);
    } else {
        report.add("System", "warn", "RAM", "could not query system memory");
    }

    // Disk space on cwd
    const uint64_t disk_free = free_disk_bytes(".");
    if (disk_free > 0) {
        report.add("System", "ok", "Disk", format_bytes(disk_free) + " free in current directory");
    } else {
        report.add("System", "warn", "Disk", "could not query free space");
    }

    // GPU via HAL backend registry
    uint64_t vram_total = 0;
    hal::IBackend *gpu = hal::get_gpu_backend();
    if (gpu && gpu->is_available()) {
        std::string detail = device_type_name(gpu->get_type()) + " backend available";
#ifdef SERIUS_USE_CUDA
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                vram_total = total_b;
                detail += " — " + std::string(prop.name);
                detail += " (" + std::to_string(device_count) + " device" + (device_count > 1 ? "s" : "") + ", " +
                          format_bytes(total_b) + " VRAM, " + format_bytes(free_b) + " free)";
            }
        }
#endif
        report.add("System", "ok", "GPU", detail);
    } else {
        report.add("System", "warn", "GPU", "no CUDA backend available — CPU inference only");
    }

    // Build flags
    {
        std::string cuda = "disabled", nccl = "disabled", mpi = "disabled";
#ifdef SERIUS_USE_CUDA
        cuda = "enabled";
#endif
#ifdef SERIUS_USE_NCCL
        nccl = "enabled";
#endif
#ifdef SERIUS_USE_MPI
        mpi = "enabled";
#endif
        report.add("Build", "info", "Flags", "C++20, CUDA " + cuda + ", NCCL " + nccl + ", MPI " + mpi);
    }

    // Readiness: recommended model envelope
    {
        std::string gpu_rec;
        if (vram_total > 0) {
            // Q4_K_M-class weights ≈ 0.6 bytes/param + overhead; keep 80% usable.
            const double max_params = static_cast<double>(vram_total) * 0.8 / 0.6;
            gpu_rec = "GPU can comfortably run up to ~" + format_params(max_params) +
                      " params (Q4-class). Examples: 7-8B full GPU, 14B tight, 32B needs ~24GB+.";
        } else {
            gpu_rec = "No GPU detected — CPU-only inference.";
        }
        report.add("Readiness", "info", "GPU budget", gpu_rec);

        if (avail > 0) {
            const double max_params_cpu = static_cast<double>(avail) * 0.8 / 1.2;
            report.add("Readiness", "info", "CPU budget",
                       "with " + format_bytes(avail) + " free RAM, CPU inference suits up to ~" +
                           format_params(max_params_cpu) +
                           " params (Q4-class, slower tok/s). Smaller models (<4B) are practical.");
        }
    }

    // Model inspection
    const std::string model_arg = parser.get("model");
    if (!model_arg.empty()) {
        inspect_model_path(model_arg, vram_total, report);
    }

    // Port check
    if (parser.has("port")) {
        const int port = parser.get_int("port", 0);
        if (port <= 0 || port > 65535) {
            report.add("Network", "warn", "Port", "invalid port number: " + std::to_string(port));
        } else if (port_bindable(static_cast<uint16_t>(port))) {
            report.add("Network", "ok", "Port",
                       std::to_string(port) + " is free on 127.0.0.1 — ready for `serius serve`");
        } else {
            report.add("Network", "warn", "Port",
                       std::to_string(port) +
                           " is already in use — pick another or stop the "
                           "existing service");
        }
    }

    if (json_out) {
        render_json(report);
    } else {
        render_text(report);
        std::cout << "\n";
        std::cout << "Tip: use `serius doctor --model <path.gguf|dir> --port 8080` for a "
                     "deployment readiness check.\n";
    }

    return 0;
}

}  // namespace cli
}  // namespace serius
