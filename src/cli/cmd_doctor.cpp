#include "cli/cmd_doctor.h"
#include "cli/args_parser.h"
#include "hal/device.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
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

void print_ok(const std::string &label, const std::string &detail)
{
    std::cout << "✅ " << label << ": " << detail << "\n";
}

void print_warn(const std::string &label, const std::string &detail)
{
    std::cout << "⚠️  " << label << ": " << detail << "\n";
}

std::string device_type_name(hal::DeviceType t)
{
    switch (t) {
        case hal::DeviceType::CUDA: return "CUDA";
        case hal::DeviceType::VULKAN: return "Vulkan";
        case hal::DeviceType::METAL: return "Metal";
        default: return "unknown";
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
    // macOS: vm_statistics via host_statistics64 is the accurate path;
    // approximate with total minus wired+active would need mach headers,
    // so report total only (handled by caller fallback).
#endif
    return 0;
}

std::string format_bytes(uint64_t bytes)
{
    if (bytes == 0) return "unknown";
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(bytes) / GiB);
    return buf;
}

}  // namespace

int cmd_doctor(int argc, char **argv)
{
    ArgsParser parser("serius doctor", "System diagnostics");

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::cout << "Serius System Diagnostics\n";
    std::cout << "========================\n\n";
    std::cout << "Version: " << SERIUS_VERSION << "\n\n";

    // CPU
    const unsigned cores = std::thread::hardware_concurrency();
    print_ok("CPU", std::to_string(cores ? cores : 0) + " logical cores");

    // RAM
    const uint64_t total = total_ram_bytes();
    const uint64_t avail = available_ram_bytes();
    if (total > 0) {
        std::string detail = format_bytes(total);
        if (avail > 0) detail += " total, " + format_bytes(avail) + " available";
        print_ok("RAM", detail);
    } else {
        print_warn("RAM", "could not query system memory");
    }

    // GPU via HAL backend registry (CUDA only; Vulkan/Metal not implemented)
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
                detail += " — " + std::string(prop.name);
                detail += " (" + std::to_string(device_count) + " device" +
                          (device_count > 1 ? "s" : "") + ", " +
                          format_bytes(total_b) + " VRAM)";
            }
        }
#endif
        print_ok("GPU", detail);
    } else {
        print_warn("GPU", "no CUDA backend available — CPU inference only");
    }

    // Build flags
    std::cout << "\nBuild:\n";
    std::cout << "  C++ standard : C++20\n";
#ifdef SERIUS_USE_CUDA
    std::cout << "  CUDA         : enabled\n";
#else
    std::cout << "  CUDA         : disabled\n";
#endif
#ifdef SERIUS_USE_NCCL
    std::cout << "  NCCL         : enabled\n";
#endif
#ifdef SERIUS_USE_MPI
    std::cout << "  MPI          : enabled\n";
#endif

    return 0;
}

}  // namespace cli
}  // namespace serius
