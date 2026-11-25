#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <cstdio>
#include <dirent.h>
#include <set>
#include <string>
#include <utility>
#endif

#include "../common/thread_pool.h"
#include "../hal/device.h"
#include "blas.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef SERIUS_USE_CUDA
#include "matmul_cuda.h"
#endif

// Platform specific includes for CPUID
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace serius {
namespace core {
namespace blas {

using core::Tensor;

// Cache CPU info
static CpuInfo internal_cpu_info = {false, false, false};
static bool cpu_info_initialized = false;

// Size-based dispatch thresholds (empirically calibrated)
// For small matrices, SIMD setup overhead exceeds the computation gain
constexpr size_t SIMD_THRESHOLD_SMALL = 128ULL * 128ULL * 128ULL;   // ~2M FLOPS
constexpr size_t SIMD_THRESHOLD_MEDIUM = 256ULL * 256ULL * 256ULL;  // ~16M FLOPS

CpuInfo get_cpu_info()
{
    if (cpu_info_initialized) {
        return internal_cpu_info;
    }

    bool has_avx2 = false;
    bool has_avx512f = false;
    bool has_avx512_vnni = false;

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    const int FEATURE_LEAF = 7;
    const int AVX2_BIT = 5;
    const int AVX512F_BIT = 16;
    const int AVX512VNNI_BIT = 11;  // ECX bit 11

    if (nIds >= FEATURE_LEAF) {
        __cpuidex(cpuInfo, FEATURE_LEAF, 0);
        has_avx2 = (cpuInfo[1] & (1 << AVX2_BIT)) != 0;        // EBX bit 5
        has_avx512f = (cpuInfo[1] & (1 << AVX512F_BIT)) != 0;  // EBX bit 16
        has_avx512_vnni = (cpuInfo[2] & (1 << AVX512VNNI_BIT)) != 0;  // ECX bit 11
    }
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    unsigned int eax, ebx, ecx, edx;
    const int FEATURE_LEAF = 7;
    const int AVX2_BIT = 5;
    const int AVX512F_BIT = 16;
    const int AVX512VNNI_BIT = 11;  // ECX bit 11
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        if (eax >= FEATURE_LEAF) {
            __get_cpuid_count(FEATURE_LEAF, 0, &eax, &ebx, &ecx, &edx);
            has_avx2 = (ebx & (1 << AVX2_BIT)) != 0;
            has_avx512f = (ebx & (1 << AVX512F_BIT)) != 0;
            has_avx512_vnni = (ecx & (1 << AVX512VNNI_BIT)) != 0;
        }
    }
#endif

    internal_cpu_info.has_avx2 = has_avx2;
    internal_cpu_info.has_avx512f = has_avx512f;
    internal_cpu_info.has_avx512_vnni = has_avx512_vnni;
    cpu_info_initialized = true;

    return internal_cpu_info;
}

unsigned physical_core_count()
{
    unsigned cores = 0;

#if defined(_WIN32)
    // One SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX per physical core.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<uint8_t> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                                             &len)) {
            auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
            for (DWORD off = 0; off < len;) {
                if (info->Relationship == RelationProcessorCore) {
                    ++cores;
                }
                off += info->Size;
                info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                    reinterpret_cast<uint8_t *>(info) + info->Size);
            }
        }
    }
#elif defined(__APPLE__)
    int value = 0;
    size_t sz = sizeof(value);
    if (sysctlbyname("hw.physicalcpu", &value, &sz, nullptr, 0) == 0 && value > 0) {
        cores = static_cast<unsigned>(value);
    }
#elif defined(__linux__)
    // Unique (package_id, core_id) pairs under /sys topology.
    std::set<std::pair<int, int>> seen;
    if (DIR *dir = opendir("/sys/devices/system/cpu")) {
        while (dirent *ent = readdir(dir)) {
            const std::string name = ent->d_name;
            if (name.rfind("cpu", 0) != 0 || name.find_first_not_of("cpu0123456789") != std::string::npos) {
                continue;
            }
            auto read_int = [&](const char *leaf) -> int {
                const std::string path = "/sys/devices/system/cpu/" + name + "/topology/" + leaf;
                if (FILE *f = fopen(path.c_str(), "r")) {
                    int v = -1;
                    if (fscanf(f, "%d", &v) == 1) {
                        fclose(f);
                        return v;
                    }
                    fclose(f);
                }
                return -1;
            };
            const int pkg = read_int("physical_package_id");
            const int core = read_int("core_id");
            if (pkg >= 0 && core >= 0) {
                seen.emplace(pkg, core);
            }
        }
        closedir(dir);
    }
    cores = static_cast<unsigned>(seen.size());
#endif

    if (cores == 0) {
        cores = std::thread::hardware_concurrency();
    }
    if (cores == 0) {
        cores = 1;
    }
    return cores;
}

void configure_omp_threads()
{
#if defined(_OPENMP)
    static const bool env_set = [] {
#if defined(_MSC_VER)
        char buf[16];
        size_t len = 0;
        return getenv_s(&len, buf, sizeof(buf), "OMP_NUM_THREADS") == 0 && len > 0;
#else
        return std::getenv("OMP_NUM_THREADS") != nullptr;
#endif
    }();
    if (!env_set) {
        const unsigned cores = physical_core_count();
        const unsigned hw = std::thread::hardware_concurrency();
        // Never exceed the logical CPU count OpenMP would pick anyway.
        const unsigned target = (hw > 0 && hw < cores) ? hw : cores;
        omp_set_num_threads(static_cast<int>(target));
    }
#else
    (void)physical_core_count;
#endif
}

// --- Main Dispatcher ---

// Helper for CPU matmul dispatch
static void dispatch_cpu_matmul(const Tensor &A, const Tensor &B, Tensor &C)
{
    if (A.dtype() != hal::DType::Float32) {
        throw std::runtime_error("BLAS::matmul CPU path only supports Float32");
    }

    const float *a_ptr = A.data();
    const float *b_ptr = B.data();
    float *c_ptr = C.data();

    size_t rows_m = A.shape()[0];
    size_t inner_k = A.shape()[1];
    size_t cols_n = B.shape()[1];

    size_t lda = A.strides()[0];
    size_t ldb = B.strides()[0];
    size_t ldc = C.strides()[0];

    auto info = get_cpu_info();
    size_t problem_size = rows_m * cols_n * inner_k;

    if (rows_m == 1) {
        if (info.has_avx512f) {
            matmul_avx512(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else if (info.has_avx2) {
            matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else {
            matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        }
        return;
    }

    if (problem_size < SIMD_THRESHOLD_SMALL) {
        matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
    } else if (problem_size < SIMD_THRESHOLD_MEDIUM) {
        if (info.has_avx2) {
            matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else {
            matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        }
    } else {
        if (info.has_avx512f) {
            matmul_avx512(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else if (info.has_avx2) {
            matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else {
            matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        }
    }
}

void matmul(const Tensor &A, const Tensor &B, Tensor &C, void *stream)
{
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }

    if (A.shape()[1] != B.shape()[0]) {
        throw std::invalid_argument("Incompatible shapes for matmul");
    }

    if (C.shape()[0] != A.shape()[0] || C.shape()[1] != B.shape()[1]) {
        throw std::invalid_argument("Output tensor C has incorrect shape");
    }

#ifdef SERIUS_USE_CUDA
    hal::IBackend *backend = hal::get_backend(hal::DeviceType::CUDA);
    bool any_on_gpu =
        (A.is_gpu_dirty() || B.is_gpu_dirty() || C.is_gpu_dirty() || A.location() == Tensor::DataLocation::GPU);
    size_t problem_size = A.shape()[0] * B.shape()[1] * A.shape()[1];

    // Non-Float32 dtypes have no CPU kernel: always prefer CUDA when the
    // backend is available (same rule as matvec/batched_matmul below).
    if (backend && backend->is_available() &&
        (problem_size > 1000000 || any_on_gpu || A.dtype() != hal::DType::Float32)) {
        A.syncToDevice();
        B.syncToDevice();
        C.syncToDevice();

        // A CPU stream must never reach the CUDA path as a cudaStream_t.
        void *cuda_stream = hal::is_cpu_stream(stream) ? nullptr : stream;
        matmul_cuda(A.raw_data_gpu(), B.raw_data_gpu(), C.raw_data_gpu(), A.shape()[0], B.shape()[1], A.shape()[1],
                    A.strides()[0], B.strides()[0], C.strides()[0], A.dtype(), cuda_stream);
        C.mark_dirty(Tensor::DataLocation::GPU);
        return;
    }

    // Fallback to CPU: ensure host data is current
    A.syncToHost();
    B.syncToHost();
    C.syncToHost();
#endif

    if (stream != nullptr && hal::is_cpu_stream(stream)) {
        // Enqueue the whole GEMM on the CPU stream; caller must sync_stream
        // before reading C and keep A/B/C alive until then (same contract
        // as CUDA async kernels, which also capture raw pointers).
        hal::cpu_stream_enqueue(stream, [&A, &B, &C] {
            dispatch_cpu_matmul(A, B, C);
            C.mark_dirty(Tensor::DataLocation::CPU);
        });
        return;
    }

    dispatch_cpu_matmul(A, B, C);
    C.mark_dirty(Tensor::DataLocation::CPU);
}

void matvec(const Tensor &mat_a, const Tensor &vec_x, Tensor &vec_y)
{
    if (mat_a.ndim() != 2 || vec_x.ndim() != 1 || vec_y.ndim() != 1) {
        throw std::invalid_argument("matvec expects A(2D), x(1D), y(1D)");
    }

    size_t rows_m = mat_a.shape()[0];
    size_t inner_k = mat_a.shape()[1];
    if (vec_x.shape()[0] != inner_k) {
        throw std::invalid_argument("matvec dimension mismatch: A.cols != x.rows");
    }
    size_t cols_n = 1;

    if (vec_y.shape()[0] != rows_m) {
        throw std::invalid_argument("matvec output tensor y must have shape [M]");
    }

    size_t lda = mat_a.strides()[0];
    size_t ldx = vec_x.strides()[0];
    size_t ldy = vec_y.strides()[0];

#ifdef SERIUS_USE_CUDA
    hal::IBackend *gpu_backend = hal::get_gpu_backend();
    bool any_on_gpu = (mat_a.is_gpu_dirty() || vec_x.is_gpu_dirty() || vec_y.is_gpu_dirty());
    if (gpu_backend && gpu_backend->is_available() &&
        (rows_m * inner_k > 1000000 || mat_a.dtype() != hal::DType::Float32 || any_on_gpu)) {
        mat_a.syncToDevice();
        vec_x.syncToDevice();
        vec_y.syncToDevice();

        matmul_cuda(mat_a.raw_data_gpu(), vec_x.raw_data_gpu(), vec_y.raw_data_gpu(), rows_m, 1, inner_k, lda, 1, 1,
                    mat_a.dtype());

        vec_y.mark_dirty(Tensor::DataLocation::GPU);
        return;
    }

    // Fallback to CPU: ensure host data is current
    mat_a.syncToHost();
    vec_x.syncToHost();
    vec_y.syncToHost();
#endif

    if (mat_a.dtype() != hal::DType::Float32) {
        throw std::runtime_error("BLAS::matvec CPU path only supports Float32");
    }

    const float *a_ptr = mat_a.data();
    const float *x_ptr = vec_x.data();
    float *y_ptr = vec_y.data();

    auto info = get_cpu_info();

    if (info.has_avx512f) {
        matmul_avx512(a_ptr, x_ptr, y_ptr, rows_m, cols_n, inner_k, lda, ldx, ldy);
    } else if (info.has_avx2) {
        matmul_avx2(a_ptr, x_ptr, y_ptr, rows_m, cols_n, inner_k, lda, ldx, ldy);
    } else {
        matmul_scalar(a_ptr, x_ptr, y_ptr, rows_m, cols_n, inner_k, lda, ldx, ldy);
    }
}

void batched_matmul(const Tensor &mat_a, const Tensor &mat_b, Tensor &mat_c, void *stream)
{
    if (mat_a.ndim() != 3 || mat_c.ndim() != 3) {
        throw std::invalid_argument("batched_matmul expects A and C to be 3D tensors");
    }
    if (mat_b.ndim() != 2 && mat_b.ndim() != 3) {
        throw std::invalid_argument("batched_matmul expects B to be 2D or 3D tensor");
    }

    size_t batch_size = mat_a.shape()[0];
    size_t rows_m = mat_a.shape()[1];
    size_t inner_k = mat_a.shape()[2];

    size_t cols_n = (mat_b.ndim() == 3) ? mat_b.shape()[2] : mat_b.shape()[1];
    if ((mat_b.ndim() == 3 && mat_b.shape()[0] != batch_size) || (mat_b.ndim() == 3 && mat_b.shape()[1] != inner_k) ||
        (mat_b.ndim() == 2 && mat_b.shape()[0] != inner_k)) {
        throw std::invalid_argument("batched_matmul dimension mismatch on B");
    }

    if (mat_c.shape()[0] != batch_size || mat_c.shape()[1] != rows_m || mat_c.shape()[2] != cols_n) {
        throw std::invalid_argument("batched_matmul output tensor C shape mismatch");
    }

    size_t lda = mat_a.strides()[1];
    size_t ldb = (mat_b.ndim() == 3) ? mat_b.strides()[1] : mat_b.strides()[0];
    size_t ldc = mat_c.strides()[1];

    auto info = get_cpu_info();

#ifdef SERIUS_USE_CUDA
    size_t total_flops = 2 * batch_size * rows_m * cols_n * inner_k;
    hal::IBackend *backend = hal::get_backend(hal::DeviceType::CUDA);

    bool any_on_gpu = (mat_a.is_gpu_dirty() || mat_b.is_gpu_dirty() || mat_c.is_gpu_dirty());

    if (backend && backend->is_available() &&
        (total_flops > 1000000 || mat_a.dtype() != hal::DType::Float32 || any_on_gpu)) {
        // CUDA kernels for matmul currently assume row-major layout with inner stride 1.
        // We ensure contiguity here. NOTE: `cond ? mat_a : mat_a.contiguous()`
        // yields a prvalue and would deep-copy the tensor (cudaMalloc + sync
        // D2D memcpy) on every call — use storage + pointer instead.
        Tensor a_storage, b_storage;
        const Tensor *a_ptr = &mat_a;
        const Tensor *b_ptr = &mat_b;
        if (!mat_a.is_contiguous()) {
            a_storage = mat_a.contiguous();
            a_ptr = &a_storage;
        }
        if (!mat_b.is_contiguous()) {
            b_storage = mat_b.contiguous();
            b_ptr = &b_storage;
        }
        const Tensor &a_cont = *a_ptr;
        const Tensor &b_cont = *b_ptr;

        a_cont.syncToDevice();
        b_cont.syncToDevice();
        mat_c.syncToDevice();

        size_t lda_c = a_cont.strides()[1];
        size_t ldb_c = (b_cont.ndim() == 3) ? b_cont.strides()[1] : b_cont.strides()[0];
        size_t ldc_c = mat_c.strides()[1];

        size_t stride_a = a_cont.strides()[0];
        size_t stride_b = (b_cont.ndim() == 3) ? b_cont.strides()[0] : 0;
        size_t stride_c = mat_c.strides()[0];

        void *cuda_stream = hal::is_cpu_stream(stream) ? nullptr : stream;
        batched_matmul_cuda(a_cont.raw_data_gpu(), b_cont.raw_data_gpu(), mat_c.raw_data_gpu(), batch_size, rows_m,
                            cols_n, inner_k, lda_c, ldb_c, ldc_c, stride_a, stride_b, stride_c, a_cont.dtype(),
                            cuda_stream);

        mat_c.mark_dirty(Tensor::DataLocation::GPU);
        return;
    }

    // Fallback to CPU: ensure host data is current
    mat_a.syncToHost();
    mat_b.syncToHost();
    mat_c.syncToHost();
#endif

    if (mat_a.dtype() != hal::DType::Float32) {
        throw std::runtime_error("BLAS::batched_matmul CPU path only supports Float32");
    }

    auto run_cpu = [&mat_a, &mat_b, &mat_c, batch_size, rows_m, cols_n, inner_k, lda, ldb, ldc, info]() {
#pragma omp parallel for
        for (int64_t b = 0; b < static_cast<int64_t>(batch_size); ++b) {
            size_t batch_idx = static_cast<size_t>(b);
            const float *a_ptr = mat_a.data() + (batch_idx * mat_a.strides()[0]);
            const float *b_ptr = (mat_b.ndim() == 3) ? (mat_b.data() + (batch_idx * mat_b.strides()[0])) : mat_b.data();
            float *c_ptr = mat_c.data() + (batch_idx * mat_c.strides()[0]);

            size_t problem_size = rows_m * cols_n * inner_k;

            if (problem_size < SIMD_THRESHOLD_SMALL) {
                matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
            } else if (problem_size < SIMD_THRESHOLD_MEDIUM) {
                if (info.has_avx2) {
                    matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
                } else {
                    matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
                }
            } else {
                if (info.has_avx512f) {
                    matmul_avx512(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
                } else if (info.has_avx2) {
                    matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
                } else {
                    matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
                }
            }
        }
        mat_c.mark_dirty(Tensor::DataLocation::CPU);
    };

    if (stream != nullptr && hal::is_cpu_stream(stream)) {
        hal::cpu_stream_enqueue(stream, run_cpu);
    } else {
        run_cpu();
    }
}

// --- Convenience Wrappers ---

// Transposed-weight GEMV dispatcher: y = W[N,K] * x[K]
void matvec_t(const float *W, const float *x, float *y, size_t N, size_t K)
{
    auto info = get_cpu_info();
    if (info.has_avx512f) {
        matvec_t_avx512(W, x, y, N, K);
    } else if (info.has_avx2) {
        matvec_t_avx2(W, x, y, N, K);
    } else {
        matvec_t_scalar(W, x, y, N, K);
    }
}

Tensor matmul(const Tensor &mat_a, const Tensor &mat_b, void *stream)
{
    if (mat_a.ndim() != 2 || mat_b.ndim() != 2) {
        throw std::invalid_argument("matmul supports only 2D tensors");
    }
    size_t rows_m = mat_a.shape()[0];
    size_t inner_k = mat_a.shape()[1];
    if (mat_b.shape()[0] != inner_k) {
        throw std::invalid_argument("matmul dimension mismatch: A.cols != B.rows");
    }
    size_t cols_n = mat_b.shape()[1];

    Tensor mat_c({rows_m, cols_n}, mat_a.dtype());
    matmul(mat_a, mat_b, mat_c, stream);
    return mat_c;
}

Tensor matvec(const Tensor &mat_a, const Tensor &vec_x)
{
    if (mat_a.ndim() != 2 || vec_x.ndim() != 1) {
        throw std::invalid_argument("matvec expects A(2D), x(1D)");
    }

    Tensor vec_y({mat_a.shape()[0]}, mat_a.dtype());
    matvec(mat_a, vec_x, vec_y);
    return vec_y;
}

Tensor batched_matmul(const Tensor &A, const Tensor &B, void *stream)
{
    size_t batch_size = A.shape()[0];
    size_t rows_m = A.shape()[1];
    size_t cols_n = (B.ndim() == 3) ? B.shape()[2] : B.shape()[1];

    Tensor C({batch_size, rows_m, cols_n}, A.dtype());
    batched_matmul(A, B, C, stream);
    return C;
}

}  // namespace blas
}  // namespace core
}  // namespace serius
