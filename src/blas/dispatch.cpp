#include <stdexcept>

#include "../hal/device.h"
#include "blas.h"

#ifdef SERIUS_USE_CUDA
#include "matmul_cuda.h"
#endif

// Platform specific includes for CPUID
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace serius {
namespace blas {

// Cache CPU info
static CpuInfo internal_cpu_info = {false, false};
static bool cpu_info_initialized = false;

CpuInfo get_cpu_info()
{
    if (cpu_info_initialized) {
        return internal_cpu_info;
    }

    bool has_avx2 = false;
    bool has_avx512f = false;

#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    const int FEATURE_LEAF = 7;
    const int AVX2_BIT = 5;
    const int AVX512F_BIT = 16;

    if (nIds >= FEATURE_LEAF) {
        __cpuidex(cpuInfo, FEATURE_LEAF, 0);
        has_avx2 = (cpuInfo[1] & (1 << AVX2_BIT)) != 0;        // EBX bit 5
        has_avx512f = (cpuInfo[1] & (1 << AVX512F_BIT)) != 0;  // EBX bit 16
    }
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    const int FEATURE_LEAF = 7;
    const int AVX2_BIT = 5;
    const int AVX512F_BIT = 16;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        if (eax >= FEATURE_LEAF) {
            __get_cpuid_count(FEATURE_LEAF, 0, &eax, &ebx, &ecx, &edx);
            has_avx2 = (ebx & (1 << AVX2_BIT)) != 0;
            has_avx512f = (ebx & (1 << AVX512F_BIT)) != 0;
        }
    }
#endif

    internal_cpu_info.has_avx2 = has_avx2;
    internal_cpu_info.has_avx512f = has_avx512f;
    cpu_info_initialized = true;

    return internal_cpu_info;
}

// --- Main Dispatcher ---

void matmul(const Tensor &mat_a, const Tensor &mat_b, Tensor &mat_c)
{
    if (mat_a.ndim() != 2 || mat_b.ndim() != 2 || mat_c.ndim() != 2) {
        throw std::invalid_argument("matmul supports only 2D tensors");
    }

    size_t rows_m = mat_a.shape()[0];
    size_t inner_k = mat_a.shape()[1];
    if (mat_b.shape()[0] != inner_k) {
        throw std::invalid_argument("matmul dimension mismatch: A.cols != B.rows");
    }
    size_t cols_n = mat_b.shape()[1];

    if (mat_c.shape()[0] != rows_m || mat_c.shape()[1] != cols_n) {
        throw std::invalid_argument("matmul output tensor C must have shape [M, N]");
    }

    size_t lda = mat_a.strides()[0];
    size_t ldb = mat_b.strides()[0];
    size_t ldc = mat_c.strides()[0];

    auto info = get_cpu_info();

    // GPU Dispatch
#ifdef SERIUS_USE_CUDA
    size_t flops = 2 * rows_m * cols_n * inner_k;
    hal::IBackend *backend = hal::get_backend(hal::DeviceType::CUDA);
    // Use GPU for relatively large matrices or non-float32
    if (backend && backend->is_available() && (flops > 1000000 || mat_a.dtype() != hal::DType::Float32)) {
        mat_a.syncToDevice();
        mat_b.syncToDevice();
        mat_c.syncToDevice();
        matmul_cuda(mat_a.data_gpu(), mat_b.data_gpu(), mat_c.data_gpu(), rows_m, cols_n, inner_k, lda, ldb, ldc,
                    mat_a.dtype());

        mat_c.mark_dirty(Tensor::DataLocation::GPU);
        mat_c.syncToHost();
        return;
    }
#endif

    if (mat_a.dtype() != hal::DType::Float32) {
        throw std::runtime_error("BLAS::matmul CPU path only supports Float32");
    }

    const float *a_ptr = mat_a.data();
    const float *b_ptr = mat_b.data();
    float *c_ptr = mat_c.data();

    if (info.has_avx512f) {
        matmul_avx512(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
    } else if (info.has_avx2) {
        matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
    } else {
        matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
    }
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
    size_t cols_n = 1;  // For matvec, N is always 1

    if (vec_y.shape()[0] != rows_m) {
        throw std::invalid_argument("matvec output tensor y must have shape [M]");
    }

    size_t lda = mat_a.strides()[0];
    size_t ldx = vec_x.strides()[0];  // Stride for x (vector)
    size_t ldy = vec_y.strides()[0];  // Stride for y (vector)

#ifdef SERIUS_USE_CUDA
    hal::IBackend *gpu_backend = hal::get_gpu_backend();
    if (gpu_backend && gpu_backend->is_available() &&
        (rows_m * inner_k > 1000000 || mat_a.dtype() != hal::DType::Float32)) {
        mat_a.syncToDevice();
        vec_x.syncToDevice();
        vec_y.syncToDevice();

        matmul_cuda(mat_a.data_gpu(), vec_x.data_gpu(), vec_y.data_gpu(), rows_m, 1, inner_k, lda, 1, 1, mat_a.dtype());

        vec_y.mark_dirty(Tensor::DataLocation::GPU);
        vec_y.syncToHost();
        return;
    }
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

void batched_matmul(const Tensor &mat_a, const Tensor &mat_b, Tensor &mat_c)
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
    if (backend && backend->is_available() && (total_flops > 1000000 || mat_a.dtype() != hal::DType::Float32)) {
        mat_a.syncToDevice();
        mat_b.syncToDevice();
        mat_c.syncToDevice();

        size_t stride_a = mat_a.strides()[0];
        size_t stride_b = (mat_b.ndim() == 3) ? mat_b.strides()[0] : 0;  // 0 stride if broadcasting 2D B
        size_t stride_c = mat_c.strides()[0];

        batched_matmul_cuda(mat_a.data_gpu(), mat_b.data_gpu(), mat_c.data_gpu(), batch_size, rows_m, cols_n, inner_k,
                            lda, ldb, ldc, stride_a, stride_b, stride_c, mat_a.dtype());

        mat_c.mark_dirty(Tensor::DataLocation::GPU);
        mat_c.syncToHost();
        return;
    }
#endif

    if (mat_a.dtype() != hal::DType::Float32) {
        throw std::runtime_error("BLAS::batched_matmul CPU path only supports Float32");
    }

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const float *a_ptr = mat_a.data() + (batch_idx * mat_a.strides()[0]);  // Use A.strides()[0] for batch stride
        const float *b_ptr = (mat_b.ndim() == 3) ? (mat_b.data() + (batch_idx * mat_b.strides()[0]))
                                                 : mat_b.data();         // Use B.strides()[0] for batch stride
        float *c_ptr = mat_c.data() + (batch_idx * mat_c.strides()[0]);  // Use C.strides()[0] for batch stride

        if (info.has_avx512f) {
            matmul_avx512(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else if (info.has_avx2) {
            matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else {
            matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        }
    }
}

// --- Convenience Wrappers ---

Tensor matmul(const Tensor &mat_a, const Tensor &mat_b)
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
    matmul(mat_a, mat_b, mat_c);
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

Tensor batched_matmul(const Tensor &mat_a, const Tensor &mat_b)
{
    if (mat_a.ndim() != 3) {
        throw std::invalid_argument("batched_matmul wrapper supports only 3D A tensors");
    }
    if (mat_b.ndim() != 2 && mat_b.ndim() != 3) {
        throw std::invalid_argument("batched_matmul wrapper supports only 2D/3D B tensors");
    }

    size_t batch_size = mat_a.shape()[0];
    size_t rows_m = mat_a.shape()[1];
    size_t cols_n = (mat_b.ndim() == 3) ? mat_b.shape()[2] : mat_b.shape()[1];

    Tensor mat_c({batch_size, rows_m, cols_n}, mat_a.dtype());
    batched_matmul(mat_a, mat_b, mat_c);
    return mat_c;
}

}  // namespace blas
}  // namespace serius
