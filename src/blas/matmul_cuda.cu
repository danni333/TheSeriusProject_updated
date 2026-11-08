#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "matmul_cuda.h"

#define CHECK_CUDA_ERROR(call)                                                                       \
    do {                                                                                             \
        cudaError_t err = call;                                                                      \
        if (err != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" \
                      << cudaGetErrorString(err) << "\"" << std::endl;                               \
        }                                                                                            \
    } while (0)

namespace serius {
namespace core {
namespace blas {

namespace {
cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t get_cublas_handle()
{
    if (g_cublas_handle == nullptr) {
        if (cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
            g_cublas_handle = nullptr;
        } else {
            // Give cuBLAS a dedicated workspace up front: without it every
            // GemmEx/GemmStridedBatchedEx call may hit cudaMalloc internally
            // (ms-scale, synchronizing) — visible as multi-ms GPU idle gaps
            // between consecutive GEMM kernels during prefill.
            static void *g_cublas_ws = nullptr;
            if (g_cublas_ws == nullptr &&
                cudaMalloc(&g_cublas_ws, 64ull * 1024 * 1024) == cudaSuccess) {
                cublasSetWorkspace(g_cublas_handle, g_cublas_ws, 64ull * 1024 * 1024);
            }
        }
    }
    return g_cublas_handle;
}

cudaDataType cublas_dtype(hal::DType dtype)
{
    switch (dtype) {
        case hal::DType::Float32: return CUDA_R_32F;
        case hal::DType::Float16: return CUDA_R_16F;
        case hal::DType::BFloat16: return CUDA_R_16BF;
        default: return CUDA_R_32F;
    }
}

// --- cuBLASLt path with cached descriptors/heuristics ---
// cublasGemmEx/GemmStridedBatchedEx re-run kernel-selection heuristics on every
// call (~0.3 ms of host time each). A transformer prefill issues ~250 of them
// (~80 ms of pure host overhead). cuBLASLt lets us cache the operation
// descriptor, matrix layouts and the chosen algorithm per problem shape, so
// repeat calls cost only the launch itself.
struct LtEntry {
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aLay = nullptr, bLay = nullptr, cLay = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    void *ws = nullptr;
    size_t ws_size = 0;
    // full problem description for cache validation
    size_t m = 0, n = 0, k = 0, batch = 0;
    size_t lda = 0, ldb = 0, ldc = 0, sa = 0, sb = 0, sc = 0;
    cudaDataType ty = CUDA_R_32F;
};

struct LtCache {
    cublasLtHandle_t h = nullptr;
    std::vector<LtEntry> entries;
    std::mutex mu;
    void *ws = nullptr;
    size_t ws_size = 32ull * 1024 * 1024;
};

LtCache &lt_cache()
{
    static LtCache c;
    return c;
}

uint64_t lt_key(size_t m, size_t n, size_t k, size_t batch, size_t lda, size_t ldb, size_t ldc,
                size_t sa, size_t sb, size_t sc, cudaDataType ty)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (uint64_t v : {m, n, k, batch, lda, ldb, ldc, sa, sb, sc, (uint64_t)ty}) mix(v);
    return h;
}

void lt_entry_destroy(LtEntry &e)
{
    if (e.opDesc) cublasLtMatmulDescDestroy(e.opDesc);
    if (e.aLay) cublasLtMatrixLayoutDestroy(e.aLay);
    if (e.bLay) cublasLtMatrixLayoutDestroy(e.bLay);
    if (e.cLay) cublasLtMatrixLayoutDestroy(e.cLay);
    if (e.ws) cudaFree(e.ws);
    e = LtEntry{};
}

// Column-major GEMM: C(m x n) = op_N(A)(m x k) @ op_N(B)(k x n), optional
// strided batch. Mirrors the operand order used by the cublasGemmEx call
// sites (row-major C = A_rm @ B_rm is expressed as the transposed problem).
bool lt_gemm(const void *A, size_t lda, size_t sa, const void *B, size_t ldb, size_t sb,
             void *C, size_t ldc, size_t sc, size_t m, size_t n, size_t k, size_t batch,
             cudaDataType ty, cudaStream_t stream)
{
    LtCache &cache = lt_cache();
    if (cache.h == nullptr) {
        if (cublasLtCreate(&cache.h) != CUBLAS_STATUS_SUCCESS) return false;
        if (cudaMalloc(&cache.ws, cache.ws_size) != cudaSuccess) {
            cache.ws_size = 0;
        }
    }
    const uint64_t key = lt_key(m, n, k, batch, lda, ldb, ldc, sa, sb, sc, ty);
    std::lock_guard<std::mutex> lock(cache.mu);

    LtEntry *e = nullptr;
    for (auto &en : cache.entries) {
        if (en.m == m && en.n == n && en.k == k && en.batch == batch && en.lda == lda &&
            en.ldb == ldb && en.ldc == ldc && en.sa == sa && en.sb == sb && en.sc == sc &&
            en.ty == ty) {
            e = &en;
            break;
        }
    }
    if (e == nullptr) {
        if (cache.entries.size() >= 64) return false;  // give up rather than grow unbounded
        cache.entries.emplace_back();
        e = &cache.entries.back();
        e->m = m; e->n = n; e->k = k; e->batch = batch;
        e->lda = lda; e->ldb = ldb; e->ldc = ldc;
        e->sa = sa; e->sb = sb; e->sc = sc; e->ty = ty;

        if (cublasLtMatmulDescCreate(&e->opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
            lt_entry_destroy(*e); cache.entries.pop_back(); return false;
        }
        cublasOperation_t nop = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(e->opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &nop, sizeof(nop));
        cublasLtMatmulDescSetAttribute(e->opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &nop, sizeof(nop));

        if (cublasLtMatrixLayoutCreate(&e->aLay, ty, (uint64_t)m, (uint64_t)k, (uint64_t)lda) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&e->bLay, ty, (uint64_t)k, (uint64_t)n, (uint64_t)ldb) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&e->cLay, ty, (uint64_t)m, (uint64_t)n, (uint64_t)ldc) != CUBLAS_STATUS_SUCCESS) {
            lt_entry_destroy(*e); cache.entries.pop_back(); return false;
        }
        if (batch > 1) {
            int b = (int)batch;
            cublasLtMatrixLayoutSetAttribute(e->aLay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b));
            cublasLtMatrixLayoutSetAttribute(e->bLay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b));
            cublasLtMatrixLayoutSetAttribute(e->cLay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b));
            uint64_t oa = sa, ob = sb, oc = sc;
            cublasLtMatrixLayoutSetAttribute(e->aLay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &oa, sizeof(oa));
            cublasLtMatrixLayoutSetAttribute(e->bLay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ob, sizeof(ob));
            cublasLtMatrixLayoutSetAttribute(e->cLay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &oc, sizeof(oc));
        }

        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t maxWs = cache.ws_size;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));
        cublasLtMatmulHeuristicResult_t res;
        int nres = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(cache.h, e->opDesc, e->aLay, e->bLay, e->cLay,
                                                        e->cLay, pref, 1, &res, &nres);
        cublasLtMatmulPreferenceDestroy(pref);
        if (hs != CUBLAS_STATUS_SUCCESS || nres < 1) {
            static const bool lt_dbg2 = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_LT_DBG") == 0 && n > 0; }();
            if (lt_dbg2) fprintf(stderr, "[lt] heuristic failed hs=%d nres=%d m=%zu n=%zu k=%zu b=%zu\n",
                                 (int)hs, nres, m, n, k, batch);
            lt_entry_destroy(*e); cache.entries.pop_back(); return false;
        }
        e->heur = res;
        e->ws_size = res.workspaceSize;
        if (e->ws_size > 0) {
            if (cudaMalloc(&e->ws, e->ws_size) != cudaSuccess) {
                lt_entry_destroy(*e); cache.entries.pop_back(); return false;
            }
        }
    }
    (void)key;

    const float alpha = 1.0f, beta = 0.0f;
    static const bool lt_dbg = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_LT_DBG") == 0 && n > 0; }();
    cublasStatus_t st = cublasLtMatmul(cache.h, e->opDesc, &alpha, A, e->aLay, B, e->bLay, &beta,
                                     C, e->cLay, C, e->cLay, &e->heur.algo,
                                     e->ws, e->ws_size, stream);
    if (lt_dbg && st != CUBLAS_STATUS_SUCCESS) {
        static int shown = 0;
        if (shown++ < 5) fprintf(stderr, "[lt] matmul failed st=%d m=%zu n=%zu k=%zu b=%zu\n",
                                 (int)st, m, n, k, batch);
    }
    return st == CUBLAS_STATUS_SUCCESS;
}
}  // namespace

// Helper to convert to float for calculation
template <typename T>
__device__ __forceinline__ float to_float(T val)
{
    return (float)val;
}
template <>
__device__ __forceinline__ float to_float(__half val)
{
    return __half2float(val);
}
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val)
{
    return (float)val;  // Built-in conversion works in modern CUDA
}

template <typename T>
__global__ void matmul_kernel_simple(const T *A, const T *B, T *C, int M, int N, int K, int lda, int ldb, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += to_float(A[row * lda + k]) * to_float(B[k * ldb + col]);
        }
        C[row * ldc + col] = (T)sum;
    }
}

template <typename T>
__global__ void batched_matmul_kernel(const T *A, const T *B, T *C, int batch_size, int M, int N, int K, int lda,
                                      int ldb, int ldc, long long stride_a, long long stride_b, long long stride_c)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < N) {
        const T *a_batch = A + batch * stride_a;
        const T *b_batch = B + batch * stride_b;
        T *c_batch = C + batch * stride_c;

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += to_float(a_batch[row * lda + k]) * to_float(b_batch[k * ldb + col]);
        }
        c_batch[row * ldc + col] = (T)sum;
    }
}

void matmul_cuda(const void *A, const void *B, void *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                 size_t ldc, hal::DType dtype, void *stream)
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    // cuBLAS fast path for contiguous row-major C = A @ B.
    // Row-major (M,N) = (M,K)@(K,N) maps to col-major (N,M) = (N,K)@(K,M),
    // i.e. operands are swapped with OP_N on both sides.
    const bool contiguous = (lda == K) && (ldb == N) && (ldc == N);
    static const bool use_cublas = [] {
        char buf[2];
        size_t n = 0;
        return !(getenv_s(&n, buf, sizeof(buf), "SERIUS_NO_CUBLAS") == 0 && n > 0);
    }();
    static const bool dbg = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_CUBLAS_DBG") == 0 && n > 0 ? 1 : 0; }();
    if (dbg && !(contiguous && use_cublas)) {
        fprintf(stderr, "[cublas] skip M=%zu N=%zu K=%zu lda=%zu ldb=%zu ldc=%zu contiguous=%d use=%d\n",
                M, N, K, lda, ldb, ldc, (int)contiguous, (int)use_cublas);
    }
    if (contiguous && use_cublas) {
        const cudaDataType ty = cublas_dtype(dtype);
        // cuBLASLt with cached heuristic: ~10us dispatch vs ~300us for GemmEx.
        // (FP32 stays on GemmEx: the Lt path is only exercised by the FP16/BF16
        // prefill and showed wrong results for CUDA_R_32F.)
        static const bool no_lt = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_NO_CUBLASLT") == 0 && n > 0; }();
        if (!no_lt && ty != CUDA_R_32F &&
            lt_gemm(B, (size_t)ldb, 0, A, (size_t)lda, 0, C, (size_t)ldc, 0,
                    (size_t)N, (size_t)M, (size_t)K, 1, ty, cu_stream)) {
            return;
        }
        cublasHandle_t handle = get_cublas_handle();
        if (handle == nullptr && dbg) fprintf(stderr, "[cublas] handle null\n");
        if (handle != nullptr) {
            cublasSetStream(handle, cu_stream);
            const float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                                             B, ty, (int)ldb, A, ty, (int)lda, &beta, C, ty, (int)ldc,
                                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) {
                return;
            }
            static int dbg = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_CUBLAS_DBG") == 0 && n > 0 ? 1 : 0; }();
            if (dbg) {
                fprintf(stderr, "[cublas] GemmEx failed st=%d M=%zu N=%zu K=%zu lda=%zu ldb=%zu ldc=%zu\n",
                        (int)st, M, N, K, lda, ldb, ldc);
            }
        }
    }

    dim3 block(16, 16);
    dim3 grid((int)((N + 15) / 16), (int)((M + 15) / 16));

    if (dtype == hal::DType::Float32) {
        matmul_kernel_simple<float><<<grid, block, 0, cu_stream>>>((const float *)A, (const float *)B, (float *)C, (int)M,
                                                                   (int)N, (int)K, (int)lda, (int)ldb, (int)ldc);
    } else if (dtype == hal::DType::Float16) {
        matmul_kernel_simple<__half><<<grid, block, 0, cu_stream>>>((const __half *)A, (const __half *)B, (__half *)C,
                                                                    (int)M, (int)N, (int)K, (int)lda, (int)ldb, (int)ldc);
    } else if (dtype == hal::DType::BFloat16) {
        matmul_kernel_simple<__nv_bfloat16><<<grid, block, 0, cu_stream>>>(
            (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (__nv_bfloat16 *)C, (int)M, (int)N, (int)K, (int)lda,
            (int)ldb, (int)ldc);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}

void batched_matmul_cuda(const void *A, const void *B, void *C, size_t batch_size, size_t M, size_t N, size_t K,
                         size_t lda, size_t ldb, size_t ldc, size_t stride_a, size_t stride_b, size_t stride_c,
                         hal::DType dtype, void *stream)
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    // cuBLAS strided-batched fast path (same row-major swap as matmul).
    const bool contiguous = (lda == K) && (ldb == N) && (ldc == N) &&
                            (stride_a == M * K) && (stride_c == M * N) &&
                            (stride_b == 0 || stride_b == K * N);
    if (contiguous && batch_size > 1) {
        const cudaDataType ty = cublas_dtype(dtype);
        // cuBLASLt strided-batch with cached heuristic: one dispatch per call
        // instead of the ~ms-scale heuristic inside GemmStridedBatchedEx.
        static const bool no_lt_b = [] { size_t n = 0; return getenv_s(&n, nullptr, 0, "SERIUS_NO_CUBLASLT") == 0 && n > 0; }();
        if (!no_lt_b && ty != CUDA_R_32F &&
            lt_gemm(B, ldb, stride_b, A, lda, stride_a, C, ldc, stride_c,
                    N, M, K, batch_size, ty, cu_stream)) {
            return;
        }
        cublasHandle_t handle = get_cublas_handle();
        if (handle != nullptr) {
            cublasSetStream(handle, cu_stream);
            const float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t st = cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                B, ty, (int)ldb, (long long)stride_b, A, ty, (int)lda, (long long)stride_a, &beta,
                C, ty, (int)ldc, (long long)stride_c, (int)batch_size, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) {
                return;
            }
        }
    }

    dim3 block(16, 16, 1);
    dim3 grid((int)((N + 15) / 16), (int)((M + 15) / 16), (int)batch_size);

    if (dtype == hal::DType::Float32) {
        batched_matmul_kernel<float><<<grid, block, 0, cu_stream>>>((const float *)A, (const float *)B, (float *)C,
                                                                    (int)batch_size, (int)M, (int)N, (int)K, (int)lda,
                                                                    (int)ldb, (int)ldc, (long long)stride_a,
                                                                    (long long)stride_b, (long long)stride_c);
    } else if (dtype == hal::DType::Float16) {
        batched_matmul_kernel<__half><<<grid, block, 0, cu_stream>>>(
            (const __half *)A, (const __half *)B, (__half *)C, (int)batch_size, (int)M, (int)N, (int)K, (int)lda,
            (int)ldb, (int)ldc, (long long)stride_a, (long long)stride_b, (long long)stride_c);
    } else if (dtype == hal::DType::BFloat16) {
        batched_matmul_kernel<__nv_bfloat16><<<grid, block, 0, cu_stream>>>(
            (const __nv_bfloat16 *)A, (const __nv_bfloat16 *)B, (__nv_bfloat16 *)C, (int)batch_size, (int)M, (int)N,
            (int)K, (int)lda, (int)ldb, (int)ldc, (long long)stride_a, (long long)stride_b, (long long)stride_c);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
}

void logits_gemv_mixed_cublas(const void *x, const void *w, void *out, size_t seq, size_t vocab, size_t hidden,
                              void *stream)
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle();
    if (handle == nullptr) return;
    cublasSetStream(handle, cu_stream);

    // Row-major out[seq, vocab] = x[seq, hidden] @ w[vocab, hidden]^T.
    // In col-major terms this is C'(vocab x seq) = A'(hidden x vocab)^T * B'(hidden x seq)
    // where A' reuses the w memory with lda=hidden, B' reuses x with ldb=hidden,
    // and C' with ldc=vocab is exactly the row-major out layout.
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)vocab, (int)seq, (int)hidden, &alpha,
                                     w, CUDA_R_16F, (int)hidden, x, CUDA_R_16F, (int)hidden, &beta,
                                     out, CUDA_R_32F, (int)vocab, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[cublas] logits_gemv_mixed failed st=" << (int)st << std::endl;
    }
}

}  // namespace blas
}  // namespace core
}  // namespace serius
