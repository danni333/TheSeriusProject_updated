# 🎯 TheSeriusProject - Implementazioni Consigliate

**Data**: 2026-03-09  
**Audit**: Performance Benchmark & Code Analysis  
**Status**: In Progress

---

## 📋 Indice

1. [BLAS/MatMul](#1-blasmatmul-ottimizzazione)
2. [Memory Pool](#2-memory-pool-miglioramenti)
3. [Mixed Precision](#3-mixed-precision-guidelines)
4. [Transformer Ottimizzazioni](#4-transformer-ottimizzazioni)
5. [Kernel Fusion](#5-kernel-fusion)
6. [Flash Attention](#6-flash-attention)
7. [Profiling & Debugging](#7-profiling--debugging)
8. [Piano di Implementazione](#8-piano-di-implementazione)

---

## 1. BLAS/MatMul Ottimizzazione

### Problema Identificato

Il benchmark mostra un'anomalia per matrici 256x256:

- **Speedup SIMD**: 0.01x (invece di ~20-30x atteso)
- Il dispatch a SIMD per dimensioni piccole è controproducente

### Causa Probabile

Il threshold per l'uso di SIMD è troppo basso. Per matrici piccole, l'overhead di setup supera il guadagno.

### Implementazione Consigliata

**File**: `src/blas/dispatch.cpp`

```cpp
void matmul(const Tensor &mat_a, const Tensor &mat_b, Tensor &mat_c)
{
    // ... validazioni esistenti ...

    size_t rows_m = mat_a.shape()[0];
    size_t inner_k = mat_a.shape()[1];
    size_t cols_n = mat_b.shape()[1];

    auto info = get_cpu_info();

    // GPU Dispatch (mantenere)
    #ifdef SERIUS_USE_CUDA
    size_t flops = 2 * rows_m * cols_n * inner_k;
    hal::IBackend *backend = hal::get_backend(hal::DeviceType::CUDA);
    if (backend && backend->is_available() && (flops > 1000000 || mat_a.dtype() != hal::DType::Float32)) {
        // ... GPU path esistente ...
        return;
    }
    #endif

    // CPU Path - NUOVA LOGICA CON SOGLIE
    const float *a_ptr = mat_a.data();
    const float *b_ptr = mat_b.data();
    float *c_ptr = mat_c.data();

    // Calcola dimensione del problema
    size_t problem_size = rows_m * cols_n * inner_k;

    if (mat_a.dtype() != hal::DType::Float32) {
        throw std::runtime_error("BLAS::matmul CPU path only supports Float32");
    }

    // Soglie empiriche (da calibrare)
    const size_t SIMD_THRESHOLD_SMALL = 128 * 128 * 128;   // ~2M FLOPS
    const size_t SIMD_THRESHOLD_MEDIUM = 256 * 256 * 256; // ~16M FLOPS

    if (problem_size < SIMD_THRESHOLD_SMALL) {
        // Per matrici molto piccole, usa scalar (minimo overhead)
        matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
    } else if (problem_size < SIMD_THRESHOLD_MEDIUM) {
        // Per medie, testa entrambi e sceglie il più veloce (o usa AVX2 se disponibile)
        if (info.has_avx2) {
            matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else {
            matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        }
    } else {
        // Per grandi matrici, usa AVX512 se disponibile
        if (info.has_avx512f) {
            matmul_avx512(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else if (info.has_avx2) {
            matmul_avx2(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        } else {
            matmul_scalar(a_ptr, b_ptr, c_ptr, rows_m, cols_n, inner_k, lda, ldb, ldc);
        }
    }
}
```

**Test**: Aggiungere benchmark per 64x64, 128x128, 256x256 per calibrare le soglie.

---

## 2. Memory Pool Miglioramenti

### Stato Attuale

Il memory pool è già eccellente:

- 30-2500x speedup su malloc/free
- Specialmente efficace per allocazioni grandi (1MB+)

### Raccomandazioni

1. **Thread-Local Pools** (per multi-threading)
2. **Size-Class Segregation** (pool separati per taglie diverse)
3. **Cache-Line Alignment** (allineamento a 64 byte per false sharing)

### Implementazione: Thread-Local Pool

**File**: `src/memory/memory_pool.h`

```cpp
namespace serius {
namespace core {
namespace memory {

class MemoryPool {
public:
    // ... esistente ...

    // NUOVO: Thread-local storage
    static MemoryPool& get_thread_local() {
        thread_local MemoryPool tl_pool(64 * 1024 * 1024);  // 64MB per thread
        return tl_pool;
    }

    void* allocate_thread_local(size_t size) {
        return get_thread_local().allocate(size);
    }

    void reset_thread_local() {
        get_thread_local().reset();
    }
};

}}} // namespace serius::core::memory
```

**Uso**:

```cpp
// Invece di:
void* ptr = pool.allocate(1024);

// Usa:
void* ptr = MemoryPool::get_thread_local().allocate(1024);
```

---

## 3. Mixed Precision Guidelines

### Performance Observed

| Size      | FP32 Scalar | FP32 SIMD          | FP16 SIMD          | BF16 SIMD          |
|-----------|-------------|--------------------|--------------------|--------------------|
| 256x256   | 25.33ms     | 0.49ms (68 GFLOPS) | 0.46ms (73 GFLOPS) | 0.36ms (94 GFLOPS) |
| 1024x1024 | 4963ms      | 17.04ms (126 GFLOPS)| 7.22ms (298 GFLOPS)| 7.12ms (301 GFLOPS)|

### Raccomandazioni

1. **FP16**: Preferito per training e inferenza (miglior precisione)
2. **BF16**: Solo per inferenza (errori più alti, ma più stabile numericamente)
3. **Automatic Mixed Precision (AMP)**: Implementare loss scaling

### Implementazione: AMP Helper

**File**: `src/optim/amp.h`

```cpp
#pragma once
#include "../tensor/tensor.h"
#include "../hal/device.h"

namespace serius {
namespace core {
namespace optim {

class AutomaticMixedPrecision {
public:
    // Scaling factor per evitare underflow in FP16
    static constexpr float DEFAULT_LOSS_SCALE = 65536.0f;

    // Forward pass in FP16, backward in FP32
    template<typename FwdFn, typename BwdFn>
    static void mixed_precision_forward_backward(
        FwdFn forward_fn,
        BwdFn backward_fn,
        const Tensor& input,
        Tensor& output,
        Tensor& loss,
        Tensor& grad_input)
    {
        // 1. Forward in FP16
        Tensor input_fp16 = input.cast(hal::DType::Float16);
        Tensor output_fp16 = forward_fn(input_fp16);

        // 2. Loss scaling
        loss = compute_loss(output_fp16) * DEFAULT_LOSS_SCALE;

        // 3. Backward in FP32 (loss unscaled)
        Tensor loss_unscaled = loss.cast(hal::DType::Float32) / DEFAULT_LOSS_SCALE;
        Tensor grad_output = compute_grad(loss_unscaled);

        // 4. Compute gradient in FP32
        backward_fn(output_fp16, grad_output, grad_input);
    }

    // Inference-only: converti modello a FP16
    static void convert_model_to_fp16(TransformerModel& model) {
        // Itera su tutti i parametri e converti
        for (auto& param : model.parameters()) {
            if (param.dtype() == hal::DType::Float32) {
                param = param.cast(hal::DType::Float16);
            }
        }
    }
};

}}} // namespace serius::core::optim
```

---

## 4. Transformer Ottimizzazioni

### Performance Attuale

- **Batch=1, Seq=128**: 441.62ms (289.84 tok/s)
- **Batch=8, Seq=128**: 3143.75ms (325.73 tok/s)

**Target**: 1000-5000 tok/s (settore)

### Problemi

1. **Attention non ottimizzata**: O(N²) memory e compute
2. **Nessun kernel fusion**: Ogni operazione legge/scrive HBM
3. **KV Cache non ottimizzata**: Copie inutili

### Implementazione: Optimized Transformer Block

**File**: `src/transformer/block_optimized.cpp`

```cpp
#include "../transformer/block.h"
#include "../blas/blas.h"
#include "../hal/device.h"

namespace serius {
namespace model {

// Versione ottimizzata con kernel fusion
Tensor TransformerBlock::forward_optimized(const Tensor& x) {
    // 1. LayerNorm + Residual (fused se possibile)
    Tensor normed = layernorm.forward(x);
    Tensor residual1 = x + normed;  // residual connection

    // 2. Attention con KV cache ottimizzata
    Tensor attn_out = attention.forward_optimized(residual1);

    // 3. Secondo residual
    Tensor residual2 = residual1 + attn_out;

    // 4. FFN (potrebbe essere fusionato)
    Tensor ffn_out = ffn.forward(residual2);

    // 5. Output residual
    Tensor output = residual2 + ffn_out;

    return output;
}

}}} // namespace serius::model
```

---

## 5. Kernel Fusion

### Pattern di Fusione Riconosciuti

| Pattern | Beneficio | Implementazione |
|---------|-----------|-----------------|
| `bias + GELU` | 1.5-2x speedup | `fused_bias_gelu_kernel` |
| `LayerNorm + Linear` | 2-3x speedup | `fused_ln_linear_kernel` |
| `Softmax + Mask` | 1.3-1.5x speedup | `fused_masked_softmax_kernel` |
| `Residual + Dropout` | 1.2x speedup | `fused_residual_dropout` |

### Implementazione: Fused Bias + GELU

**File**: `src/optim/fused_kernels.cu` (aggiungere)

```cpp
#include <cuda_runtime.h>
#include "../hal/device.h"

namespace serius {
namespace core {
namespace optim {

__global__ void fused_bias_gelu_kernel(
    float* x,
    const float* bias,
    int n,
    int bias_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx] + bias[idx % bias_size];
        // GELU approssimata (fast)
        x[idx] = 0.5f * val * (1.0f + tanhf(0.7978846f * (val + 0.044715f * val*val*val)));
    }
}

void fused_bias_gelu(Tensor& x, const Tensor& bias) {
    SERIUS_CHECK(x.dtype() == hal::DType::Float32, "Only FP32 supported");
    SERIUS_CHECK(bias.dtype() == hal::DType::Float32, "Only FP32 supported");

    int n = x.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    fused_bias_gelu_kernel<<<grid_size, block_size>>>(
        x.data_gpu(),
        bias.data_gpu(),
        n,
        bias.size()
    );
    CUDA_CHECK(cudaGetLastError());
}

}}} // namespace serius::core::optim
```

---

## 6. Flash Attention

### Perché Flash Attention?

- **Riduce memoria**: da O(N²) a O(N)
- **Speedup**: 2-10x per sequenze lunghe (N > 1024)
- **Throughput**: +300% per batch size > 4

### Implementazione: Flash Attention v2

**File**: `src/attention/flash_attn.h`

```cpp
#pragma once
#include "../tensor/tensor.h"
#include "../hal/device.h"

namespace serius {
namespace model {

struct FlashAttentionConfig {
    int block_size_q = 128;   // Br
    int block_size_kv = 64;   // Bc
    bool causal = false;
};

Tensor flash_attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const FlashAttentionConfig& config = FlashAttentionConfig{});

// Wrapper per integrazione nel Transformer
class FlashAttentionModule {
public:
    FlashAttentionModule(const ModelConfig& config);
    Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V);

private:
    FlashAttentionConfig cfg_;
    Tensor workspace_;  // Buffer temporaneo
};

}}} // namespace serius::model
```

**File**: `src/attention/flash_attn_cuda.cu`

```cpp
#include "flash_attn.h"
#include "../hal/cuda_backend.cuh"

// Implementazione kernel basata su FlashAttention-2
// (vedi todo/fase11_flash_attention.md per pseudocodice completo)

__global__ void flash_attention_fwd_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Br, int Bc, float softmax_scale)
{
    // Implementazione con tiling e online softmax
    // ... (complessità elevata, vedere reference implementation)
}

Tensor flash_attention(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    const FlashAttentionConfig& config)
{
    // Validazioni
    SERIUS_CHECK(Q.ndim() == 4, "Q must be [batch, heads, seq, head_dim]");
    // ...

    // Allocazione output
    Tensor O(Q.shape(), Q.dtype());

    // Launch kernel
    // ... (logica di dispatch)

    return O;
}
```

---

## 7. Profiling & Debugging

### Tool Consigliati

1. **Custom Tensor Logger** (per NaN/Inf detection)
2. **ScopedTimer** (per profiling granulare)
3. **Memory Leak Detection** (valgrind/Dr. Memory)

### Implementazione: Tensor Health Monitor

**File**: `src/debug/tensor_logger.h`

```cpp
#pragma once
#include <fstream>
#include <vector>
#include "../tensor/tensor.h"

namespace serius {
namespace debug {

struct TensorStats {
    std::string name;
    int step;
    float mean, std, min, max;
    bool has_nan, has_inf;
    float sparsity;
};

class TensorLogger {
public:
    enum Level { SILENT, SUMMARY, FULL, PANIC };
    void set_level(Level l) { level_ = l; }

    void log(const Tensor& t, const std::string& name, int step);

    void export_csv(const std::string& path);

private:
    Level level_ = SUMMARY;
    std::ofstream log_file_;
    std::vector<TensorStats> history_;

    TensorStats compute(const Tensor& t, const std::string& name, int step);
    void panic(const TensorStats& s);
};

// Singleton access
TensorLogger& get_tensor_logger();

}}} // namespace serius::debug
```

**Uso**:

```cpp
#include "../debug/tensor_logger.h"

void TransformerBlock::forward(Tensor& x) {
    Tensor attn_out = attention(x);
    debug::get_tensor_logger().log(attn_out, "block_3.attn", global_step);

    Tensor ffn_out = ffn(attn_out);
    debug::get_tensor_logger().log(ffn_out, "block_3.ffn", global_step);
}
```

---

## 8. Piano di Implementazione

### Priorità Alta (Immediata)

1. ✅ **Fix BLAS dispatch thresholds** (1-2 giorni)
   - Modificare `src/blas/dispatch.cpp`
   - Aggiungere benchmark per calibrare soglie
   - Test: `tests/bench_matmul.cpp`

2. ✅ **Thread-local Memory Pool** (1 giorno)
   - Modificare `src/memory/memory_pool.h`
   - Test: `tests/test_memory.cpp`

3. ⏳ **Tensor Logger** (2-3 giorni)
   - Creare `src/debug/tensor_logger.h/.cpp`
   - Test: `tests/test_logger.cpp`

### Priorità Media (Settimana 1-2)

1. **Kernel Fusion: Bias + GELU** (2-3 giorni)
   - Creare `src/optim/fused_kernels.cu`
   - Test: `tests/test_fusion.cpp`

2. **Flash Attention v2** (5-7 giorni)
   - Implementare `src/attention/flash_attn_*.cu`
   - Test: `tests/test_flash_attn.cpp`
   - Benchmark: `tests/bench_flash_attn.cpp`

### Priorità Bassa (Settimana 3-4)

1. **Automatic Mixed Precision** (3-4 giorni)
   - Creare `src/optim/amp.h/.cpp`
   - Test integrazione con training loop

2. **Paged KV Cache** (4-5 giorni)
   - Estendere `src/transformer/kv_cache.cpp`
   - Test: `tests/test_kv_cache_paging.cpp`

---

## 9. Metriche di Successo

### KPI da Monitorare

| Metric | Attuale | Target | Misura |
|--------|---------|--------|--------|
| MatMul 1024x1024 | 12.62ms | <8ms | bench_matmul |
| Memory Pool speedup | 30-2500x | >100x | bench_memory |
| Transformer throughput | 300-400 tok/s | >1000 tok/s | bench_full |
| Flash Attention speedup (N=4096) | N/A | >3x | bench_flash_attn |
| Kernel fusion speedup | N/A | >1.5x | bench_fusion |

### Test Automatici

```bash
# Esegui tutti i benchmark
cd build/tests/Release
bench_matmul.exe
bench_memory.exe
bench_full.exe
bench_flash_attn.exe  # dopo implementazione

# Confronta con baseline
python scripts/compare_benchmarks.py --baseline baseline.txt --current current.txt
```

---

## 10. Note Aggiuntive

### Build System

Assicurarsi che i nuovi file siano aggiunti a:

- `src/CMakeLists.txt` (per libreria)
- `tests/CMakeLists.txt` (per test)

### Documentazione

Per ogni nuova funzionalità:

1. Aggiungere commenti Doxygen
2. Aggiornare `docs/api-reference.md`
3. Aggiungere esempio in `docs/getting-started.md`

### Testing

- **Unit test**: Ogni modulo deve avere >80% coverage
- **Integration test**: Test end-to-end con modello reale (es. GPT-2 small)
- **Regression test**: Salvare benchmark baseline e confrontare ad ogni commit

---

## 📊 Riferimenti

- `todo/fase11_flash_attention.md` - Specifiche Flash Attention
- `todo/fase12_profiling.md` - Strumenti di profiling
- `todo/fase30_kernel_fusion.md` - Kernel fusion patterns
- `benchmark_results.txt` - Risultati attuali

---

**Ultimo aggiornamento**: 2026-03-09  
**Responsabile**: Team Development  
**Stato**: Da implementare
