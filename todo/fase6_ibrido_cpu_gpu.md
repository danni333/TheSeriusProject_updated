# Fase 6 — Architettura Ibrida Universale (CPU + HAL: CUDA/ROCm/Metal)

> La CPU gestisce la logica complessa, il caricamento dati e i layer piccoli.
> La GPU macina le enormi moltiplicazioni di matrici dell'Attention.
> L'obiettivo è **dividere il carico** in modo ottimale.

---

## A. Gestione della Memoria Unificata

### Il problema

RAM della CPU e VRAM della GPU sono **fisicamente separate**, collegate dal bus PCIe. Trasferire dati è costoso (~12 GB/s su PCIe 3.0 x16).

### Approccio 1: CUDA Unified Memory

```cpp
float* data;
cudaMallocManaged(&data, size * sizeof(float));
// 'data' è accessibile sia da CPU che da GPU
// Il driver CUDA migra le pagine automaticamente
```

| Pro | Contro |
|---|---|
| Semplicità di codice | Overhead da page migration |
| Nessun sync manuale | Meno controllo sui trasferimenti |

### Approccio 2: Sync Manuale (consigliato per performance)

### Approccio 2: Sincronizzazione tramite Hardware Abstraction Layer (HAL)

Al fine di supportare molteplici backend (NVIDIA, AMD, Apple), il Tensor deve astenersi dal chiamare API specifiche se non tramite un'interfaccia.

```cpp
class Backend {
public:
    virtual void* malloc(size_t size) = 0;
    virtual void memcpy_h2d(void* dst, const void* src, size_t size) = 0;
    virtual void memcpy_d2h(void* dst, const void* src, size_t size) = 0;
    // ... matmul, attention, etc. ...
};

class Tensor {
    float* data_cpu_ = nullptr;
    void* data_gpu_ = nullptr;
    Backend* backend_ = nullptr;
    enum Location { CPU, DEVICE, BOTH };
    Location current_;

    void syncToDevice() {
        if (!data_gpu_) data_gpu_ = backend_->malloc(bytes());
        backend_->memcpy_h2d(data_gpu_, data_cpu_, bytes());
        current_ = BOTH;
    }
    // ...
};
```

### Strategia consigliata

Iniziare con **Unified Memory** per semplicità, poi ottimizzare i colli di bottiglia con **sync manuale** dove serve.

---

## B. Parallelismo CPU (OpenMP + SIMD)

Per i layer che restano sulla CPU (Tokenizer, Loss, layer piccoli).

### OpenMP

```cpp
// Distribuisce il loop su tutti i core
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < N; i++) {
    result[i] = expensive_computation(data[i]);
}
```

### AVX Intrinsics (recap dalla Fase 1)

```cpp
// 8 moltiplicazioni in un singolo ciclo di clock
__m256 a = _mm256_loadu_ps(&A[i]);
__m256 b = _mm256_loadu_ps(&B[i]);
__m256 c = _mm256_fmadd_ps(a, b, acc);  // acc += a * b
```

### Combinazione OpenMP + SIMD

```cpp
#pragma omp parallel for
for (int i = 0; i < N; i += 8) {
    __m256 va = _mm256_load_ps(&A[i]);
    __m256 vb = _mm256_load_ps(&B[i]);
    _mm256_store_ps(&C[i], _mm256_mul_ps(va, vb));
}
```

---

## C. Parallelismo GPU (CUDA Kernels)

### Kernel matmul semplificato

```cpp
__global__ void matmul_kernel(float* A, float* B, float* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Lancio: ogni thread computa un elemento di C
dim3 block(16, 16);
dim3 grid((N+15)/16, (M+15)/16);
matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
```

### Ottimizzazioni GPU

| Tecnica | Descrizione |
|---|---|
| **Shared Memory Tiling** | I blocchi caricano tile in shared memory (più veloce della global) |
| **Coalesced Access** | Thread adiacenti accedono a indirizzi contigui |
| **Warp-level Primitives** | `__shfl_sync` per riduzione veloce dentro un warp |
| **cuBLAS** | Fallback su NVIDIA cuBLAS per matmul ultra-ottimizzata |

### AMD ROCm (HIP)

HIP è il dialetto C++ AMD equivalente a CUDA. Il codice CUDA può essere facilmente tradotto e compilato per acceleratori AMD (Radeon/Instinct).

```cpp
// Sostituzione "cuda*" con "hip*"
hipStream_t stream1;
hipStreamCreate(&stream1);
hipMemcpyAsync(d_batch, h_batch, size, hipMemcpyHostToDevice, stream1);
matmul_kernel<<<grid, block, 0, stream1>>>(...);
```

### Apple Metal Shading Language (MPS)

Per MacBook con chip serie M, sfruttare l'Unified Memory (CPU e GPU condividono fisicamente la stessa memoria RAM).
Tramite Metal Performance Shaders (MPS), si richiamano matmul e operations native senza overhead di copia.

---

## D. Strategie Ibride

### Ibrido 1 — Tensor Storage Duale

La classe `Tensor` mantiene **due puntatori** (`data_cpu_`, `data_gpu_`) e traccia dove sono i dati aggiornati.

### Ibrido 2 — Dispatcher Automatico

```cpp
Tensor matmul(const Tensor& A, const Tensor& B) {
    size_t flops = A.rows() * A.cols() * B.cols();
    if (flops > GPU_THRESHOLD && gpu_available()) {
        A.syncToDevice();
        B.syncToDevice();
        return gpu::matmul(A, B);
    } else {
        return cpu::matmul(A, B);  // AVX2/OpenMP
    }
}
```

### Ibrido 3 — Overlap I/O

```
Timeline:
─────────────────────────────────────────────▶ tempo

CPU Thread:    [Load Batch 2] [Load Batch 3] [Load Batch 4]
GPU Stream 1:  [Compute Batch 1] [Compute Batch 2] [Compute Batch 3]
GPU Stream 2:  [Transfer Batch 1→Host]  [Transfer Batch 2→Host]

Nessun tempo morto!
```

### Ibrido 4 — Pipeline Parallelism

Per modelli molto profondi, split dei layer tra dispositivi:

```
GPU 0:  Layer 0-5  (Forward → invia a GPU 1)
GPU 1:  Layer 6-11 (Forward → invia a CPU per loss)
CPU:    Loss + Backward start → invia gradiente a GPU 1
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/device/device.h` | Enum `Device {CPU, CUDA}`, detection |
| `src/device/cuda_utils.h/.cu` | Helper CUDA (error check, stream mgmt) |
| `src/device/tensor_cuda.cu` | `syncToDevice()`, `syncToHost()`, allocazione GPU |
| `src/blas/matmul_cuda.cu` | Kernel CUDA matmul + tiling |
| `src/blas/dispatch.cpp` | Dispatcher CPU vs GPU (aggiornato) |
| `src/transformer/attention_cuda.cu` | Attention kernel GPU |
| `src/training/trainer.cpp` | Training loop ibrido (aggiornato) |
| `tests/test_cuda.cpp` | Unit test GPU: allocazione, sync, matmul |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_sync` | `syncToDevice → syncToHost`: i dati sono identici |
| `test_matmul_cuda` | matmul GPU vs CPU: risultati ≈ uguali (atol 1e-5) |
| `test_dispatcher` | Tensori piccoli → CPU, tensori grandi → GPU |
| `test_streams` | Due kernel su stream diversi → overlap reale |
| `test_attention_cuda` | Attention GPU produce stesso output di Attention CPU |

### Benchmark ibrido

| Benchmark | Cosa misura |
|---|---|
| `bench_cpu_vs_gpu` | Matmul 1024×1024: speedup GPU vs CPU |
| `bench_pipeline` | Throughput training con overlap I/O vs sequenziale |
| `bench_dispatcher` | Overhead del dispatch automatico |

---

## 7. Backend Vulkan (GPU Cross-Vendor)

> Alternativa a CUDA/ROCm: funziona su **qualsiasi GPU** (AMD, Intel, NVIDIA)
> senza SDK proprietari. Più portabile, meno performance di CUDA ma copre
> hardware che altrimenti non sarebbe supportato.

### Perché Vulkan

| Backend | Pro | Contro |
| --- | --- | --- |
| **CUDA** | Massime performance, ecosystem maturo | Solo NVIDIA |
| **ROCm** | Buone performance su AMD | Setup complesso, solo Linux |
| **Vulkan** | Cross-vendor, cross-OS | ~70-80% performance di CUDA |
| **Metal** | Ottimo su Apple Silicon | Solo macOS/iOS |

### Implementazione con Vulkan Compute Shaders

```cpp
namespace serius::hal::vulkan {

class VulkanBackend : public IBackend {
    VkDevice device_;
    VkQueue compute_queue_;
    VkPipeline matmul_pipeline_;
    VkPipeline softmax_pipeline_;

public:
    bool init() override;
    void matmul(const Tensor& A, const Tensor& B, Tensor& C) override;
    void softmax(const Tensor& x, Tensor& out) override;
    Tensor alloc(Shape shape, DType dtype) override;
    void free(Tensor& t) override;

    // Carica compute shader SPIR-V
    VkPipeline load_shader(const string& spirv_path);
};

} // namespace serius::hal::vulkan
```

### Struttura File Vulkan

| File | Contenuto |
| --- | --- |
| `src/hal/vulkan_backend.h/.cpp` | Backend Vulkan compute |
| `src/hal/shaders/matmul.comp` | GLSL compute shader matmul |
| `src/hal/shaders/softmax.comp` | GLSL compute shader softmax |
| `cmake/FindVulkan.cmake` | CMake module per Vulkan SDK |

---

## 8. Requisiti aggiuntivi

| Requisito | Minimo |
|---|---|
| GPU NVIDIA | Compute Capability ≥ 6.0 (Pascal) |
| CUDA Toolkit | 11.0+ |
| Driver NVIDIA | 450+ |
| CMake CUDA | `enable_language(CUDA)` in CMakeLists.txt |

> [!CAUTION]
> La Fase 6 è **opzionale** se non si dispone di una GPU NVIDIA.
> Il modello funziona interamente su CPU con le Fasi 1-5.
> La GPU accelera il training di 10-100x ma non è strettamente necessaria.
