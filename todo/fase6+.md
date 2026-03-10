# Fase 6+ — Cose mancanti (Ibrido CPU + HAL)

Riferimento: `fase6_ibrido_cpu_gpu.md`

---

## 1. HAL (Hardware Abstraction Layer)

| Elemento | Stato | Nota |
|----------|--------|-----|
| Interfaccia Backend: malloc, memcpy_h2d, d2h, d2d | ✅ device.h, cpu_backend, cuda_backend |
| CPU backend | ✅ cpu_backend.cpp |
| CUDA backend | ✅ cuda_backend.cu |
| Metal backend | Parziale | metal_backend.cpp: TODO MTLBuffer, buffer release, BlitCommandEncoder |
| Vulkan backend | Parziale | vulkan_backend.cpp: TODO alloc/free/map/copy |
| get_backend(DeviceType), is_available() | Da verificare | |
| test_hal.cpp | ✅ |

---

## 2. Tensor e memoria

| Elemento | Stato |
|----------|--------|
| Tensor con data_cpu_, data_gpu_, backend_, Location (CPU/DEVICE/BOTH) | Da verificare: tensor usa hal::DType e device; presenza sync esplicito da verificare |
| syncToDevice() / syncToHost() | Da verificare |
| Unified Memory vs sync manuale | Da verificare strategia attuale |

---

## 3. GPU kernel e dispatch

| Elemento | Stato |
|----------|--------|
| matmul CUDA kernel (o cuBLAS) | ✅ matmul_cuda.cu |
| Fused kernels (es. attention su GPU) | ✅ fused_kernels.cu, test_fused_cuda |
| ROCm/HIP (sostituzione cuda* → hip*) | ❌ | Nessun backend ROCm |
| Metal MPS (matmul/ops) | ❌ | Solo stub Metal |
| OpenMP + SIMD su CPU per layer piccoli | Parziale | OpenMP linkato; uso in loop da verificare |

---

## 4. File e test

| File | Stato |
|------|--------|
| hal/device.h, cpu_backend, cuda_backend, metal_backend, vulkan_backend | ✅ (metal/vulkan stub) |
| test_hal, test_cuda | ✅ |

---

**Priorità:** implementare i TODO in Metal e Vulkan (almeno alloc/copy base); documentare strategia CPU/GPU e sync; opzionale: ROCm.
