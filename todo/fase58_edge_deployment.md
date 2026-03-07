# Fase 58 — Edge Deployment (Mobile, Raspberry Pi, Browser)

> Un modello che gira solo su workstation con GPU da 1000€ non è per tutti.
> Ottimizzazioni per girare su dispositivi con risorse limitate:
> smartphone, Raspberry Pi 5, e persino nel browser via WebAssembly.

---

## 1. Target Hardware

| Dispositivo | RAM | Compute | Modello Consigliato |
| --- | --- | --- | --- |
| **Raspberry Pi 5** | 8 GB | ARM Cortex-A76 | Qwen2.5-0.5B Q4 |
| **Android** (Snapdragon 8 Gen 3) | 12 GB | Adreno 750 GPU | Phi-3-mini Q4 (3.8B) |
| **iOS** (Apple A17/M-series) | 8 GB | ANE + Metal GPU | Gemma-2B Q4 |
| **Browser** (WebAssembly) | 4 GB tab limit | WASM SIMD | SmolLM-360M Q8 |
| **Laptop senza GPU** | 16 GB | x86 AVX2 | Llama-3.2-3B Q4 |

---

## 2. Ottimizzazioni per Edge

### 2.1 Compilazione ARM NEON

```cmake
# Cross-compile per ARM64 (Raspberry Pi / Android)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+fp16")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16")
```

### 2.2 WebAssembly (WASM)

```bash
# Compila con Emscripten per il browser
emcmake cmake -B build-wasm \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_WASM=ON \
    -DUSE_WASM_SIMD=ON \
    -DUSE_THREADING=ON

cmake --build build-wasm
# Output: serius.wasm + serius.js (loader)
```

### 2.3 Android NDK

```bash
# Build per Android ARM64
cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release
```

---

## 3. Struttura File

| File | Contenuto |
| --- | --- |
| `cmake/CrossCompile.cmake` | Toolchain ARM, WASM, Android |
| `src/hal/arm_neon_backend.cpp` | Kernel ottimizzati NEON |
| `src/hal/wasm_backend.cpp` | Backend WebAssembly SIMD |
| `wasm/index.html` | Demo browser con WebWorker |
| `wasm/worker.js` | Worker thread per inference WASM |
| `scripts/build_android.sh` | Build script Android |
| `scripts/build_rpi.sh` | Build script Raspberry Pi |

---

## 4. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_arm_neon` | Matmul NEON produce risultati corretti |
| `test_wasm_build` | Build WASM produce .wasm valido |
| `test_wasm_inference` | Inference nel browser genera testo coerente |
| `test_rpi_benchmark` | ≥ 5 tok/s su RPi5 con modello 0.5B Q4 |
| `test_android_build` | APK si installa e esegue su emulatore |
