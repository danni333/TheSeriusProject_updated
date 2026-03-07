# Fase 0 — Fondamenta del Progetto (Convenzioni, Nomenclatura, Architettura)

> Prima di scrivere una sola riga di codice, servono decisioni fondamentali.
> Questo file definisce **tutte le convenzioni** che ogni fase successiva deve rispettare.

---

## 1. Standard C++ e Compilazione

| Decisione | Scelta Consigliata | Note |
| --- | --- | --- |
| **Standard C++** | C++20 | `std::span`, concepts, `<format>`, coroutines |
| **Compilatore minimo** | GCC 12+ / Clang 16+ / MSVC 17.6+ | Tutti supportano C++20 |
| **CUDA** | CUDA 12.x (nvcc) | Per kernel GPU NVIDIA |
| **ROCm** | ROCm 6.x (hipcc) | Per GPU AMD (HIP = portabilità CUDA) |
| **Build system** | CMake 3.25+ | `cmake_minimum_required(VERSION 3.25)` |
| **Generatore** | Ninja (preferito) / Make | `cmake -G Ninja -B build` |

### Flag di compilazione

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warning rigorosi
add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)

# Release con simboli di debug
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")
```

---

## 2. Nomenclatura e Stile Codice

### Regole di naming

| Elemento | Stile | Esempio |
| --- | --- | --- |
| **Namespace** | `snake_case` | `serius::core`, `serius::tensor` |
| **Classi / Struct** | `PascalCase` | `Tensor`, `KVCache`, `MoERouter` |
| **Funzioni / Metodi** | `snake_case` | `forward()`, `compute_attention()` |
| **Variabili locali** | `snake_case` | `batch_size`, `seq_len` |
| **Variabili membro** | `snake_case_` (trailing `_`) | `data_`, `shape_`, `device_` |
| **Costanti / Enum** | `kPascalCase` | `kMaxSeqLen`, `kDefaultBatchSize` |
| **Enum class values** | `PascalCase` | `DeviceType::Cuda`, `DType::Float16` |
| **Macro (evitare)** | `SCREAMING_SNAKE` | `SERIUS_ASSERT(x)` |
| **File header** | `snake_case.h` | `tensor.h`, `kv_cache.h` |
| **File source** | `snake_case.cpp` | `tensor.cpp`, `flash_attention.cpp` |
| **File CUDA** | `snake_case.cu` | `matmul_kernel.cu` |
| **Test** | `test_snake_case.cpp` | `test_tensor.cpp` |

### Namespace globale del progetto

```cpp
namespace serius {
    namespace core { }      // Tensor, BLAS, Memory
    namespace model { }     // Transformer, MoE, Attention
    namespace quant { }     // Quantization, FP8, INT4
    namespace train { }     // Optimizer, LR Scheduler, Loss
    namespace infer { }     // KVCache, Sampler, Speculative
    namespace safety { }    // Guardrails, Toxicity
    namespace io { }        // GGUF, Safetensors, Tokenizer
    namespace hal { }       // CUDA, ROCm, Metal abstraction
    namespace agent { }     // MCP, Tool orchestration, ReAct
    namespace rl { }        // RLHF, GRPO, Reward Model
}
```

### Regole extra

- **NO `using namespace std;`** — mai, in nessun file
- **Header guards**: `#pragma once` (supportato ovunque)
- **Include order**: (1) header corrispondente, (2) headers progetto, (3) third-party, (4) STL
- **Commenti**: in inglese nel codice, italiano nella documentazione
- **Max line length**: 100 caratteri (soft limit), 120 (hard limit)
- **Indentazione**: 4 spazi (NO tabs)
- **Braces**: Allman style per classi/funzioni, K&R per `if`/`for`/`while`

```cpp
class Tensor                     // Allman per classi
{
public:
    void reshape(Shape new_shape) {  // K&R per metodi brevi
        if (new_shape.numel() != numel_) {
            throw std::invalid_argument("Shape mismatch");
        }
        shape_ = new_shape;
    }
};
```

---

## 3. Struttura Directory del Progetto

```text
TheSeriusProject/
├── CMakeLists.txt
├── LICENSE                         ← MIT o Apache 2.0
├── README.md
├── .gitignore
├── .clang-format                   ← Formatter automatico
├── .clang-tidy                     ← Linter C++
│
├── cmake/                          ← Moduli CMake custom
│   ├── FindCUDA.cmake
│   ├── FindROCm.cmake
│   └── CompilerFlags.cmake
│
├── src/
│   ├── core/                       ← Fase 1-2: Tensor, BLAS, Autograd
│   │   ├── tensor.h / .cpp
│   │   ├── blas.h / .cpp
│   │   ├── autograd.h / .cpp
│   │   └── memory_pool.h / .cpp
│   │
│   ├── hal/                        ← Fase 6: Hardware Abstraction Layer
│   │   ├── device.h                   (interfaccia astratta)
│   │   ├── cuda_backend.cu
│   │   ├── rocm_backend.cpp
│   │   └── cpu_backend.cpp
│   │
│   ├── model/                      ← Fase 3, 11, 16, 17, 36, 38
│   │   ├── transformer.h / .cpp
│   │   ├── attention.h / .cpp         (MLA, FlashAttention)
│   │   ├── ffn.h / .cpp
│   │   ├── moe.h / .cpp              (DeepSeekMoE)
│   │   ├── mamba.h / .cpp             (Mamba-2 SSD)
│   │   ├── embedding.h / .cpp
│   │   └── normalization.h / .cpp
│   │
│   ├── quant/                      ← Fase 7: Quantizzazione
│   │   ├── quantize.h / .cpp
│   │   ├── awq.h / .cpp
│   │   ├── gptq.h / .cpp
│   │   └── fp8_ops.cu
│   │
│   ├── train/                      ← Fase 5, 9, 10, 19: Training
│   │   ├── optimizer.h / .cpp
│   │   ├── lr_scheduler.h / .cpp
│   │   ├── loss.h / .cpp
│   │   ├── lora.h / .cpp
│   │   └── data_loader.h / .cpp
│   │
│   ├── infer/                      ← Fase 13, 14, 35: Inference
│   │   ├── kv_cache.h / .cpp
│   │   ├── sampler.h / .cpp
│   │   ├── speculative.h / .cpp
│   │   └── reasoning.h / .cpp        (System 2 / CoT)
│   │
│   ├── rl/                         ← Fase 23: RLHF
│   │   ├── reward_model.h / .cpp
│   │   ├── grpo.h / .cpp
│   │   └── dpo.h / .cpp
│   │
│   ├── io/                         ← Fase 4, 15: I/O
│   │   ├── tokenizer.h / .cpp
│   │   ├── gguf_loader.h / .cpp
│   │   ├── safetensors_loader.h / .cpp
│   │   └── config_loader.h / .cpp
│   │
│   ├── safety/                     ← Fase 18, 24: Safety
│   │   ├── guardrails.h / .cpp
│   │   ├── toxicity.h / .cpp
│   │   └── grounding.h / .cpp
│   │
│   ├── agent/                      ← Fase 32-34: MCP & Tools
│   │   ├── mcp_client.h / .cpp
│   │   ├── tool_registry.h / .cpp
│   │   └── react.h / .cpp
│   │
│   ├── server/                     ← Fase 13: API Server
│   │   ├── server.h / .cpp
│   │   └── api_handler.h / .cpp
│   │
│   └── debug/                      ← Fase 12: Profiling
│       ├── tensor_logger.h / .cpp
│       ├── profiler.h / .cpp
│       └── nan_debugger.h / .cpp
│
├── tests/                          ← Tutti i test
│   ├── CMakeLists.txt
│   ├── test_tensor.cpp
│   ├── test_blas.cpp
│   ├── test_attention.cpp
│   └── ...
│
├── scripts/                        ← Script di utilità
│   ├── generate_reference.py       ← NumPy reference data
│   ├── download_model.py
│   └── benchmark.sh
│
├── configs/                        ← File di configurazione modelli
│   ├── model_7b.json
│   ├── model_1b.json
│   └── training_config.json
│
├── models/                         ← Pesi scaricati (gitignored)
│   └── .gitkeep
│
├── data/                           ← Dataset (gitignored)
│   └── .gitkeep
│
├── docs/                           ← Documentazione extra
│   └── architecture.md
│
└── todo/                           ← Roadmap fasi (questo!)
    ├── fase0_fondamenta.md
    ├── fase1_motore_matematico.md
    └── ...
```

---

## 4. Dipendenze Esterne

| Libreria | Scopo | Licenza | Gestione |
| --- | --- | --- | --- |
| **nlohmann/json** | Parsing JSON (config, API) | **MIT** ✅ | Header-only, FetchContent |
| **spdlog** | Logging strutturato | **MIT** ✅ | FetchContent |
| **Google Test** | Unit testing | **BSD-3** ✅ | FetchContent |
| **cpp-httplib** | Server HTTP (REST API) | **MIT** ✅ | Header-only |
| **sentencepiece** | Tokenizer BPE | **Apache 2.0** ✅ | Build separato |
| **CUDA Toolkit 12.x** | GPU NVIDIA | **Proprietario** (uso comm. gratuito) ✅ | Sistema |
| **ROCm/HIP 6.x** | GPU AMD | **MIT** ✅ | Sistema |
| **oneTBB** (opzionale) | Thread pool CPU | **Apache 2.0** ✅ | FetchContent |

> ⚠️ **Regola ferrea**: nessuna dipendenza GPL, LGPL o copyleft.
> Solo licenze **MIT, BSD, Apache 2.0** o equivalenti permissive.

### CMake FetchContent esempio

```cmake
include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)

FetchContent_MakeAvailable(json spdlog googletest)
```

---

## 5. Formato Configurazione Modello

Tutti i modelli usano un file JSON standard. Esempio per un modello 7B:

```json
{
    "model_name": "SeriusModel-7B",
    "architecture": "transformer",
    "hidden_size": 4096,
    "num_layers": 32,
    "num_attention_heads": 32,
    "num_kv_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rope_theta": 500000.0,
    "rms_norm_eps": 1e-5,
    "attention_type": "mla",
    "moe": {
        "enabled": false,
        "num_experts": 0,
        "top_k": 0
    },
    "quantization": {
        "type": "none",
        "bits": 16
    }
}
```

---

## 6. Error Handling

| Situazione | Strategia |
| --- | --- |
| **Errori programmatore** (bug) | `SERIUS_ASSERT(cond)` → abort in Debug, no-op in Release |
| **Errori runtime recuperabili** | `std::expected<T, Error>` (C++23) o `Result<T>` custom |
| **Errori fatali** (OOM, GPU fail) | `throw serius::FatalError("msg")` |
| **Validazione input utente** | Return error code, **mai** crash |

```cpp
// Macro di assert custom
#define SERIUS_ASSERT(cond) \
    do { if (!(cond)) { \
        serius::debug::panic(__FILE__, __LINE__, #cond); \
    }} while(0)

#define SERIUS_CHECK(cond, msg) \
    do { if (!(cond)) { \
        throw serius::Error(msg, __FILE__, __LINE__); \
    }} while(0)
```

---

## 7. Logging

Usa **spdlog** con livelli standard:

```cpp
#include <spdlog/spdlog.h>

// Setup all'avvio
spdlog::set_level(spdlog::level::info);
spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");

// Uso nei moduli
spdlog::info("Loading model: {} ({} layers)", name, num_layers);
spdlog::debug("Attention output shape: {}", output.shape_str());
spdlog::warn("KV cache full, evicting oldest entries");
spdlog::error("CUDA kernel failed: {}", cuda_error_str);
```

---

## 8. Testing

| Framework | Google Test (gtest) |
| --- | --- |
| **Struttura** | Un file `test_*.cpp` per modulo |
| **Naming test** | `TEST(ModuleName, DescriptiveAction)` |
| **Dati di riferimento** | NumPy-generated binari in `test_data/` |
| **Tolleranze** | FP32: 1e-5, FP16: 1e-3, INT4: exact |
| **CI** | `ctest --output-on-failure` |

```cpp
TEST(Tensor, MatmulMatchesNumPy) {
    auto A = serius::io::load_tensor("test_data/A.bin", {64, 128});
    auto B = serius::io::load_tensor("test_data/B.bin", {128, 64});
    auto C_ref = serius::io::load_tensor("test_data/C.bin", {64, 64});
    auto C = serius::core::matmul(A, B);
    EXPECT_NEAR_TENSOR(C, C_ref, 1e-4f);
}
```

---

## 9. Versioning

| Decisione | Scelta |
| --- | --- |
| **Schema** | Semantic Versioning `MAJOR.MINOR.PATCH` |
| **Versione iniziale** | `0.1.0` |
| **Branch strategy** | `main` (stabile) + `dev` (sviluppo) + feature branches |
| **Commit messages** | Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:` |

---

## 10. Licenza

| Decisione | Scelta |
| --- | --- |
| **Licenza progetto** | **Apache 2.0** |
| **Motivazione** | Permissiva + protezione brevetti, compatibile con uso commerciale |
| **Vincolo** | Tutte le dipendenze devono essere MIT / BSD / Apache 2.0 (NO GPL/LGPL) |

---

## 11. Checklist Pre-Fase 1

- [x] Creare repository con la struttura directory sopra
- [x] Generare `CMakeLists.txt` root con FetchContent
- [x] Creare `.clang-format` con le regole di stile
- [x] Creare `.clang-tidy` con check abilitati
- [x] Scegliere licenza e creare `LICENSE`
- [x] Creare `README.md` con overview del progetto
- [x] Scrivere `.gitignore` completo
- [x] Verificare che `cmake -B build && cmake --build build` funzioni
- [x] Verificare che `ctest` esegua un test dummy
