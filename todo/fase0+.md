# Fase 0+ — Cose mancanti (Fondamenta)

Riferimento: `fase0_fondamenta.md`

---

## 1. Standard e build

| Elemento | Stato | Nota |
|----------|--------|-----|
| C++20 | ✅ Presente | `CMAKE_CXX_STANDARD 20` in CMakeLists.txt |
| CMake 3.25+ | ✅ | `cmake_minimum_required(VERSION 3.25)` |
| `.clang-format` | ❌ Mancante | Formatter automatico con regole (4 spazi, 100 char, Allman/K&R) |
| `.clang-tidy` | ❌ Mancante | Linter con check abilitati come da fase0 |
| Flag warning rigorosi | Parziale | Solo in src/; manca `-Wall -Wextra -Wpedantic -Werror=return-type` in root se desiderato |
| Ninja come generatore | Non forzato | Documentare `cmake -G Ninja -B build` nel README |

---

## 2. Struttura directory

| Previsto (fase0) | Attuale | Azione |
|------------------|---------|--------|
| `src/core/` (tensor, blas, autograd, memory) | `src/tensor/`, `src/blas/`, `src/autograd/`, `src/memory/` | Nomenclatura diversa: core vs. sottocartelle — ok se intenzionale |
| `src/model/` | `src/transformer/`, `src/attention/` | Manca namespace `serius::model` unificato |
| `src/quant/` | ✅ | Ok |
| `src/train/` | ✅ | Ok |
| `src/infer/` | Parziale | KV cache in transformer/; nessuna cartella `infer/` con sampler, speculative |
| `src/io/` | Parziale | Tokenizer in `tokenizer/`, data in `data/`; manca `io/` (gguf, safetensors, config) |
| `src/safety/` | ❌ | Non presente |
| `src/agent/` | ❌ | Non presente |
| `src/server/` | ❌ | Non presente |
| `cmake/FindCUDA.cmake`, FindROCm, CompilerFlags | ❌ | Moduli CMake custom non presenti |
| `configs/` | ❌ | Cartella config con model_7b.json ecc. assente |
| `scripts/` | ❌ | generate_reference.py, download_model.py, benchmark.sh assenti |

---

## 3. Nomenclatura e namespace

| Elemento | Stato |
|----------|--------|
| Namespace `serius::core`, `serius::model`, ecc. | Da verificare: codice usa `serius::` ma sottonamespace non allineati alla fase0 |
| Variabili membro con trailing `_` | Da verificare in tutti i moduli |
| NO `using namespace std` | Da verificare globalmente |
| Max line length 100/120 | Nessuna enforcement senza .clang-format |

---

## 4. Error handling e logging

| Elemento | Stato |
|----------|--------|
| `SERIUS_ASSERT` / `SERIUS_CHECK` | Presente in `src/common/serius_check.h` — verificare uso esteso |
| `serius::FatalError` | Da verificare se definito e usato |
| `std::expected` / Result custom (C++23) | Non richiesto obbligatorio; opzionale per API pubbliche |
| spdlog pattern e livelli | In uso; verificare pattern `[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v` |

---

## 5. Testing e versioning

| Elemento | Stato |
|----------|--------|
| Google Test, un file `test_*.cpp` per modulo | ✅ Coerente |
| Dati di riferimento in `test_data/` (NumPy binari) | Da verificare presenza e uso |
| CI con ctest | Documentare in README/docs |
| Semantic Versioning + Conventional Commits | README non cita schema versioni; branch strategy non documentata |

---

## 6. Licenza e dipendenze

| Elemento | Stato |
|----------|--------|
| Licenza Apache 2.0 | README indica Apache 2.0; file LICENSE da verificare |
| Dipendenze solo MIT/BSD/Apache (no GPL/LGPL) | json, spdlog, googletest, httplib conformi |
| sentencepiece (tokenizer) | Non in FetchContent; BPE custom presente |
| oneTBB (opzionale) | Non integrato |

---

## 7. Checklist pre-Fase 1 (riesame)

- [x] Repository e struttura directory
- [x] CMakeLists.txt con FetchContent
- [ ] `.clang-format` con regole di stile
- [ ] `.clang-tidy` con check
- [x] Licenza e README
- [ ] `.gitignore` completo (docs/ e todo/ in ignore — verificare se voluto)
- [x] Build e ctest dummy

---

**Priorità suggerite:** introdurre `.clang-format` e `.clang-tidy`, aggiungere `configs/` e `scripts/` base, documentare versioning e CI.
