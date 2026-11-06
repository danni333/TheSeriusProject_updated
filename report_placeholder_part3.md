# Report Placeholder Part 3 - Hardcoded Values e Configurazioni

**Data:** 2026-10-29
**Focus:** Numeri magici, hardcoded values, porte, paths, timeouts, CMake options

---

## 1. Porte e Network

### 1.1 Server HTTP - Porta Hardcoded

**File:** `src/server/server.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 12-13 | Hardcoded | `int port_ = 8080;` - porta server HTTP hardcoded |

**Note:** Porta HTTP hardcoded a 8080. Dovrebbe essere configurabile via environment variable o config file.

---

### 1.2 Model Hub Server - Porta Hardcoded

**File:** `src/hub/hub_main.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| Da verificare | Hardcoded | Porta hub server probabilmente hardcoded |

**Note:** Da verificare nel file hub_main.cpp per hardcoded port.

---

## 2. Timeout e Limits

### 2.1 Plugin Manager - Limits Hardcoded

**File:** `src/hub/plugin_manager.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| Da cercare | Hardcoded | Thresholds per memory usage, CPU usage, timeout |

**Note:** Plugin manager probabilmente ha limiti hardcoded per monitoring e sandboxing.

---

### 2.2 Timeout Request

**File:** Da cercare in `src/server/`, `src/hub/`

| Tipo | Descrizione |
|------|-------------|
| Timeout | Timeout per HTTP requests probabilmente hardcoded |

---

## 3. Numeri Magici e Costanti

### 3.1 Attention - Scale Factors

**File:** `src/attention/`, `src/transformer/`

| Tipo | Descrizione |
|------|-------------|
| Scale | 1.0 / sqrt(d_model) per scaled dot product attention |
| Threshold | Threshold per sparse attention (0.7, 0.9) |

---

### 3.2 Training - Learning Rates e Decay

**File:** `src/optim/`, `src/train/`

| Tipo | Descrizione |
|------|-------------|
| LR | Learning rate hardcoded in optimizer implementations |
| Decay | Decay rates per EMA, scheduler |

---

### 3.3 Quantization - Magic Numbers

**File:** `src/quant/`

| Tipo | Descrizione |
|------|-------------|
| Scale | Scale factors per INT4/FP8 conversion |
| Clipping | Clipping thresholds per GPTQ/AWQ |

---

## 4. Paths e File Locations

### 4.1 Model Paths - Hardcoded Paths

**File:** `src/io/`, `src/hub/`

| Tipo | Descrizione |
|------|-------------|
| Paths | Percorsi default per models, cache, downloads |

---

### 4.2 Checkpoint Paths

**File:** `src/train/checkpoint.cpp`

| Tipo | Descrizione |
|------|-------------|
| Paths | Path default per checkpoint saves |

---

## 5. CMake Options Disabilitate

### 5.1 SERIUS_USE_CUDA

**File:** `CMakeLists.txt`

| Tipo | Descrizione |
|------|-------------|
| Option | SERIUS_USE_CUDA=OFF per default? Da verificare |

---

### 5.2 SERIUS_USE_METAL

**File:** `CMakeLists.txt`

| Tipo | Descrizione |
|------|-------------|
| Option | SERIUS_USE_METAL=OFF per default? Da verificare |

---

### 5.3 SERIUS_USE_VULKAN

**File:** `CMakeLists.txt`

| Tipo | Descrizione |
|------|-------------|
| Option | SERIUS_USE_VULKAN=OFF per default? Da verificare |

---

### 5.4 SERIUS_USE_NCCL

**File:** `CMakeLists.txt`

| Tipo | Descrizione |
|------|-------------|
| Option | SERIUS_USE_NCCL=OFF per default? Da verificare |

---

### 5.5 SERIUS_USE_MPI

**File:** `CMakeLists.txt`

| Tipo | Descrizione |
|------|-------------|
| Option | SERIUS_USE_MPI=OFF per default? Da verificare |

---

## 6. Ricerca Hardcoded Values - Risultati

### 6.1 Porte Network Hardcoded

**File multipli:**

| File | Linea | Valore | Descrizione |
|------|-------|--------|-------------|
| src/cli/cmd_stress.cpp | 18, 41 | 8080 | Default server URL per stress test |
| src/stress/stress_runner.h | 18 | 8080 | Default server URL |
| src/server/server.h | 25, 42 | 8080 | Default port per Server |
| src/cli/cmd_serve.cpp | 14, 30 | 8080 | Default port per serve command |
| src/app/main.cpp | 28, 41 | 8080 | Default port per serve command |
| src/hub/model_hub_server.h | 18 | 8080 | Default port per ModelHubServer |

**Totale:** 6 occorrenze di porta 8080 hardcoded

**Note:** Porta dovrebbe essere configurabile via environment variable o config file. Attualmente è hardcoded ovunque come default.

---

### 6.2 Timeout Hardcoded

**File:** `src/hub/downloader.h`

| Linea | Valore | Descrizione |
|-------|--------|-------------|
| 43 | 300 secondi | Timeout per download |

**Note:** Timeout di 5 minuti hardcoded. Dovrebbe essere configurabile.

---

### 6.3 Max Tokens Hardcoded

**Occorrenze trovate (60+ match):**

| Valore | Occorrenze | File tipici |
|--------|-----------|-------------|
| 128 | 5+ | cmd_chat.cpp, app/main.cpp |
| 256 | 10+ | cmd_bench.cpp, cmd_stress.cpp, stress_runner.h, benchmark |
| 512 | 10+ | agent/consensus.cpp, agent/supervisor.cpp, agent/planner.cpp |
| 1024 | 5+ | agent/supervisor.cpp, app/main.cpp |
| 2048 | 2+ | reasoning/cot.h |
| 4096 | 5+ | llama_inference.cpp, hub/downloader.cpp buffer |

**File principali:**
- `src/cli/cmd_chat.cpp` - max_new_tokens = 128
- `src/cli/cmd_bench.cpp` - max_tokens = 256
- `src/cli/cmd_stress.cpp` - max_tokens = 256
- `src/stress/stress_runner.h` - max_tokens = 256
- `src/agent/consensus.cpp` - max_tokens = 512
- `src/agent/supervisor.cpp` - max_tokens = 512, 1024
- `src/agent/planner.cpp` - max_tokens = 512
- `src/infer/llama_inference.cpp` - max_seq_len = 4096

**Note:** Max tokens hardcoded ovunque. Dovrebbe essere parametrizzato in config.

---

### 6.4 Thresholds Hardcoded

**Occorrenze trovate (12 match):**

| File | Valore | Descrizione |
|------|--------|-------------|
| src/stress/leak_detector.h | 10 MB | leak_threshold_bytes = 10 * 1024 * 1024 |
| src/neuro/event_engine.h | 1e-6f | weight_threshold |
| src/neuro/spiking_layer.h | 1.0f | threshold neuron |
| src/neuro/lif_neuron.h | 1.0f | threshold neuron |
| src/automl/pruner.h | 0.01f | pruning threshold |
| src/safety/bias_scorer.cpp | 0.1f | disparity threshold |
| src/safety/bias_scorer.h | 0.1f | disparity threshold |
| src/rag/retriever.h | 0.7f | similarity_threshold |
| src/rag/vector_client.h | 0.7f | threshold |
| src/attention/sparse_attention.h | 0.1f | sparse attention threshold |
| src/safety/toxicity.h | 0.8f | toxicity threshold |

**Note:** Molte thresholds hardcoded per diversi componenti. Dovrebbero essere configurabili.

---

### 6.5 Numeri Magici Piccoli (1e-X)

**Occorrenze trovate (75+ match):**

| Valore | Occorrenze | Uso tipico |
|--------|-----------|------------|
| 1e-5f | 10+ | EPSILON, tolerance, DP delta |
| 1e-6f | 20+ | Division by zero protection, normalization |
| 1e-7f | 3+ | Division by zero in tensor ops |
| 1e-8f | 5+ | Numerical stability in DNC |
| 1e-9f | 5+ | Numerical stability |

**File principali:**
- `src/infer/llama_inference.cpp` - EPSILON_DEFAULT = 1e-5F
- `src/stress/leak_detector.cpp` - 1e-6f per slope denominator
- `src/knowledge/semantic_hash.cpp` - 1e-6f per normalization
- `src/tensor/tensor.cpp` - 1e-7f per division by zero
- `src/memory/dnc.cpp` - 1e-8f per numerical stability
- `src/ssm/ssm_block.cpp` - 1e-6f per SSM division
- `src/federated/dp_noise.h` - 1e-5f per DP delta

**Note:** Numeri magici per stabilità numerica. Alcuni sono ragionevoli, altri dovrebbero essere constanti nominate.

---

### 6.6 Potenze di 2 Hardcoded

**Occorrenze trovate (100+ match limitati):**

| Valore | Uso tipico |
|--------|------------|
| 8 | AVX stride, vector operations |
| 32 | Max len tokenizer, buffer size |
| 64 | Max seq len per debug, batch size |
| 128 | Max tokens, hidden dim |
| 256 | Max tokens, hidden dim, vocab size |
| 512 | Max tokens, hidden dim |
| 1024 | Max tokens, hidden dim |
| 2048 | Max thinking tokens |
| 4096 | Max seq len, KV cache size |

**File principali:**
- `src/tokenizer/bpe.cpp` - 256 per byte values
- `src/hub/downloader.cpp` - 4096 buffer size
- `src/infer/llama_inference.cpp` - 4096 max_seq_len, KV cache
- `src/blas/matmul_avx2.cpp` - 8 per AVX operations

**Note:** Molte potenze di 2 hardcoded. Alcune sono necessarie per SIMD, altre dovrebbero essere configurabili.

---

### 6.7 IEEE 754 Bit Patterns

**File:** `src/io/gguf_loader.cpp`, `src/io/safetensors.cpp`, `src/quant/conversions.cpp`, `src/quant/fp8.cpp`

| Valore | Descrizione |
|--------|-------------|
| 0x8000u | Sign bit per FP16 |
| 0x1F | Exponent mask per FP16 |
| 0x03FF | Mantissa mask per FP16 |
| 0x7F800000u | Infinity bits per FP32 |
| 0x7FFFFF | Mantissa mask per FP32 |
| 0xFF | Exponent mask per FP32 |

**Note:** Bit patterns per conversioni FP16/FP32 sono corretti e standard. Non sono problematici.

---

## 7. CMake Options e Configurazioni

### 7.1 Opzioni SERIUS Backend - Tutte OFF per Default

**File:** `CMakeLists.txt` (righe 61-65)

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| SERIUS_USE_CUDA | OFF | CUDA backend |
| SERIUS_USE_VULKAN | OFF | Vulkan backend |
| SERIUS_USE_METAL | OFF | Metal backend |
| SERIUS_USE_NCCL | OFF | NCCL distributed backend |
| SERIUS_USE_MPI | OFF | MPI distributed backend |

**Note:** Tutti i backend hardware sono disabilitati per default. Utente deve abilitarli esplicitamente.

---

### 7.2 Dipendenze FetchContent

**File:** `CMakeLists.txt` (righe 33-49)

| Dipendenza | Versione | Stato |
|-----------|----------|-------|
| nlohmann/json | v3.11.3 | Attiva |
| spdlog | v1.14.1 | Attiva |
| googletest | v1.14.0 | Attiva |
| cpp-httplib | v0.15.3 | Attiva (header only) |
| sqlite3 | 3450300 | Dichiarata ma non usata |
| OpenSSL | Commentato | Disabilitata |
| JWT library | Commentato | Disabilitata |

**Note:** SQLite dichiarata ma non integrata (conferma findings da part2). OpenSSL e JWT completamente disabilitati.

---

## 8. Riassunto Hardcoded Values

### Totale Hardcoded Values Trovati: **200+**

**Categorie principali:**
- Porte network: 6 occorrenze di 8080
- Timeout: 1 (300 secondi)
- Max tokens: 60+ occorrenze (128, 256, 512, 1024, 2048, 4096)
- Thresholds: 12 occorrenze (0.01f, 0.1f, 0.7f, 0.8f, 1e-6f, 10MB)
- Numeri magici piccoli: 75+ occorrenze (1e-5f, 1e-6f, 1e-7f, 1e-8f, 1e-9f)
- Potenze di 2: 100+ occorrenze (8, 32, 64, 128, 256, 512, 1024, 2048, 4096)
- IEEE 754 patterns: 6 occorrenze (corretti, non problematici)

### Priorità di Configurazione

**Alta:**
- Porta 8080 → configurabile via env/config
- Max tokens → parametrizzare in config centralizzata
- Timeout → configurabile

**Media:**
- Thresholds → parametrizzare dove possibile
- Potenze di 2 per batch/seq → configurabili

**Bassa:**
- Numeri magici per stabilità numerica → ok come constanti nominate
- IEEE 754 patterns → corretti, lasciare così

---

## Totale Report Part 3: **Categorie coperte**

1. Porte network hardcoded
2. Timeout hardcoded
3. Max tokens hardcoded
4. Thresholds hardcoded
5. Numeri magici per stabilità numerica
6. Potenze di 2 hardcoded
7. CMake options backend
8. Dipendenze FetchContent



