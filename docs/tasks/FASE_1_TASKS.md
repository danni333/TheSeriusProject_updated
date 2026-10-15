# ⚡ FASE 1 — Fondamenta d'Acciaio
**Periodo**: Q2 2026 (Aprile - Giugno) | **Tasks**: 28 | **Completate**: 0

---

## 1.1 CPU Optimization

### T1.01 — Thread Pool con Work-Stealing Scheduler
- **Priorità**: 🔴 Critica
- **Stima**: 3d
- **File**: `src/common/thread_pool.h`, `src/common/thread_pool.cpp` (NUOVI)
- **Descrizione**: Implementare un thread pool che distribuisce lavoro tra N thread. Ogni thread ha una coda locale; quando la coda è vuota, "ruba" task da altri thread.
- **Acceptance Criteria**:
  - [ ] Thread pool con N configurabile (default: `std::thread::hardware_concurrency()`)
  - [ ] API: `pool.submit(callable)` → `std::future<Result>`
  - [ ] Work-stealing quando una coda è vuota
  - [ ] Benchmark: matmul 2048x2048 single-thread vs pool → ≥3x speedup su 4-core
- **Dipendenze**: Nessuna

---

### T1.02 — Parallelizzare Matmul Multi-Thread
- **Priorità**: 🔴 Critica
- **Stima**: 2d
- **File**: `src/blas/matmul_avx2.cpp`, `src/blas/dispatch.cpp`
- **Descrizione**: Partizionare le righe della matrice A tra thread del pool. Ogni thread calcola un chunk di righe del risultato C.
- **Acceptance Criteria**:
  - [ ] Matmul parallelo con `#pragma omp parallel for` o thread pool
  - [ ] TinyLlama forward pass: da ~1.4 tok/s a ≥5 tok/s
  - [ ] No race conditions (test con ThreadSanitizer)
- **Dipendenze**: T1.01

---

### T1.03 — Benchmark Harness
- **Priorità**: 🔴 Critica
- **Stima**: 2d
- **File**: `tests/bench_inference.cpp` (NUOVO)
- **Descrizione**: Script che misura: tokens/secondo, peak memory, latency per token, tempo di caricamento del modello. Output in formato CSV per tracking storico.
- **Acceptance Criteria**:
  - [ ] Misura tok/s (generazione), TTFT (time-to-first-token), peak RSS
  - [ ] Output CSV con timestamp per comparazione temporale
  - [ ] Supporta `--model`, `--device`, `--prompt`, `--max-tokens`
  - [ ] Baseline registrata per TinyLlama su CPU
- **Dipendenze**: Nessuna

---

### T1.04 — GGUF Format Importer
- **Priorità**: 🟡 Media
- **Stima**: 4d
- **File**: `src/io/gguf_loader.h`, `src/io/gguf_loader.cpp` (NUOVI)
- **Descrizione**: Parser per il formato GGUF (usato da llama.cpp e l'intero ecosistema). Permette di caricare qualsiasi modello quantizzato dalla community.
- **Acceptance Criteria**:
  - [ ] Parsing header GGUF v3
  - [ ] Lettura metadata (architettura, n_layers, vocab, etc.)
  - [ ] Lettura tensori con tipo (F32, F16, Q4_0, Q4_1, Q8_0)
  - [ ] Conversione automatica in formato Serius Tensor
  - [ ] Test: caricare TinyLlama GGUF Q4_0 e fare inference
- **Dipendenze**: Nessuna (indipendente)

---

### T1.05 — AVX-512 / AMX Testing
- **Priorità**: 🟢 Bassa
- **Stima**: 2d
- **File**: `src/blas/matmul_avx512.cpp`
- **Descrizione**: Verificare e ottimizzare i kernel AVX-512 esistenti. Testare su CPU con supporto AMX (Intel Sapphire Rapids+).
- **Acceptance Criteria**:
  - [ ] Benchmark AVX2 vs AVX-512 su matmul 4096x4096
  - [ ] Fix eventuali bug nel kernel AVX-512
  - [ ] Documentare speedup misurato
- **Dipendenze**: T1.03

---

## 1.2 CUDA Backend

### T1.06 — CUDA Stream Management
- **Priorità**: 🟠 Alta
- **Stima**: 3d
- **File**: `src/hal/cuda_backend.h`, `src/hal/cuda_backend.cu`
- **Descrizione**: Aggiungere supporto per CUDA streams nell'IBackend. Ogni operazione accetta un stream opzionale per esecuzione asincrona.
- **Acceptance Criteria**:
  - [ ] `IBackend` esteso con `create_stream()`, `sync_stream()`, `destroy_stream()`
  - [ ] Memcpy async con stream
  - [ ] Test: 2 matmul in parallelo su 2 stream → throughput ≥1.5x vs sequenziale
- **Dipendenze**: Nessuna

---

### T1.07 — Kernel CUDA: RMSNorm
- **Priorità**: 🟠 Alta
- **Stima**: 2d
- **File**: `src/hal/fused_kernels.cu` (estendere)
- **Descrizione**: CUDA kernel per RMSNorm: `y = x * rsqrt(mean(x²) + eps) * gamma`. Usare shared memory per la riduzione.
- **Acceptance Criteria**:
  - [ ] Kernel corretto (confronto vs CPU reference, max diff < 1e-5)
  - [ ] Support per FP32 e FP16
  - [ ] Benchmark: ≥10x vs CPU per tensore [1, 4096]
- **Dipendenze**: T1.06

---

### T1.08 — Kernel CUDA: SwiGLU
- **Priorità**: 🟠 Alta
- **Stima**: 1d
- **File**: `src/hal/fused_kernels.cu` (estendere)
- **Descrizione**: Fused kernel: `SwiGLU(x) = silu(gate) * up` dove gate e up sono due proiezioni lineari. Fonde la moltiplicazione element-wise con l'attivazione.
- **Acceptance Criteria**:
  - [ ] Kernel corretto (confronto vs CPU)
  - [ ] Fused: gate proj + silu + element-wise mul in un unico kernel
- **Dipendenze**: T1.06

---

### T1.09 — Kernel CUDA: RoPE
- **Priorità**: 🟡 Media
- **Stima**: 1d
- **File**: `src/hal/fused_kernels.cu` (estendere)
- **Descrizione**: CUDA kernel per Rotary Positional Embedding. Applica rotazione complessa ai vettori Q e K in-place.
- **Acceptance Criteria**:
  - [ ] Kernel corretto (confronto vs `positional.cpp`)
  - [ ] In-place: nessuna allocazione aggiuntiva
- **Dipendenze**: T1.06

---

### T1.10 — Kernel CUDA: Softmax Standalone
- **Priorità**: 🟠 Alta
- **Stima**: 1d
- **File**: `src/hal/fused_kernels.cu` (estendere)
- **Descrizione**: Softmax CUDA kernel generale (non solo per attention). `y[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))`.
- **Acceptance Criteria**:
  - [ ] Numericamente stabile (sottrae max prima di exp)
  - [ ] Supporta dimensioni arbitrarie lungo l'ultimo asse
- **Dipendenze**: T1.06

---

### T1.11 — GPU Memory Pool
- **Priorità**: 🟠 Alta
- **Stima**: 3d
- **File**: `src/memory/gpu_memory_pool.h`, `src/memory/gpu_memory_pool.cpp` (NUOVI)
- **Descrizione**: Sub-allocator per GPU che alloca un grande blocco CUDA e lo suddivide internamente, evitando chiamate costose a `cudaMalloc` per ogni tensor piccolo.
- **Acceptance Criteria**:
  - [ ] Alloca un pool iniziale configurabile (default 1GB)
  - [ ] Sub-allocation con best-fit o buddy system
  - [ ] Free list per riuso
  - [ ] Benchmark: 1000 alloc/free → ≥10x vs `cudaMalloc` diretto
- **Dipendenze**: T1.06

---

### T1.12 — Multi-GPU Device Selection
- **Priorità**: 🟡 Media
- **Stima**: 2d
- **File**: `src/hal/cuda_backend.cu`
- **Descrizione**: Supporto per selezionare e gestire multiple GPU. `--device cuda:0`, `--device cuda:1`.
- **Acceptance Criteria**:
  - [ ] `cudaSetDevice(id)` gestito correttamente
  - [ ] Ogni GPU ha il suo memory pool
  - [ ] Peer-to-peer transfer tra GPU (se supportato)
- **Dipendenze**: T1.11

---

### T1.13 — LLaMA Inference su CUDA End-to-End
- **Priorità**: 🔴 Critica
- **Stima**: 5d
- **File**: `src/infer/llama_inference.cpp` (estendere)
- **Descrizione**: Far girare l'intero forward pass di TinyLlama su GPU CUDA. Richiede che tutti i tensori vivano in GPU memory e tutti i kernel CUDA siano funzionali.
- **Acceptance Criteria**:
  - [ ] `serius chat --type llama --device cuda` funziona
  - [ ] TinyLlama ≥100 tok/s su RTX 3060
  - [ ] Output coerente (uguale a CPU, modulo precision)
  - [ ] Peak GPU memory ≤3 GB per TinyLlama
- **Dipendenze**: T1.06, T1.07, T1.08, T1.09, T1.10, T1.11

---

### T1.14 — CUDA Graph Capture
- **Priorità**: 🟢 Bassa
- **Stima**: 3d
- **File**: `src/hal/cuda_graph.h` (NUOVO)
- **Descrizione**: Catturare il forward pass come CUDA Graph per eliminare overhead di launch dei kernel. Utile per single-token generation dove il launch overhead domina.
- **Acceptance Criteria**:
  - [ ] Capture + replay del forward pass per singolo token
  - [ ] Speedup ≥10% su generazione singolo token
- **Dipendenze**: T1.13

---

## 1.3 Flash Attention v2+

### T1.15 — Sliding Window Attention
- **Priorità**: 🟠 Alta
- **Stima**: 3d
- **File**: `src/attention/flash_attn.h`, `src/attention/flash_attn.cpp`
- **Descrizione**: Aggiungere modalità sliding window a Flash Attention. Ogni token attende solo agli ultimi W token (es. W=4096).
- **Acceptance Criteria**:
  - [ ] `FlashAttentionConfig.window_size` parametro
  - [ ] CPU implementation corretta (test vs naive)
  - [ ] KV-cache limitato a W token per layer → risparmio memoria
- **Dipendenze**: Nessuna

---

### T1.16 — Flash Attention: Local + Global Interleaved
- **Priorità**: 🟠 Alta
- **Stima**: 3d
- **File**: `src/attention/flash_attn.h`
- **Descrizione**: Implementare il pattern Gemma-style: alcuni layers usano sliding window (local), altri usano full attention (global).
- **Acceptance Criteria**:
  - [ ] `FlashAttentionConfig.attention_type` = LOCAL | GLOBAL
  - [ ] Local: sliding window di W token
  - [ ] Global: full causal attention
  - [ ] Test: alternare local/global ogni 4 layers
- **Dipendenze**: T1.15

---

### T1.17 — Flash Attention CUDA: Sliding Window
- **Priorità**: 🟡 Media
- **Stima**: 3d
- **File**: `src/attention/flash_attn_cuda.cu`
- **Descrizione**: Port del sliding window attention su CUDA.
- **Acceptance Criteria**:
  - [ ] Output identico alla versione CPU (max diff < 1e-4)
  - [ ] Performance: ≥2x vs full attention per seq_len > window_size
- **Dipendenze**: T1.15, T1.06

---

## 1.4 Quantizzazione Avanzata

### T1.18 — Integrare INT4 nell'Inference Path
- **Priorità**: 🟠 Alta
- **Stima**: 4d
- **File**: `src/quant/int4.cpp`, `src/infer/llama_inference.cpp`
- **Descrizione**: Il modulo INT4 esiste ma non è integrato nel path di inference. Aggiungere: carica weights in INT4, dequantizza on-the-fly durante matmul.
- **Acceptance Criteria**:
  - [ ] `serius chat --type llama --quant int4` funziona
  - [ ] Riduzione memoria ≥3x (da FP32)
  - [ ] Perplexity degradation ≤5% vs FP32
  - [ ] Performance: ≥1.5x tok/s vs FP32 (meno dati da spostare)
- **Dipendenze**: T1.03

---

### T1.19 — BitNet 1.58-bit Matmul
- **Priorità**: 🟠 Alta
- **Stima**: 5d
- **File**: `src/quant/bitnet.h`, `src/quant/bitnet.cpp`, `src/blas/matmul_bitnet.cpp` (NUOVI)
- **Descrizione**: Implementare matmul per pesi ternari {-1, 0, +1}. Nessuna moltiplicazione: solo add/sub/skip.
- **Acceptance Criteria**:
  - [ ] Struct `BitNetWeight`: packing ternario (2 bit per peso)
  - [ ] `matmul_bitnet(input_fp32, weights_ternary) → output_fp32`
  - [ ] Correttezza: output identico a matmul standard con pesi {-1,0,+1}
  - [ ] Performance: ≥3x vs FP32 matmul (no FMA, solo add/sub)
  - [ ] Variante AVX2: usa `_mm256_add_ps` / `_mm256_sub_ps` con mask
- **Dipendenze**: Nessuna

---

### T1.20 — FP4 Emulazione Software
- **Priorità**: 🟡 Media
- **Stima**: 3d
- **File**: `src/quant/fp4.h`, `src/quant/fp4.cpp` (NUOVI)
- **Descrizione**: Emulazione del formato NVFP4 (E2M1) in software. Utile per testare modelli quantizzati senza hardware Blackwell.
- **Acceptance Criteria**:
  - [ ] Conversione FP32 ↔ FP4
  - [ ] Matmul con dequant on-the-fly
  - [ ] Test: perplexity di TinyLlama FP4 vs FP32
- **Dipendenze**: Nessuna

---

### T1.21 — Auto-Calibration Pipeline
- **Priorità**: 🟡 Media
- **Stima**: 3d
- **File**: `src/quant/calibration.h`, `src/quant/calibration.cpp` (NUOVI)
- **Descrizione**: Dato un modello e un dataset di calibrazione, determinare automaticamente i range di quantizzazione ottimali per ogni layer.
- **Acceptance Criteria**:
  - [ ] Esegui N forward pass con dati di calibrazione
  - [ ] Registra min/max per ogni tensor peso
  - [ ] Produce scale/zero-point per INT4/INT8/FP8
  - [ ] Output: file di calibrazione JSON
- **Dipendenze**: T1.18

---

## 1.5 Infrastruttura & Testing

### T1.22 — Test Suite per HAL CUDA
- **Priorità**: 🟠 Alta
- **Stima**: 2d
- **File**: `tests/test_cuda.cpp` (estendere)
- **Descrizione**: Test completi per tutti i kernel CUDA: matmul, RMSNorm, SwiGLU, RoPE, softmax, flash attention.
- **Acceptance Criteria**:
  - [ ] Ogni kernel testato con input noti e output verificato vs CPU
  - [ ] Test per edge cases: dimensioni non power-of-2, seq_len=1, batch=1
  - [ ] CI-friendly: skip se CUDA non disponibile
- **Dipendenze**: T1.07-T1.10

---

### T1.23 — Test Suite per Quantizzazione
- **Priorità**: 🟠 Alta
- **Stima**: 2d
- **File**: `tests/test_quantization.cpp` (NUOVO)
- **Descrizione**: Test per INT4, FP8, BitNet, FP4: correttezza di quantize/dequantize, matmul quantizzata, perplexity check.
- **Acceptance Criteria**:
  - [ ] Round-trip error: quantize→dequantize con errore misurabile
  - [ ] Matmul quantizzata: output vicino a FP32 (RMSE < threshold)
  - [ ] BitNet: verifica che usa solo add/sub
- **Dipendenze**: T1.18, T1.19, T1.20

---

### T1.24 — CI Pipeline Setup
- **Priorità**: 🟡 Media
- **Stima**: 2d
- **File**: `.github/workflows/ci.yml` (NUOVO)
- **Descrizione**: GitHub Actions CI che compila il progetto e esegue i test su ogni push/PR.
- **Acceptance Criteria**:
  - [ ] Build su Linux (GCC) e Windows (MSVC)
  - [ ] Esegui tutti i test (skip CUDA se non disponibile)
  - [ ] Report test pass/fail
  - [ ] Build time ≤5 min
- **Dipendenze**: Nessuna

---

### T1.25 — Documentazione API dei Kernels
- **Priorità**: 🟢 Bassa
- **Stima**: 1d
- **File**: `docs/api/kernels.md` (NUOVO)
- **Descrizione**: Documentazione di tutte le operazioni disponibili (BLAS, fused kernels, quantization) con signature, input/output, e complessità.
- **Acceptance Criteria**:
  - [ ] Ogni funzione pubblica documentata
  - [ ] Esempi di utilizzo per le operazioni principali
- **Dipendenze**: T1.07-T1.10

---

### T1.26 — Performance Regression Test
- **Priorità**: 🟠 Alta
- **Stima**: 2d
- **File**: `tests/bench_regression.cpp` (NUOVO)
- **Descrizione**: Test che registra performance baseline e fallisce se una nuova commit degrada le performance oltre una soglia.
- **Acceptance Criteria**:
  - [ ] Baseline registrata in `tests/baselines.json`
  - [ ] Test fallisce se tok/s cala >10% rispetto alla baseline
  - [ ] Aggiornamento automatico baseline con `--update-baseline`
- **Dipendenze**: T1.03

---

### T1.27 — Memory Leak Detection
- **Priorità**: 🟡 Media
- **Stima**: 1d
- **File**: `tests/test_memory_leaks.cpp` (NUOVO)
- **Descrizione**: Test che verifica assenza di memory leak durante inference (100 token di generazione, RSS prima e dopo deve essere uguale ±margine).
- **Acceptance Criteria**:
  - [ ] RSS delta ≤ 1 MB dopo 100 token generation
  - [ ] GPU memory delta = 0 dopo generation complete
- **Dipendenze**: T1.13

---

### T1.28 — CLI: Aggiungere `--device` e `--quant` flags
- **Priorità**: 🟠 Alta
- **Stima**: 1d
- **File**: `src/app/main.cpp`
- **Descrizione**: Aggiungere flag CLI per selezionare device (cpu/cuda/cuda:N) e tipo di quantizzazione (none/int4/int8/fp8/bitnet).
- **Acceptance Criteria**:
  - [ ] `serius chat --device cuda --quant int4`
  - [ ] Auto-detect: se CUDA disponibile, default a cuda; altrimenti cpu
  - [ ] Errore chiaro se device richiesto non disponibile
- **Dipendenze**: T1.13, T1.18
