# Report Placeholder - TheSeriusProject

**Data:** 2026-10-29
**Ricerca:** Pattern `TODO|FIXME|XXX|HACK|PLACEHOLDER|placeholder|implementare|da implementare|Not implemented|to be done|stub|incomplete|implementation pending`
**Totale match:** 55+ (esteso con ulteriori pattern)

---

## 1. Placeholder nel Codice Sorgente (Codice da Implementare)

### 1.1 Backend Hardware - Metal (2 TODO)

**File:** `src/hal/metal_backend.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 14 | TODO | Implementare Metal MTLBuffer allocation |
| 21 | TODO | Implementare Metal buffer release |

**Note:** Il backend Metal è incompleto. Mancano `malloc_pinned`, `malloc_managed`, `free_pinned`, device-to-host/host-to-device copies, stream lifecycle. `is_available()` ritorna sempre false. Richiede toolchain macOS/Metal.

---

### 1.2 Backend Hardware - Vulkan (4 TODO)

**File:** `src/hal/vulkan_backend.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 14 | TODO | Implementare Vulkan memory allocation via VMA or native Vulkan APIs |
| 21 | TODO | Implementare Vulkan memory free |
| 27 | TODO | Implementare Vulkan buffer mapping/copying (h2d) |
| 34 | TODO | Implementare Vulkan buffer mapping/copying (d2h) |

**Note:** Il backend Vulkan è solo stub. Mancano le stesse funzionalità di Metal. Il device-to-device copying usa `std::memcpy` (non valido per memoria Vulkan). Richiede Vulkan SDK e runtime/device.

---

### 1.3 Stress Testing - Monitoraggio Memoria (2 Placeholder)

**File:** `src/stress/stress_runner.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 267 | Placeholder | Integrate with CUDA backend for actual VRAM monitoring (ritorna 0) |
| 273 | Placeholder | Integrate with OS-specific memory APIs (ritorna 0) |

**Note:** Funzioni di monitoraggio attualmente restituiscono 0. Richiedono integrazione con backend CUDA e API OS-specific.

---

### 1.4 Pipeline Distribuita - Gradiente Placeholder (1)

**File:** `src/distributed/pipeline.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 76-78 | Placeholder | Usa placeholder di 1.0 per il gradiente (il vero gradiente di loss sarebbe calcolato dal caller) |

**Note:** Nel backward pass dell'ultimo stage, il gradiente viene inizializzato a 1.0f come placeholder. Implementazione completa richiede calcolo reale del gradiente di loss.

---

### 1.5 Inference LLaMA - Tensor Placeholder (1)

**File:** `src/infer/llama_inference.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 480 | Placeholder | `return Tensor({0})` per intermediate ranks in pipeline parallel |

**Note:** In pipeline parallel, i rank intermedi ritornano un tensor vuoto. Serve implementazione corretta del passaggio output tra rank.

---

### 1.6 Hub Converter - Safetensors to GGUF (1)

**File:** `src/hub/converter.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 31-33 | Placeholder | Implementazione completa richiede parsing Safetensors e scrittura GGUF |

**Note:** La funzione di conversione è solo un placeholder con stampa a console. Mancano: parsing Safetensors, estrazione tensori e metadata, scrittura GGUF.

---

### 1.7 SSM Block - Input Projection Placeholder (1)

**File:** `src/ssm/ssm_block.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 20 | Placeholder | `in_proj_ = core::Tensor({1})` - tensor placeholder |

**Note:** Input projection tensor inizializzato come placeholder con dimensione 1. Richiede implementazione corretta delle dimensioni del modello.

---

### 1.8 MCP HTTP Transport - Not Implemented (1)

**File:** `src/mcp/transport_http.cpp` e `src/mcp/transport_http.h`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 24 | Not implemented | HTTP transport non implementato (ritorna errore JSON) |

**Note:** Transport HTTP è solo stub. Richiede libreria client HTTP (cpp-httplib/curl) come dipendenza opzionale.

---

### 1.9 Graph Compiler - Trace Not Implemented (1)

**File:** `src/optim/graph_compiler.h`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 61 | Not implemented | Metodo `trace()` per dummy forward pass non implementato |

**Note:** Funzione di tracing dichiarata ma non implementata in questa versione.

---

### 1.10 Flash Attention Backward - CPU Not Implemented (1)

**File:** `src/attention/flash_attn.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 191 | Not implemented | Flash attention backward non implementato su CPU |

**Note:** Il backward pass di flash attention lancia eccezione su CPU. Richiede implementazione CPU o fallback a standard attention.

---

### 1.11 Hub Search Engine - Multiple Stub Methods (13)

**File:** `src/hub/search_engine.cpp`

| Metodo | Tipo | Descrizione |
|--------|------|-------------|
| search_models() | Stub | `return {}` - implementazione ricerca base |
| semantic_search() | Stub | `return {}` - ricerca semantica |
| find_similar_models() | Stub | `return {}` - modelli simili |
| get_suggestions() | Stub | `return {}` - suggerimenti query |
| get_trending_searches() | Stub | `return {}` - ricerche trending |
| log_search() | Stub | Metodo vuoto - logging ricerche |
| get_popular_queries() | Stub | `return {}` - query popolari |
| compute_query_embedding() | Stub | `return {}` - embedding query |
| get_cached_results() | Stub | `return {}` - risultati cache |
| fulltext_search() | Stub | `return {}` - ricerca fulltext |
| tag_search() | Stub | `return {}` - ricerca per tag |
| author_search() | Stub | `return {}` - ricerca per autore |
| merge_and_rank_results() | Stub | `return {}` - merge ranking |

**Note:** SearchEngine è completamente stub. Richiede implementazione completa di tutti i metodi di ricerca e indexing.

---

### 1.12 Hub Embedding Manager - Stub Methods (3)

**File:** `src/hub/search_engine.cpp`

| Metodo | Tipo | Descrizione |
|--------|------|-------------|
| compute_embeddings() | Stub | `return {}` - batch embeddings |
| load_model() | Stub | `return false` - carica modello embedding |
| compute_with_sentence_transformer() | Stub | `return {}` - SentenceTransformer |
| compute_with_simple_tokenizer() | Stub | `return {}` - tokenizer semplice |

**Note:** EmbeddingManager non ha implementazione. Richiede integrazione con modello embedding reale.

---

### 1.13 Model Hub Server - Search Stub (1)

**File:** `src/hub/model_hub_server.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 116 | Stub | `search_models()` ritorna `{}` |

**Note:** Funzione di ricerca nel server hub non implementata.

---

### 1.14 CLI Commands - Implementation Pending (3)

**File:** `src/cli/cmd_serve.cpp`, `src/cli/cmd_doctor.cpp`

| Comando | Linea | Tipo | Descrizione |
|---------|-------|------|-------------|
| cmd_serve | 38 | Implementation pending | Server mode non implementato |
| cmd_doctor | 32 | Implementation pending | Diagnostics GPU/RAM non implementato |

**Note:** Comandi CLI dichiarati ma con stampa solo "implementation pending".

---

### 1.15 Backend Availability Stubs (2)

**File:** `src/hal/vulkan_backend.cpp`, `src/hal/metal_backend.cpp`

| File | Linea | Tipo | Descrizione |
|------|-------|------|-------------|
| vulkan_backend.cpp | 76 | Stub | `is_available()` ritorna `false` |
| metal_backend.cpp | 76 | Stub | `is_available()` ritorna `false` |

**Note:** Entrambi i backend ritornano sempre false perché non completamente implementati.

---

### 1.16 MCP Transport Stdio - Stub Implementation (1)

**File:** `src/mcp/transport_stdio.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 18 | Stub | Constructor con `(void)command_` - subprocess bidirectional non implementato |
| 34-35 | Stub | `send()` con `(void)method; (void)params` - ritorna errore JSON |

**Note:** Transport stdio è stub perché manca implementazione bidirectional pipes.

---

### 1.17 CPU Backend - Stream Operations No-Op (5)

**File:** `src/hal/cpu_backend.cpp`

| Metodo | Linea | Tipo | Descrizione |
|--------|-------|------|-------------|
| copy_async | 37 | No-op | `(void)stream` - stream non usato |
| destroy_stream | 135 | No-op | `(void)stream` - operazione vuota |
| sync_stream | 141 | No-op | `(void)stream` - CPU è sincrono |
| memcpy_h2d_async | 147 | Fallback | `(void)stream` - chiama memcpy_h2d sincrono |
| memcpy_d2h_async | 153 | Fallback | `(void)stream` - chiama memcpy_d2h sincrono |
| memcpy_d2d_async | 159 | Fallback | `(void)stream` - chiama memcpy_d2d sincrono |

**Note:** CPU backend non supporta vera asincronicità con stream. Le operazioni async degradano a sync.

---

### 1.18 BLAS Dispatch - Stream Parameter Not Used (2)

**File:** `src/blas/dispatch.cpp`

| Metodo | Linea | Tipo | Descrizione |
|--------|-------|------|-------------|
| matmul | 133 | Non usato | `(void)stream` - stream parameter ignorato |
| batched_matmul | 235 | Non usato | `(void)stream` - stream parameter ignorato |

**Note:** Funzioni BLAS accettano stream parameter ma non lo usano. Implementazione CUDA dovrebbe usarlo.

---

### 1.19 Quantization FP8 Kernel - Stub (1)

**File:** `src/quant/quant_kernels.cu`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 155 | Stub | `float_to_fp8_e4m3_cuda()` con `(void)src; (void)dst; (void)n` |

**Note:** Kernel FP8 è stub con commento "(Reuse existing kernel or implement one specialized)". Non implementa conversione reale.

---

### 1.20 Memory Pool Allocator - Free No-Op (1)

**File:** `src/memory/pool_allocator.h`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 46-47 | No-op | `free()` con `(void)p; (void)n` - commento "Real freeing happens when pool is reset" |

**Note:** Pool allocator non libera memoria individualmente. Libera solo su reset/destroy del pool.

---

### 1.21 GGUF Loader - Unused Parameters (3)

**File:** `src/io/gguf_loader.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 332, 367, 405 | Unused | `(void)key` - key parameter non usato in skip_gguf_value |

**Note:** Key parameter passato ma non utilizzato nelle funzioni di skip. Indica implementazione incompleta.

---

### 1.22 AutoML Auto Tune - Reference Function Not Used (1)

**File:** `src/automl/auto_tune.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 49 | Unused | `(void)reference_fn` - reference function passata ma non usata |

**Note:** Funzione di benchmarking usa baseline sintetico invece di reference_fn.

---

### 1.23 Benchmark Latency - Simulated Metrics (1)

**File:** `src/bench/latency.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 118 | Simulated | Commento "Realistic simulated metrics when model not available" |

**Note:** Quando modello non disponibile, usa metriche simulate (TTFT ~100-200ms, decode ~50-100 tok/s). Non sono misurazioni reali.

---

### 1.24 AutoML Auto Tune - Simulated Benchmark (1)

**File:** `src/automl/auto_tune.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 51 | Simulated | Commento "Simulated benchmark: each config has a speed factor" |

**Note:** Benchmark di auto-tuning usa simulazione basata su tile size e shared memory, non esecuzione reale dei kernel.

---

### 1.25 Hub Metadata Manager - SQLite Disabled (1)

**File:** `src/hub/metadata_manager.h`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 2 | Disabled | `#include <sqlite3.h>` commentato - "Requires SQLite - disabled for now" |

**Note:** SQLite non integrato come dipendenza. Metadata Manager probabilmente usa implementazione in-memory o file-based senza database reale.

---

### 1.26 Transformer FFN - Fused Bias GELU Comment (1)

**File:** `src/transformer/ffn.cpp`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 87-88 | Comment | "Let's assume we map it in ops::fused_bias_gelu later. Actually, I should just implement ops::fused_bias_gelu first." |

**Note:** Commento di sviluppo che indica implementazione incompleta o pianificata per fused_bias_gelu.

---

### 1.27 Tokenizer BPE - Save/Load Optional (1)

**File:** `src/tokenizer/bpe.h`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 34 | Optional | Commento "Logic to save/load (optional but recommended for later)" |

**Note:** Funzioni save/load dichiarate ma implementazione è opzionale/non richiesta.

---

## 2. Test Placeholder

### 2.1 Test Dummy

**File:** `tests/test_placeholder.cpp`
**File:** `tests/CMakeLists.txt` (righe 232-233, 322)

```cpp
TEST(Placeholder, TrivialPass)
{
    EXPECT_TRUE(true);
}
```

**Note:** Test di placeholder per verificare il setup di Google Test. Può essere rimosso o sostituito con test reali.

---

## 3. UI Placeholder (HTML)

**File:** `ui/index.html`

| Linea | Tipo | Descrizione |
|-------|------|-------------|
| 26 | UI placeholder | `placeholder="Type your message..."` nel textarea input |
| 46 | UI placeholder | `placeholder="You are a helpful assistant."` nel system prompt |

**Note:** Questi sono placeholder UI standard per input forms, non indicano codice incompleto.

---

## 4. Placeholder nella Documentazione

### 4.1 Commenti nei Roadmap

I seguenti file contengono riferimenti a "todo" come parte della documentazione di progetto, non come codice incompleto:

- `todo/fase18+.md` - Riferimento a fase 24 per grounding/fact-checking
- `todo/fase12+.md` - nan_debugger da implementare
- `todo/fase6+.md` - Note su implementazione Metal/Vulkan
- `todo/fase1+.md` - Note su matvec e transpose
- `todo/fase0+.md` - Note su .gitignore
- `todo/fase0_fondamenta.md` - Struttura directory todo/
- `todo/fase17_moe.md` - Tabella feature da implementare
- `todo/fase64_riservata.md` - File placeholder per future funzionalità
- `todo/fase48_api_openai.md` - Tabella endpoint
- `todo/fase40_multi_agent.md` - Sezione consensus algorithms
- `todo/fase8_distribuito.md` - Raccomandazione completamento fasi 1-7
- `todo/fase1_motore_matematico.md` - Sezione obiettivi

**Note:** Questi sono parte della documentazione di roadmap, non indicano codice incompleto.

---

### 4.2 Commenti in README e Regole

- `README.md` (righe 21, 31, 96, 128) - Riferimenti a `todo/` per roadmap
- `.cursor/rules/theseriusproject.mdc` (righe 53, 64) - Riferimenti a roadmap in todo/
- `src/tensor/tensor.h` (riga 29) - Commento su default-constructed tensor come placeholder valido per containers

**Note:** Commenti esplicativi, non codice incompleto.

---

### 4.3 Report Esistenti

- `MISSING_PARTS_REPORT.md` - Documenta 6 TODO rimanenti
- `analysis_report.md` - Documenta 6 TODO commenti e gap implementazione backend

**Note:** Report preesistenti sullo stato del progetto.

---

## 5. Altri Match Non Pertinenti

- `.gitignore` (righe 76-77) - Commento su todo/ directory
- `fase10_finetuning.md` (riga 19) - Nota su trainer SFT da implementare

---

## Riepilogo Azioni Richieste

### Priorità Critica (Codice Incompleto)

1. **Vulkan Backend** - 4 TODO + availability stub
   - Alloc/free memory
   - Buffer mapping/copying
   - is_available() ritorna false
   - Richiede Vulkan SDK

2. **Metal Backend** - 2 TODO + availability stub
   - MTLBuffer allocation/release
   - is_available() ritorna false
   - Richiede macOS/Metal toolchain

3. **Hub Search Engine** - 13 stub methods
   - Tutti i metodi di ricerca ritornano `{}`
   - Richiede implementazione completa indexing e search

4. **Hub Embedding Manager** - 3 stub methods
   - compute_embeddings(), load_model(), compute_with_*()
   - Richiede integrazione modello embedding

5. **Hub Metadata Manager** - SQLite disabled
   - SQLite non integrato come dipendenza
   - Probabilmente usa implementazione in-memory

### Priorità Alta (Codice Funzionale)

6. **MCP HTTP Transport** - 1 not implemented
   - HTTP transport non implementato
   - Richiede libreria client HTTP (cpp-httplib/curl)

7. **MCP Stdio Transport** - 1 stub
   - Bidirectional pipes non implementato

8. **Flash Attention Backward** - 1 not implemented (CPU)
   - Backward pass su CPU lancia eccezione
   - Richiede implementazione CPU o fallback

9. **Graph Compiler Trace** - 1 not implemented
   - Metodo trace() per dummy forward pass

10. **Stress Runner** - 2 Placeholder
    - VRAM monitoring (CUDA)
    - RAM monitoring (OS APIs)

11. **Pipeline Parallel** - 1 Placeholder
    - Calcolo reale gradiente loss

12. **LLaMA Inference** - 1 Placeholder
    - Output passaggio tra rank

13. **Hub Converter** - 1 Placeholder
    - Conversione Safetensors → GGUF

14. **SSM Block** - 1 Placeholder
    - Input projection dimensioni corrette

15. **CLI Commands** - 2 implementation pending
    - cmd_serve, cmd_doctor non implementati

16. **Model Hub Server** - 1 stub
    - search_models() ritorna `{}`

### Priorità Media (Limitazioni/Non usati)

17. **CPU Backend** - 5 stream no-ops
    - Stream operations degradano a sync
    - Non supporta vera asincronicità

18. **BLAS Dispatch** - 2 stream non usati
    - Stream parameter ignorato in matmul/batched_matmul

19. **Quantization FP8 Kernel** - 1 stub
    - float_to_fp8_e4m3_cuda() non implementa conversione

20. **Memory Pool Allocator** - 1 no-op
    - deallocate() non libera individualmente

21. **GGUF Loader** - 3 unused parameters
    - key parameter non usato in skip_gguf_value

22. **AutoML Auto Tune** - 1 unused + 1 simulated
    - reference_fn non usato
    - Benchmark usa simulazione, non esecuzione reale

23. **Benchmark Latency** - 1 simulated
    - Metriche simulate quando modello non disponibile

### Priorità Bassa (Commenti/Opzionale)

24. **Transformer FFN** - 1 commento sviluppo
    - Indica implementazione pianificata per fused_bias_gelu

25. **Tokenizer BPE** - 1 optional
    - save/load opzionali/non richiesti

26. **Test Placeholder** - 1 dummy test
    - Rimuovere o espandere test_placeholder.cpp

### Non richiedono azione

- UI placeholder in HTML (standard)
- Riferimenti a todo/ in documentazione
- Commenti esplicativi nel codice
- (void) parameter supressions (validi in C++)
- std::future usage in thread pool (pattern standard)

---

## Totale Placeholder Identificati: **27+**

**Categorie:**
- Backend hardware incompleti: 2 (Vulkan, Metal)
- Hub/Search engine stub: 17 metodi
- MCP transport stub: 2
- Flash attention: 1
- Graph compiler: 1
- Stress testing: 2
- Pipeline distributed: 1
- Inference: 1
- Converter: 1
- SSM: 1
- CLI: 2
- Server hub: 1
- CPU backend limitazioni: 5
- BLAS: 2
- Quantization: 1
- Memory: 1
- GGUF: 3
- AutoML: 2
- Benchmark: 1
- Commenti/opzionale: 2
- Test: 1
