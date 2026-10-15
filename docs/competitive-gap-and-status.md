# Gap competitivi e stato del progetto

**Aggiornamento:** 2026-03-30 (audit da sorgenti `src/`, `tests/`, `todo/task.md`, `docs/architecture.md`).

Questo documento unisce il confronto con motori di inferenza / training maturi e una fotografia operativa di cosa è effettivamente cablato nel repo.

---

## 1. Posizionamento

TheSeriusProject è un **motore LLM in C++ “from scratch”** (tensor, autograd, transformer, training, quantizzazione, HAL, server minimale). I competitor di riferimento per il gap funzionale sono soprattutto:

- **llama.cpp** — caricamento GGUF, CPU/GPU, ecosistema quantizzazioni
- **vLLM / SGLang** — continuous batching, PagedAttention a scala serving, throughput
- **TensorRT-LLM** — kernel NVIDIA, FP8, grafi ottimizzati per produzione
- **Ollama** (come riferimento UX/packaging) — esperienza “scarico e uso”

Obiettivo dichiarato del repo resta **didattico / estensibile**; la roadmap `todo/` è molto più ampia del nucleo già integrato in produzione tipica dei competitor.

---

## 2. Gap rispetto ai competitor (sintesi)

| Area | Competitor tipici | Gap relativo per Serius |
|------|-------------------|-------------------------|
| **Serving ad alto throughput** | vLLM, TGI | Continuous batching end-to-end, scheduling richieste, SLO/latency come prodotto |
| **KV cache “data center”** | PagedAttention + gestione blocchi a runtime | Esiste `PagedKVCache` / `BlockManager` nel transformer; integrazione completa con stack HTTP e molte richieste concorrenti non è paragonabile a vLLM |
| **Formati pesi** | llama.cpp (GGUF completo) | GGUF: lettura metadati e nomi tensor; **non** caricamento tensor end-to-end come llama.cpp. Safetensors: percorso LLaMA HF (`hf_llama_loader`) più maturo |
| **API compatibile OpenAI** | LangChain, client generici | Solo parte delle rotte; chat è ancora **stub** (nessun modello agganciato nel handler). Manca parità con `todo/fase48_api_openai.md` (completions, embeddings, `GET /v1/models/{id}`, SSE streaming, metriche Prometheus) |
| **Quantizzazione “production”** | TensorRT-LLM, GGML quants | INT4/dequant; `fp8.cpp` in tree; tracker segnala refinement FP8/BF16; copertura formati/industry minor di stack dedicati |
| **Distribuito** | DeepSpeed, FSDP, NCCL-first | Moduli `distributed/*` (local, data parallel, pipeline, tensor parallel); **NCCL/MPI** opzionali via CMake. Fase 8 in roadmap ancora ambiziosa rispetto allo stato corrente |
| **MLOps / osservabilità** | Prometheus, tracing, CI | Metriche server in JSON semplice, non formato Prometheus. **Nessun workflow GitHub Actions** nella root del repo (solo nelle dipendenze in `build_*`) |
| **Feature modello avanzate** | MoE, speculative, lunghi contesti | Molte fasi `todo/` ancora `[ ]`: MoE, speculative decoding, long context strutturato, LoRA, multimodale, RAG/tooling lato engine |

---

## 3. Punti di forza (differenziazione)

- **Pipeline software completa per un motore didattico:** autograd, ops, training (AdamW, AMP, checkpoint), flash attention (CUDA), tokenizer BPE, quant (INT4, AWQ, GPTQ in tree), modelli LLaMA/Qwen, inference CPU-oriented (`llama_inference`).
- **Paged KV** presente nel codice del transformer (test `test_paged_kv`).
- **Loader safetensors** per pesi stile Hugging Face LLaMA.
- **HAL** multi-backend (CPU, CUDA condizionale, Metal/Vulkan condizionali).

---

## 4. Audit stato repository (verificato nel codice)

### 4.1 Server HTTP (`src/server/server.cpp`)

| Rotta | Stato |
|-------|--------|
| `GET /health` | Attivo — JSON `{"status":"ok"}` |
| `GET /metrics` | Attivo — **JSON** (`uptime_seconds`, `request_count`), non formato Prometheus testo (diverso da `todo/fase48_api_openai.md`) |
| `POST /v1/chat/completions` | Attivo — **risposta fissa stub** (“Load a model…”), nessun collegamento al forward del modello |
| `GET /v1/models` | Attivo — lista stub con `default` |
| `POST /v1/completions`, `POST /v1/embeddings`, `GET /v1/models/{id}` | **Non presenti** (solo in roadmap Fase 48) |
| Streaming SSE | **Non implementato** nel handler corrente |
| Model hub | Route hub **disabilitate** (`#if 0` / commento “disabled to fix build”) |

### 4.2 I/O modelli (`src/io/`)

| Componente | Stato |
|------------|--------|
| `config_loader` | JSON → ModelConfig (documentato in architettura) |
| `gguf_loader` | Metadati header + `gguf_tensor_names`; **nessun load tensor weights nel modello** |
| `safetensors_loader` + `hf_llama_loader` | Caricamento pesi LLaMA da `.safetensors` (anche sharded) |
| `simple_weights_loader` | Presente (test dedicato) |

Allineamento con `docs/architecture.md`: la frase su “GGUF/safetensors in fase di implementazione” per il flusso completo resta in parte valida: **safetensors LLaMA sì; GGUF pesi no**.

### 4.3 Core build (`src/CMakeLists.txt`)

- Libreria `serius_core` include: memory, tensor, blas (CPU SIMD + dispatch), autograd, transformer (incl. `paged_kv_cache`), flash attn, tokenizer, data, optim, train, quant, io, infer, safety, debug, **distributed** (local, data_parallel, pipeline, tensor_parallel).
- **CUDA:** backend, fused kernels, matmul CUDA, quant CUDA, flash attn CUDA — se `SERIUS_USE_CUDA`.
- **NCCL:** `nccl_comm.cpp` solo se `SERIUS_USE_NCCL`.
- **MPI:** `mpi_comm.cpp` solo se `SERIUS_USE_MPI`.
- **Metal / Vulkan:** backend condizionali.

### 4.4 Test e benchmark (`tests/CMakeLists.txt`)

**Target di test registrati con `gtest_discover_tests`:** tensor, autograd, hal, (CUDA: fused, llama_inference_cuda), paged_kv, logger, quant, flash_attn, config_loader, simple_weights_loader, llama_model, sampler, guardrails, nan_debugger, transformer, transformer_optim, amp, blas, dataloader, tokenizer, training, kv_cache_logic, memory, distributed, thread_pool.

**Target `add_executable` ma non passati a `gtest_discover_tests`:**

- `test_cuda` (solo se CUDA)
- `test_distributed_llama`
- `test_pipeline_llama`

→ Con `ctest` dalla build standard, questi tre **possono non essere eseguiti** automaticamente (restano eseguibili manuali).

Benchmark: `bench_full`, `bench_matmul`, `bench_memory`, `bench_mixed_precision`, `bench_quant` (non necessariamente legati a `ctest`).

### 4.5 CI/CD

- Ricerca `.github/workflows` nella **root del repository:** nessun workflow di progetto trovato (i file sotto `.github` compaiono nelle dipendenze FetchContent in `build_cuda/_deps/`, non come CI del repo Serius).

### 4.6 Allineamento `todo/task.md` (marzo 2026)

Segnaposto utili dal master tracker:

- Completate (**[x]**): fasi 1–6, 11–12, parti di 4–5, ecc.
- Parziali (**[/]**): Fase 7 quant, Fase 8 distribuito, Fase 10 finetuning, Fase 15 checkpoint/GGUF metadata, Fase 30 kernel fusion
- Aperte (**[ ]**): stabilità, deployment, speculative decoding, long context, MoE, sicurezza, LoRA, benchmarking, container, CI/CD, monitoring, e molte fasi “frontiera”

---

## 5. Azioni consigliate (priorità indicative)

1. **Collegare `/v1/chat/completions` al pipeline reale** (tokenizer → forward → sampler) e introdurre **SSE** per streaming.
2. **Completare o documentare esplicitamente** il confine GGUF (solo meta vs load pesi) e, se obiettivo è parità llama.cpp, pianificare load tensor.
3. **Registrare `test_cuda`, `test_distributed_llama`, `test_pipeline_llama` in `gtest_discover_tests`** (o documentare che sono opt-in).
4. **CI minima** (build + `ctest --output-on-failure`) su almeno una piattaforma.
5. **Metriche**: allineare `/metrics` a Prometheus o rinominare la rotta per evitare aspettative errate.

---

## 6. Note di manutenzione

Quando lo stato cambia in modo significativo (es. chat non più stub, GGUF load, nuove rotte API), aggiornare la sezione **4** e la data in testa. Il confronto competitivo (sezioni **1–3**) va rivisto se cambia il target (es. solo research vs serving produzione).
