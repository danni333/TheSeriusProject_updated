# Gap competitivi e stato del progetto

**Aggiornamento:** 2026-10-18 (audit completo da sorgenti `src/`, `tests/`, `todo/task.md`, workflow CI e container).

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

### 4.5 CI/CD & Containerizzazione

- **CI/CD:** Presente workflow in `.github/workflows/ci.yml` configurato per pipeline di test su CPU (Ubuntu + Ninja + ctest) e GPU.
- **Container:** Presenti `Dockerfile` multi-stage (CUDA devel per build, runtime per deployment), `docker-compose.yml` e manifest Kubernetes in `k8s/` (`deployment.yaml`, `pvc.yaml`, `service.yaml`).

### 4.6 Allineamento `todo/task.md` (ottobre 2026)

Stato aggiornato del master tracker:

- **Completate ([x]) e verificate:** Fasi 1–5 (Tensori, Autograd, Transformer core, Tokenizer, Training), Fase 6 (HAL CPU/CUDA), Fase 8 (Distribuito), Fase 9 (Stabilità), Fase 10 (SFT Fine-Tuning), Fase 11 (Flash Attention CUDA), Fase 12 (Tensor Logger), Fase 14 (Speculative Decoding lib), Fase 15 (Safetensors/sharding), Fase 16 (Paged KV, Sliding Window, Prefix Cache), Fase 17 (MoE DeepSeek), Fase 18 (Sicurezza/Guardrails), Fase 19 (LoRA), Fase 20 (Multimodale), Fase 21 (Interpretabilità), Fase 22 (RAG & Tools), Fase 23 (RLHF), Fase 24 (Grounding), Fase 25 (Benchmarking), Fase 26 (Containerizzazione), Fase 27 (CI/CD), Fase 28 (Monitoring/Batcher), Fase 29 (Data Curation), Fase 30 (Kernel Fusion), Fase 31 (Ethical Bias), Fase 32 (MCP Client), Fase 33 (ReAct core), Fase 35 (Test-Time Compute core), Fase 36 (Sparse Attention kernels), Fase 37 (Federated Learning), Fase 38 (AutoML / AutoTune & Pruner), Fase 40 (Multi-Agent), Fase 47 (CLI chat interattiva con KV cache), Fase 53 (Plugin System).
- **Parziali o Stub ([ ]) da completare:** Fase 13 & 48 (Server HTTP e API OpenAI: endpoint stub senza inferenza forward reale collegata), Fase 34 (Sicurezza MCP: fix inizializzazione statica tool_risk_), Fase 39 (Long-Term Memory: DNC presente, estensioni aperte), Fase 50 (Model Hub: server/metadata base, auth JWT e search completa aperte).
- **Aperte ([ ]):** Frontiere (41–46), Chat UI (49), Documentazione release (51), Packaging installer (52), Benchmarks su larga scala (54–59), Piattaforma e swarm (60–72).

---

## 5. Azioni consigliate (priorità operative)

1. **Collegare `/v1/chat/completions` all'inferenza reale del modello** (agganciare `llama_forward_inference` o `ContinuousBatcher` con caricamento safetensors e streaming SSE token-by-token anziché lo stub corrente).
2. **Correggere l'inizializzazione statica in `src/mcp/guard.cpp`** per garantire che `MCPGuard::risk_level()` funzioni in isolamento senza richiedere preventiva istanziazione (sbloccando `Guard.RiskLevelDefaults`).
3. **Mappare GGUF sui modelli** (`LLaMAModel`/`QwenModel`) per abilitare l'esecuzione diretta da file `.gguf` oltre ai `.safetensors`.
4. **Integrare la quantizzazione INT4 nel forward del Transformer** (collegare i kernel `dequant_matmul` al loop di inferenza, chiudendo il gap T1.05).
5. **Creare un adapter tra `agent::Model` / `reasoning::Model` e i modelli reali** (per consentire a ReAct, MCTS e Multi-Agent di eseguire prompt reali invece dei mock).

---

## 6. Note di manutenzione

Quando lo stato cambia in modo significativo (es. chat non più stub, GGUF load, nuove rotte API), aggiornare la sezione **4** e la data in testa. Il confronto competitivo (sezioni **1–3**) va rivisto se cambia il target (es. solo research vs serving produzione).
