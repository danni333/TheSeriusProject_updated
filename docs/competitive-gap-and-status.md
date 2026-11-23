# Gap competitivi e stato del progetto

**Aggiornamento:** 2026-11-22 (sezioni 2, 4.1, 4.2, 5 riviste su codice corrente; vedi anche `AUDIT_STUBS_AND_GAPS_2026-11-22.md` per l'audit stub/placeholder completo).

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
| **Formati pesi** | llama.cpp (GGUF completo) | GGUF: metadati, tokenizer, **caricamento tensor end-to-end** con dequant packed (Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/Q5_0). dtype non supportati (Q2_K/Q3_K/Q8_K/IQ/TQ) falliscono esplicitamente per tensori critici. Safetensors: percorso LLaMA HF (`hf_llama_loader`) maturo |
| **API compatibile OpenAI** | LangChain, client generici | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` (+ load/unload) attivi con inferenza reale e SSE. Manca ancora parità completa con `todo/fase48_api_openai.md` (GET /v1/models/{id}, metriche Prometheus) |
| **Quantizzazione “production”** | TensorRT-LLM, GGML quants | INT4/dequant + kernel packed GGUF (Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K/Q5_0); `fp8.cpp` in tree; manca coverage Q2_K/Q3_K/IQ/TQ e formati industry |
| **Distribuito** | DeepSpeed, FSDP, NCCL-first | Moduli `distributed/*` (local, data parallel, pipeline, tensor parallel); **NCCL/MPI** opzionali via CMake. Fase 8 in roadmap ancora ambiziosa rispetto allo stato corrente |
| **MLOps / osservabilità** | Prometheus, tracing, CI | Metriche server in JSON semplice, non formato Prometheus. **Nessun workflow GitHub Actions** nella root del repo (solo nelle dipendenze in `build_*`) |
| **Feature modello avanzate** | MoE, speculative, lunghi contesti | Molte fasi `todo/` segnate `[x]` ma con aree a scala didattica (vedi audit stub); MoE, speculative decoding, long context (paged/sliding/prefix cache), LoRA, RAG/tooling agentico (ReAct+ToolRegistry+LlamaAgentModel, 2026-11) collegati al modello reale |

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
| `GET /metrics` | Attivo — **JSON** (`uptime_seconds`, `request_count`, …), non formato Prometheus testo |
| `POST /v1/chat/completions` | Attivo — inferenza reale (prefill KV + decode + sampler), SSE streaming, KV cache persistente multi-turn, multi-modello a runtime |
| `POST /v1/completions` | Attivo — inferenza reale |
| `POST /v1/embeddings` | Attivo |
| `GET /v1/models`, `POST /v1/models/load`, `DELETE /v1/models/{id}` | Attivi — registry multi-modello, load `hf://` e GGUF |
| `GET /v1/models/{id}` | Non presente |
| Streaming SSE | Implementato sui completions/chat |

### 4.2 I/O modelli (`src/io/`)

| Componente | Stato |
|------------|--------|
| `config_loader` | JSON → ModelConfig (documentato in architettura) |
| `gguf_loader` | Metadati, tokenizer, **caricamento pesi tensor end-to-end**: dtype supportati F32/F16/BF16/Q4_0/Q4_1/Q8_0/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/Q8_K/IQ4_NL/IQ4_XS/TQ1_0/TQ2_0 con dequant verificato contro `gguf` (Python) — Q3_K, TQ2_0 bit-exact; generazione coerente verificata su Qwen2.5-0.5B per Q3_K_M, Q2_K, IQ4_NL, IQ4_XS (TQ2_0 degenere = limite intrinseco del quant, confermato vs llama.cpp). Le K-quant non-native (Q2_K/Q3_K/Q8_K/IQ/TQ) vengono dequantizzate in FP32 e requantizzate a Q4_1 per il fast path `matvec_q`. Dtype non supportati (IQ1/IQ2/IQ3 families, MXFP4, NVFP4, …) falliscono esplicitamente per tensori critici, skippati con warning per non-critici |
| `safetensors_loader` + `hf_llama_loader` | Caricamento pesi LLaMA da `.safetensors` (anche sharded) |
| `simple_weights_loader` | Presente (test dedicato) |

Allineamento con `docs/architecture.md`: la frase su “GGUF/safetensors in fase di implementazione” per il flusso completo non è più valida: **entrambi i formati caricano pesi end-to-end** (GGUF con 17 dtype, safetensors LLaMA).

### 4.3 Core build (`src/CMakeLists.txt`)

- Libreria `serius_core` include: memory, tensor, blas (CPU SIMD + dispatch), autograd, transformer (incl. `paged_kv_cache`), flash attn, tokenizer, data, optim, train, quant, io, infer, safety, debug, **distributed** (local, data_parallel, pipeline, tensor_parallel).
- **CUDA:** backend, fused kernels, matmul CUDA, quant CUDA, flash attn CUDA — se `SERIUS_USE_CUDA`.
- **NCCL:** `nccl_comm.cpp` solo se `SERIUS_USE_NCCL`.
- **MPI:** `mpi_comm.cpp` solo se `SERIUS_USE_MPI`.
- **Metal / Vulkan:** backend condizionali.

### 4.4 Test e benchmark (`tests/CMakeLists.txt`)

**Target di test registrati con `gtest_discover_tests`:** tensor, autograd, hal, (CUDA: fused, llama_inference_cuda), paged_kv, logger, quant, flash_attn, config_loader, simple_weights_loader, llama_model, sampler, guardrails, nan_debugger, transformer, transformer_optim, amp, blas, dataloader, tokenizer, training, kv_cache_logic, memory, distributed, thread_pool.

**Tutti i target `add_executable` sono registrati in `ctest`** via `gtest_discover_tests` ( inclusi `test_cuda`, `test_distributed_llama`, `test_pipeline_llama`).

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

Stato al 2026-11-22 (dettaglio completo in `AUDIT_STUBS_AND_GAPS_2026-11-22.md`):

1. ~~**Collegare `/v1/chat/completions` all'inferenza reale**~~ — **FATTO**: inference reale + SSE + KV cache multi-turn.
2. ~~**Fix `src/mcp/guard.cpp`**~~ — **FATTO**: `tool_risk_` inizializzato a dichiarazione.
3. ~~**Mappare GGUF sui modelli**~~ — **FATTO**: `load_llama_weights_from_gguf_file` con pesi packed quantizzati.
4. ~~**Quantizzazione nel forward**~~ — **FATTO**: fast path packed-quant GEMV nel decode (`llama_inference.cpp`).
5. ~~**Adapter agent ↔ modelli reali**~~ — **FATTO**: `agent/llama_model_adapter` + `ToolRegistry` in `ReActEngine`.
6. **Residui aperti**: Vulkan/Metal backend rimossi (solo CPU+CUDA); `graph_compiler` fallisce onestamente se eseguito; benchmark vs llama.cpp **fatto** (`BENCHMARKS.md` — decode GPU paritario/superiore, prefill e CPU in svantaggio); test E2E sensibili al timeout sotto `ctest -j` alto (fix consigliato: `set_read_timeout_sec(60)` su `httplib::Client` in `test_e2e.cpp`).
7. **Distribuito — stato reale verificato (2026-11)**: `LocalCommunicator` (thread in-process) supporta DP/TP/PP con test di equivalenza numerica che **passano tutti** (TP e PP vs single-rank, diff <1e-4). `NCCLCommunicator` e `MPICommunicator` sono implementati e gated da `SERIUS_USE_NCCL`/`SERIUS_USE_MPI`, ma: (a) NCCL è Linux-only — non buildabile su Windows; (b) `Communicator::create_nccl(world,rank)` richiede MPI per il bootstrap di `ncclUniqueId`; (c) **non esiste launcher multi-processo** — nessun path in `serius` CLI/server costruisce una `DistributedTopology` reale da env/argomenti, quindi TP/PP sono oggi usabili solo come libreria o via test. Gap verso llama.cpp/DeepSpeed: servono rank bootstrap (env `RANK`/`WORLD_SIZE` o file-store), wiring in `serve`/`generate`, e verifica NCCL su Linux.

---

## 6. Note di manutenzione

Quando lo stato cambia in modo significativo (es. chat non più stub, GGUF load, nuove rotte API), aggiornare la sezione **4** e la data in testa. Il confronto competitivo (sezioni **1–3**) va rivisto se cambia il target (es. solo research vs serving produzione).
