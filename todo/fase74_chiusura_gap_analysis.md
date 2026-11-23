# Fase 74 — Chiusura Gap Analysis (post report_placeholder)

Piano di chiusura dei debiti tecnici emersi da `report_placeholder_consolidated.md`
(2026-10-29, aggiornato 2026-11-15) e dal lavoro su web UI / inferenza.

## Stato attuale (2026-11-15, secondo aggiornamento)

Già risolti: batched prefill (Tensor host/device sync), KV cache persistente,
multi-modello + load/unload runtime + `hf://`, BOS/EOS reali, stress VRAM/RAM
server-side, TPS reale, chaos assertions, test_placeholder rimosso,
**A1 MCP transports** (HTTP + stdio, 11/11 test), **A2 Hub Search Engine**
(file-backed, 4/4 test), **A3 catch silenti** io/hub (tutti loggati),
MetadataManager CRUD completo (JSON file-based, SQLite resta disabilitata),
**B1 GGUF writer** (`io/gguf_writer.cpp`: HF dir/.safetensors → GGUF con
config KV + tokenizer embedding + RoPE perm per arch llama; `serius convert`
bidirezionale; verificato end-to-end su Qwen2.5-Coder-1.5B reale),
**B2 Hub server routes + auth** (REST reale: upload/download/search/
suggest/delete + API key file-based, `test_hub_server` 2/2),
**B3 MCP server end-to-end** (`mcp/server.{h,cpp}` + `serius mcp-serve`,
E2E stdio+HTTP, `test_mcp_server` 11/11).

---

## Track A — Quick win (1–3 giorni l'uno)

### A1. MCP transports (`src/mcp/`) ✅ FATTO
- `transport_http.cpp`: POST JSON-RPC via `cpp-httplib` — implementato.
- `transport_stdio.cpp`: subprocess `CreateProcess`/fork con pipe bidirezionali.
- Test: roundtrip HTTP + stdio in `test_mcp_client` (11/11).

### A2. Hub Search Engine (`src/hub/search_engine.cpp`) ✅ FATTO
- `search_models`: keyword + filtri tag/author su `MetadataManager` JSON.
- `semantic_search`: ranking deterministico (embedding hash-like, 256-dim).
- `get_suggestions`, trending, analytics, cache, highlight, paginazione.
- Test: `test_hub_search` (4/4).

### A3. Catch silenti residui ✅ FATTO
- `config_loader`, `safetensors_loader`, `sharding`, `downloader`,
  `metadata_manager` ora loggano `[Component] Failed ...` con `e.what()`.
- `nccl_comm` rilancia (non silenzioso), `gguf_export` skip legittimo.

### A4. Config centralizzata ✅ FATTO
- `defaults.h` esteso: `kHubPort` (8081), `kHost`, timeout MCP/stress,
  default generazione (`kMaxNewTokens`, `kTemperature`, `kTopK`, `kTopP`).
- Usati in `ServerConfig`, `Sampler`, `ModelHubServer` (default ora
  `kHubPort`, prima `kHttpPort` errato), `hub_main` (fixato `--port`
  che veniva parsato ma ignorato), `transport_http`, `stress_runner`,
  `worker`, `cmd_chat`/`cmd_serve`/`cmd_bench`.

## Track B — Media (1–2 settimane)

### B1. Hub Converter Safetensors→GGUF ✅ FATTO
- `src/io/gguf_writer.cpp`: GGUF v3 writer (header, KV metadata, tensor infos,
  allineamento 32B). FP16 default, FP32 opt-in.
- Mappatura nomi HF→GGUF inversa a `gguf_export`, RoPE perm per arch "llama",
  embedding tokenizer.ggml.* da tokenizer.json/merges.txt.
- `ModelConverter::safetensors_to_gguf` delega al writer; `serius convert`
  ora bidirezionale (input .gguf → export, dir/.safetensors → import).
- Test: `test_gguf_writer` (4/4) + verifica live su Qwen2.5-Coder-1.5B
  ("2+2?" → "4", stop corretto).

### B2. Model Hub Server routes + auth (`src/hub/model_hub_server.cpp`) ✅ FATTO
- Route REST reali via httplib: `GET /health`, `GET /metrics`,
  `GET /v1/models`, `GET /v1/models/{id}`, `GET /v1/models/search`,
  `GET /v1/search/suggest`, `POST /v1/models/upload` (multipart
  `model`+`metadata`), `GET /v1/models/{id}/download` (range provider),
  `DELETE /v1/models/{id}` (solo owner o permesso `delete`).
- Auth file-based (`hub_users.json`): `POST /v1/auth/register` emette
  `sk-*` API key, `GET /v1/auth/me`; `Authorization: Bearer` verificato
  sulle route di scrittura. Modalità aperta (default_user) finché non
  esiste alcun utente registrato — JWT/SQLite restano disabilitati.
- Path risolti a assoluti nel ctor (robustezza a cwd che cambia).
- Test: `test_hub_server` (2/2) — register→401→upload→list→get→search→
  download→delete end-to-end.

### B3. MCP client/server end-to-end ✅ FATTO
- `src/mcp/server.{h,cpp}`: `MCPServer` con dispatch JSON-RPC 2.0
  (`initialize`, `ping`, `tools/list`, `tools/call`), errori spec
  (-32600/-32601/-32602/-32603), notifiche senza risposta.
- Trasporti server: `serve_stdio()` (newline-delimited) e
  `register_http_route()` per `POST /mcp`.
- CLI `serius mcp-serve [--audit PATH] [--http PORT]` con tool built-in
  (`echo`, `current_time`, `serius_info`) e audit log via `Audit`.
- Test `test_mcp_server` (11/11): unit dispatch + E2E HTTP
  (HttpTransport↔MCPServer) + E2E stdio reale (`serius.exe mcp-serve`
  spawnato via StdioTransport, audit verificato).

## Track C — Backend hardware (DEPRIORITIZZATO)

### C1. Vulkan backend (`src/hal/vulkan_backend.cpp`, 92 righe, tutto stub)
- Richiede Vulkan SDK, SPIR-V shaders, VMA. Ordine di grandezza: settimane.
- **Raccomandazione**: non partire ora. CPU (AVX2/512) + CUDA coprono i target
  attuali. Rivalutare solo se serve GPU vendor-diversa da NVIDIA.

### C2. Metal backend
- Solo macOS; ambiente di sviluppo è Windows. **Skip** salvo porting futuro.

## Track D — Inference avanzata (media)

### D1. Flash Attention backward CPU (`src/attention/`) ✅ FATTO
- `flash_attn.cpp`: forward CPU (online softmax, O(seq_k) scratch/riga) e
  backward CPU (ricombina P=exp(S-LSE) riga per riga, niente matrice S²;
  gradienti parziali supportate). Prima: throw "not implemented".
- Test: `CpuForwardMatchesReference` + `CpuBackwardMatchesReference` in
  `test_flash_attn` (9/9), parity con il riferimento naive.

### D2. Pipeline-parallel gradient reale ✅ FATTO (parziale)
- `PipelineExecutor::execute` accetta `loss_gradients` opzionale
  ([n_mb * hidden_size]): l'ultimo stage usa dL/d(out) reale invece del
  placeholder 1.0. Test `LossGradientPropagatesToFirstStage` verifica la
  propagazione mb-reverse attraverso 2 stage.
- `PipelineExecutor::execute_1f1b()` implementa lo scheduling 1F1B
  (Megatron-LM): warmup = `num_stages-1-stage` forward, poi alternanza
  1F/1B; `execute()` resta la variante GPipe. Test: `OneF1BPropagatesLossGradient`,
  `OneF1BThreeStages`.
- RestaOOT: integrazione con micro-batch su dati reali.

### D3. Graph compiler trace reale ✅ FATTO (parziale)
- `GraphCompiler::trace_graph(Node*)`: walk topologico del grafo autograd
  reale (usa `core::autograd::topological_sort`), un GraphOp per nodo
  calcolato (type = node->name, input/output shape reali).
- `ExecutionPlan::execute` ora ri-esegue davvero il grafo: `GraphOp::executor`
  copiado da `Node::forward_fn` (replay zero-copy via `detail::replay_forward`,
  view host per operand float32/contiguous); `in_replay()` evita le catture
  Tensor di backward/forward durante il replay. `trace()` dummy resta per
  compatibilità test (i suoi piani non sono eseguibili).
- Optimization passes implementati: `fuse_layernorm_linear` (fusione
  norm→matmul con executor composto dai due executor originali),
  `eliminate_redundant_copies` (match sul nome reale `contiguous(...)` +
  rewiring degli slot consumatori), `reorder_for_locality` (list-scheduling
  greedy topologico), `preallocate_buffers` (liveness → `workspace_size()` =
  picco tensor vivi invece della somma conservativa). Fix leak workspace:
  `~ExecutionPlan` libera via `hal::IBackend::free`, `compile()` alloca col
  CPU backend alla dimensione peak (non più `::operator new` di `total_size`).
  `fuse_elementwise_chains` fonde tutte le catene; `is_executable()` +
  errore esplicito per piani da `trace()` sintetico.
- Test: `TraceGraphFromRealAutogradGraph`, `TraceGraphNullIsEmpty`,
  `ExecuteReplaysTracedGraph`, `ExecuteBranchingGraph`, `IsExecutable`,
  `FuseLayernormLinear`, `PreallocateBuffersUsesLivenessPeak`,
  `EliminateCopiesPreservesExecution`, `ReorderForLocalityPreservesExecution`.

### D4. CPU backend stream asincrono / BLAS stream params — ✅ FATTO
- `CpuBackend::create_stream()` ora restituisce un `CpuStream` reale: worker
  thread + code queue (`std::function`), `sync_stream` = barrier di coda,
  `destroy_stream` ferma il worker e fa join.
- `memcpy_h2d_async`/`d2h_async`/`d2d_async` accodano la copia sullo stream;
  senza stream restano sincrone (contratto invariato).
- `blas::matmul` e `batched_matmul` instradano il param `stream` sullo stream
  CPU quando `hal::is_cpu_stream(stream)` è vero (registry dei CpuStream
  vivi → un `cudaStream_t` non viene mai interpretato come CpuStream).
  Il chiamante deve `sync_stream` prima di leggere l'output.
- Test: `test_hal` verifica ordinamento, sync, distruzione stream, GEMM su
  stream e rejection di puntatori estranei.

### D5. Quantization FP8 kernel — ✅ FATTO (CPU)
- `fp8_dequant_matmul()`: kernel CPU fused dequant+matmul per pesi FP8
  E4M3/E5M2 con scale per blocco (layout di `fp8_quantize_tensor`).
- Test: `QuantFP8.DequantMatmulMatchesReference` (parità vs riferimento
  dequant+dot) ed `E5M2`.
- Resta: path CUDA FP8 matmul (le conversioni CUDA esistono già in
  `quant_kernels`).

---

## Priorità consigliata

1. **A1 + A2** — chiudono gli stub più visibili (MCP + Hub search), sforzo piccolo.
2. **A3** — igiene, veloce.
3. **B1** — GGUF writer, valore ecosistema alto.
4. **B2 + B3** — completano il percorso MCP/Hub.
5. Track C: **non fare** a meno che non serva Vulkan/Metal davvero.
6. Track D: solo se si punta al training distribuito.

## Track E — Follow-up: Nomic embedding performance (IN CORSO)

Encoder `nomic-bert` funzionante (commit `a62e1c8`). Stato attuale:

- **GEMM via `blas::matmul`** (2026-11-16): pesi trasposti a load-time
  (`[in,out]`), `linear_rows` scalare sostituito da `linear_gemm` con
  `syncToHost()` post-matmul (richiesto: `blas::matmul` può dispatchare
  su CUDA async e il risultato non tornerebbe nel buffer host).
  RoPE cos/sin hoisted fuori dal loop layer.
- **Risultato**: ~3.8s → ~0.84s per embedding (**~4.5x**), cosine
  semanticamente identiche (max_diff GEMM-vs-scalare ~3e-6, verificato).
  CPU e CUDA disabilitato danno lo stesso tempo → path CPU AVX2.
- **Rimasto per "record"**: batching multi-testo in un solo GEMM,
  pesi GPU persistenti (evitare H2D per chiamata), eventualmente
  kernel attention su GPU.
- **Test su corpus reale** (2026-11-16): `HubSearch.EmbeddingManagerNomicRepoCorpus`
  embedd-a 9 doc reali della repo (chunk ~480 char, ~12 chunk/doc, score
  0.7*cos_top2 + 0.3*jaccard come `semantic_search`). Risultato misurato:
  query topiche su doc in prosa ranking corretto (gpu-setup #1, huggingface
  #2), ma doc reference pesanti di codice/tabelle (api-reference, cli-reference,
  supported-models) non entrano in top-3 — i chunk JSON/command/simboli
  degradano la rappresentazione WordPiece+mean-pool e doc keyword-dense
  vincono per overlap lessicale. **Bug trovato e fixato**: `Tensor::free_data`
  non liberava `data_gpu_` per tensori non-owner → leak VRAM (CUDA OOM dopo
  ~3k GEMM); ora `data_gpu_` è sempre tensor-owned e liberato sempre.
  Follow-up qualità: retrieval ibrido BM25-first, chunking semantic-aware,
  o modello di embedding più adatto a codice.

## Criteri di "fatto"

- Ogni stub rimosso ha test dedicato in `tests/`.
- Nessun `return {}` / `not implemented` silenzioso senza env opt-out.
- Report consolidato aggiornato a ogni track completata.
