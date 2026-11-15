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

### A4. Config centralizzata
- `defaults.h` esiste già; estenderlo per porte/timeout/threshold sparsi
  (vedi report Part 3). Bassa priorità.

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

### D1. Flash Attention backward CPU (`src/attention/`)
- Richiede per il training su sequenze lunghe. Media complessità.

### D2. Pipeline-parallel gradient reale
- Oggi `run_transformer_layers` ha placeholder per intermediate ranks.
- Serve per training distribuito PP; non per inferenza.

### D3. Graph compiler trace reale
- strumentazione per auto-tuning kernel.

### D4. CPU backend stream asincrono / BLAS stream params
- Micro-ottimizzazioni; guadagno modesto.

### D5. Quantization FP8 kernel
- Serve solo se si vuole FP8 inference; CUDA ha già FP16/BF16 path.

---

## Priorità consigliata

1. **A1 + A2** — chiudono gli stub più visibili (MCP + Hub search), sforzo piccolo.
2. **A3** — igiene, veloce.
3. **B1** — GGUF writer, valore ecosistema alto.
4. **B2 + B3** — completano il percorso MCP/Hub.
5. Track C: **non fare** a meno che non serva Vulkan/Metal davvero.
6. Track D: solo se si punta al training distribuito.

## Criteri di "fatto"

- Ogni stub rimosso ha test dedicato in `tests/`.
- Nessun `return {}` / `not implemented` silenzioso senza env opt-out.
- Report consolidato aggiornato a ogni track completata.
