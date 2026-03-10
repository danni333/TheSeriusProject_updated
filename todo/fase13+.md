# Fase 13+ — Cose mancanti (Deployment)

Riferimento: `fase13_deployment.md`

---

## 1. Server HTTP / API

| Elemento | Stato | Nota |
|----------|--------|-----|
| Server C++ (Crow / cpp-httplib / uWebSockets) | Parziale | httplib in FetchContent; nessun server avviabile |
| POST /v1/chat/completions (OpenAI-compat) | ❌ | |
| WebSocket /v1/chat/stream (streaming) | ❌ | |
| GET /v1/models | ❌ | |
| Request router, thread pool I/O | ❌ | |
| Cartella src/server/ | ❌ | |

---

## 2. Inference engine

| Elemento | Stato |
|----------|--------|
| Thread dedicato inference (modello + KV cache + sampler) | ❌ |
| Continuous batching (fase13) | ❌ |
| PagedAttention (paged KV cache) | Parziale | paged_kv_cache presente; integrazione server da fare |
| Sampler (temperature, top_p, top_k) | ❌ | infer/sampler non presente |

---

## 3. File previsti

| File | Stato |
|------|--------|
| src/server/server.h, .cpp, api_handler | ❌ |
| src/infer/sampler.h, .cpp | ❌ |
| CLI per `serius serve --model ...` | ❌ (fase47 CLI) |

---

**Priorità:** endpoint REST base con cpp-httplib; inferenza singola richiesta; streaming e continuous batching dopo.
