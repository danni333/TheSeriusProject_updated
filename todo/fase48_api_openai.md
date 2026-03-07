# Fase 48 — API Server OpenAI-Compatible (REST + SSE Streaming)

> Per essere un drop-in replacement di OpenAI, il server deve esporre
> le stesse rotte e lo stesso formato JSON. Così qualsiasi client
> esistente (LangChain, AutoGen, Continue.dev) funziona senza modifiche.

---

## 1. Endpoint Compatibili

| Endpoint | Metodo | Descrizione |
| --- | --- | --- |
| `/v1/chat/completions` | POST | Chat completion (streaming e non) |
| `/v1/completions` | POST | Text completion classico |
| `/v1/embeddings` | POST | Generazione embedding vettoriali |
| `/v1/models` | GET | Lista modelli disponibili |
| `/v1/models/{id}` | GET | Dettagli singolo modello |
| `/health` | GET | Health check (200 OK) |
| `/metrics` | GET | Metriche Prometheus-format |

---

## 2. Formato Richiesta/Risposta

### Chat Completion (non-streaming)

```json
// POST /v1/chat/completions
{
    "model": "SeriusModel-7B",
    "messages": [
        {"role": "system", "content": "Sei un assistente utile."},
        {"role": "user", "content": "Cos'è la quantizzazione?"}
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
}

// Response
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1709654400,
    "model": "SeriusModel-7B",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "La quantizzazione..."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 128,
        "total_tokens": 153
    }
}
```

### Streaming via SSE (Server-Sent Events)

```text
POST /v1/chat/completions  (con "stream": true)

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"La"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":" quant"},"index":0}]}

data: [DONE]
```

---

## 3. Implementazione Server

```cpp
namespace serius::server {

class APIServer {
    httplib::Server svr_;
    Model* model_;
    Tokenizer* tokenizer_;
    Sampler* sampler_;

public:
    void setup_routes() {
        svr_.Post("/v1/chat/completions",
            [this](const auto& req, auto& res) {
                auto body = json::parse(req.body);
                bool stream = body.value("stream", false);

                if (stream) {
                    handle_stream(req, res, body);
                } else {
                    handle_completion(req, res, body);
                }
            });

        svr_.Get("/v1/models", [this](const auto&, auto& res) {
            res.set_content(get_models_json(), "application/json");
        });

        svr_.Get("/health", [](const auto&, auto& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
    }

    void handle_stream(const auto& req, auto& res, const json& body) {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");

        res.set_chunked_content_provider("text/event-stream",
            [this, body](size_t offset, auto& sink) {
                // Genera token per token, invia via SSE
                for (auto& token : generate_tokens(body)) {
                    string chunk = format_sse_chunk(token);
                    sink.write(chunk.data(), chunk.size());
                }
                sink.write("data: [DONE]\n\n", 14);
                return false; // chiudi stream
            });
    }
};

} // namespace serius::server
```

---

## 4. Autenticazione API Key

```cpp
// Middleware per API key (opzionale, per deployment pubblici)
void auth_middleware(const httplib::Request& req, httplib::Response& res) {
    string auth = req.get_header_value("Authorization");
    if (!auth.starts_with("Bearer ")) {
        res.status = 401;
        res.set_content("{\"error\":\"Unauthorized\"}", "application/json");
        return;
    }
    string key = auth.substr(7);
    if (!validate_api_key(key)) {
        res.status = 403;
        res.set_content("{\"error\":\"Invalid API key\"}", "application/json");
    }
}
```

---

## 5. Struttura File

| File | Contenuto |
| --- | --- |
| `src/server/api_server.h/.cpp` | Server principale, routing |
| `src/server/handlers_chat.cpp` | Handler /v1/chat/completions |
| `src/server/handlers_embed.cpp` | Handler /v1/embeddings |
| `src/server/handlers_models.cpp` | Handler /v1/models |
| `src/server/sse_stream.h/.cpp` | SSE streaming helper |
| `src/server/auth.h/.cpp` | API key authentication |
| `src/server/rate_limiter.h/.cpp` | Rate limiting per IP/key |

---

## 6. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_openai_compat` | Client OpenAI Python funziona senza modifiche |
| `test_streaming_sse` | SSE produce chunks validi con [DONE] finale |
| `test_rate_limit` | Più di N richieste/sec → 429 Too Many Requests |
| `test_auth` | Richiesta senza Bearer → 401 |
| `test_embeddings` | /v1/embeddings ritorna vettori della dimensione corretta |
