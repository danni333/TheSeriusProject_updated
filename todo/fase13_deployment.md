# Fase 13 — Deployment Engine (PagedAttention, Continuous Batching, SGLang-class)

> Un LLM è inutile se resta chiuso in un `.exe`.
> Serve un'API per permettere ad altri programmi di interagire con il modello,
> con streaming delle risposte token per token.

---

## 1. C++ Web Server

### Librerie candidate

| Libreria | Pro | Contro |
|---|---|---|
| **Crow** | API simile a Flask, facile | Meno performante |
| **uWebSockets** | Ultra-veloce, WebSocket nativo | API più complessa |
| **cpp-httplib** | Header-only, zero dipendenze | No async nativo |

### Architettura consigliata: Crow + WebSocket

```
Client (Browser/App)
    │
    │  HTTP POST /v1/chat/completions
    │  WebSocket /v1/chat/stream
    │
    ▼
┌──────────────────────────┐
│     Crow Web Server       │
│     (thread pool I/O)     │
├──────────────────────────┤
│     Request Router        │
│  ├─ /v1/models            │
│  ├─ /v1/chat/completions  │
│  └─ /v1/chat/stream (WS)  │
├──────────────────────────┤
│     Inference Engine      │
│     (thread dedicato)     │
│  ├─ Model                 │
│  ├─ KV Cache              │
│  └─ Sampler               │
└──────────────────────────┘
```

### Endpoint REST (compatibile OpenAI)

```cpp
CROW_ROUTE(app, "/v1/chat/completions")
.methods("POST"_method)
([&](const crow::request& req) {
    auto body = json::parse(req.body);
    string prompt = format_chat(body["messages"]);
    auto params = parse_sampling_params(body);

    // Genera (bloccante per REST)
    string response = engine.generate(prompt, params);

    json result = {
        {"id", generate_id()},
        {"object", "chat.completion"},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"message", {{"role","assistant"},{"content",response}}},
            {"finish_reason", "stop"}
        }}}
    };
    return crow::response(200, result.dump());
});
```

---

## 2. Streaming delle Risposte

### SSE (Server-Sent Events) — per HTTP

```cpp
CROW_ROUTE(app, "/v1/chat/completions")  // con stream=true
.methods("POST"_method)
([&](const crow::request& req, crow::response& res) {
    auto body = json::parse(req.body);
    if (!body.value("stream", false)) {
        // ... risposta bloccante (sopra)
    }

    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");

    string prompt = format_chat(body["messages"]);
    auto params = parse_sampling_params(body);

    engine.generate_streaming(prompt, params,
        [&res](const string& token, bool is_done) {
            json chunk = {
                {"choices", {{
                    {"delta", {{"content", token}}},
                    {"finish_reason", is_done ? "stop" : nullptr}
                }}}
            };
            res.write("data: " + chunk.dump() + "\n\n");
            if (is_done) {
                res.write("data: [DONE]\n\n");
                res.end();
            }
        }
    );
});
```

### WebSocket — per streaming bidirezionale

```cpp
CROW_WEBSOCKET_ROUTE(app, "/v1/chat/stream")
.onopen([](crow::websocket::connection& conn) {
    log("New WebSocket connection");
})
.onmessage([&](crow::websocket::connection& conn, const string& msg,
                bool is_binary) {
    auto request = json::parse(msg);
    string prompt = format_chat(request["messages"]);

    engine.generate_streaming(prompt, default_params,
        [&conn](const string& token, bool is_done) {
            json chunk = {{"token", token}, {"done", is_done}};
            conn.send_text(chunk.dump());
        }
    );
})
.onclose([](crow::websocket::connection& conn, const string& reason) {
    log("Connection closed: " + reason);
});
```

### Generate Streaming (engine side)

```cpp
void InferenceEngine::generate_streaming(
    const string& prompt,
    const SamplingParams& params,
    function<void(string, bool)> callback)
{
    auto ids = tokenizer_.encode(prompt);

    for (int step = 0; step < params.max_tokens; step++) {
        Tensor logits = model_.forward(ids, kv_cache_);
        int next_id = sampler_.sample(logits, params);

        if (next_id == EOS_TOKEN) {
            callback("", true);  // fine
            break;
        }

        string token_str = tokenizer_.decode({next_id});
        callback(token_str, false);  // invia al client
        ids.push_back(next_id);
    }
}
```

---

## 3. SIMD-JSON (Parsing Ultra-Veloce)

### Perché

Il parsing JSON standard (nlohmann/json) è lento per richieste frequenti. **simdjson** usa istruzioni SIMD per parsare GB/s di JSON.

### Integrazione

```cpp
#include <simdjson.h>

simdjson::ondemand::parser parser;

// Parse di una richiesta
auto doc = parser.iterate(req.body);
string model = doc["model"].get_string().value();
double temp = doc["temperature"].get_double().value();
auto messages = doc["messages"].get_array().value();

for (auto msg : messages) {
    string role = msg["role"].get_string().value();
    string content = msg["content"].get_string().value();
    // ...
}
```

### Performance

| Libreria | Throughput | Note |
|---|---|---|
| nlohmann/json | ~200 MB/s | Bella API, lenta |
| RapidJSON | ~800 MB/s | Veloce, API verbosa |
| **simdjson** | **~3 GB/s** | Usa AVX2/NEON, zero-copy |

---

## 4. Request Queue e Batching

Per servire più utenti contemporaneamente:

```cpp
class RequestQueue {
    struct Request {
        string prompt;
        SamplingParams params;
        function<void(string, bool)> callback;
    };

    concurrent_queue<Request> queue_;
    thread worker_;

    void worker_loop() {
        while (running_) {
            // Raccogli fino a MAX_BATCH richieste
            vector<Request> batch;
            queue_.try_pop_batch(batch, MAX_BATCH, TIMEOUT_MS);

            if (batch.empty()) continue;

            // Continuous batching: processa tutte insieme
            auto results = engine_.batch_generate(batch);

            for (size_t i = 0; i < batch.size(); i++)
                batch[i].callback(results[i].token, results[i].done);
        }
    }
};
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/server/server.h/.cpp` | Crow server setup, routing |
| `src/server/api_handlers.h/.cpp` | Handler REST e WebSocket |
| `src/server/streaming.h/.cpp` | SSE e WebSocket streaming |
| `src/server/request_queue.h/.cpp` | Queue + continuous batching |
| `src/server/json_parser.h/.cpp` | Wrapper simdjson |
| `tests/test_server.cpp` | Test endpoint (localhost) |
| `tests/test_streaming.cpp` | Test streaming token per token |

### Dipendenze

| Libreria | Versione | CMake |
|---|---|---|
| Crow | 1.0+ | `find_package(Crow)` o header-only |
| simdjson | 3.0+ | `FetchContent` o vcpkg |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_completions` | POST /v1/chat/completions → risposta JSON valida |
| `test_streaming_sse` | Stream: riceve token uno alla volta, termina con [DONE] |
| `test_websocket` | Connessione WS, invio prompt, ricezione token |
| `test_simdjson` | Parse di request body complesso senza errori |
| `test_concurrent` | 10 richieste parallele → tutte servite senza crash |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_latency` | Time-to-first-token (TTFT) |
| `bench_throughput` | Token/secondo con 1, 4, 8 client paralleli |
| `bench_json_parse` | simdjson vs nlohmann su payload tipico |
