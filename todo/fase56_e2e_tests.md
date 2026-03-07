# Fase 56 — End-to-End Integration Tests

> I test unitari verificano i singoli pezzi.
> I test E2E verificano che **tutto il percorso funzioni insieme**:
> dall'input utente alla risposta finale, passando per
> tokenizer → modello → sampler → detokenizer → API.

---

## 1. Scenari End-to-End

### 1.1 Full Inference Pipeline

```text
Scenario: il percorso completo di una richiesta

Input:  "What is 2+2?"
Flow:   Input → Tokenizer → Prefill → Decode → Detokenize → Output
Verify: Output contiene "4"
        Tokens generati < max_tokens
        Finish reason = "stop" (non "length")
```

### 1.2 Multi-Turn Chat

```text
Scenario: conversazione multi-turno con contesto

Turn 1: User: "My name is Marco"
        Verify: risposta coerente
Turn 2: User: "What is my name?"
        Verify: risposta contiene "Marco" (usa il contesto)
Turn 3: User: "Say it again"
        Verify: ancora "Marco" (KV cache funziona)
```

### 1.3 Model Load & Serve

```text
Scenario: pipeline completa server

1. serius serve --model model.gguf --port 9090
2. Aspetta health check: GET /health → 200
3. POST /v1/chat/completions → risposta valida JSON
4. POST /v1/chat/completions (stream=true) → SSE chunks validi
5. GET /v1/models → lista contiene il modello caricato
6. Shutdown graceful
```

### 1.4 Import → Convert → Infer

```text
Scenario: dall'import alla generazione

1. serius import --source test-model-tiny --format gguf
2. serius convert --input model.safetensors --output model-q4.gguf --quant int4
3. serius generate --model model-q4.gguf --prompt "Hello" --max-tokens 16
4. Verify: output non vuoto, non NaN, non token ripetuti
```

### 1.5 Safety Pipeline

```text
Scenario: guardrails funzionano end-to-end

1. Prompt innocuo → risposta normale
2. Prompt con trigger di sicurezza → risposta bloccata
3. Tentativo di jailbreak → guardrail interviene
4. Verifica: nessun contenuto tossico generato
```

---

## 2. Framework di Test E2E

```cpp
namespace serius::e2e {

class E2ETestRunner {
    string model_path_;
    int port_ = 9090;
    pid_t server_pid_;

public:
    void setup() {
        // Avvia server in background
        server_pid_ = fork_server(model_path_, port_);
        wait_for_health(port_, /*timeout_ms=*/30000);
    }

    void teardown() {
        kill(server_pid_, SIGTERM);
        waitpid(server_pid_, nullptr, 0);
    }

    // Test helper
    json chat(const vector<json>& messages, bool stream = false) {
        httplib::Client cli("localhost", port_);
        json body = {
            {"model", "default"},
            {"messages", messages},
            {"max_tokens", 128},
            {"stream", stream}
        };
        auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
        return json::parse(res->body);
    }
};

// Macro per test E2E
#define E2E_TEST(name) \
    TEST_F(E2ETestFixture, name)

} // namespace serius::e2e
```

---

## 3. Struttura File

| File | Contenuto |
| --- | --- |
| `tests/e2e/e2e_runner.h/.cpp` | Server lifecycle + HTTP helpers |
| `tests/e2e/test_inference.cpp` | Pipeline completa di inferenza |
| `tests/e2e/test_chat.cpp` | Multi-turn chat con contesto |
| `tests/e2e/test_api.cpp` | Tutti gli endpoint API |
| `tests/e2e/test_import.cpp` | Import → convert → infer |
| `tests/e2e/test_safety.cpp` | Guardrails end-to-end |
| `tests/e2e/test_streaming.cpp` | SSE streaming E2E |

---

## 4. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_e2e_basic_infer` | Prompt → risposta coerente (non token soup) |
| `test_e2e_multiturn` | Memoria contestuale funzionante |
| `test_e2e_server_lifecycle` | Start → serve → shutdown senza errori |
| `test_e2e_streaming` | SSE chunks tutti validi, termina con [DONE] |
| `test_e2e_safety` | Prompt pericoloso bloccato, prompt normale ok |
| `test_e2e_import_pipeline` | Import → quant → infer produce output sensato |
