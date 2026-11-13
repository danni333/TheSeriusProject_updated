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

---

## Stato avanzamento

**Fatto e verificato** (`tests/test_e2e.cpp`, 6 test verdi, ~21s):

- `E2EServer.*` — fixture che carica il GGUF reale
  `models/smollm2-q4/SmolLM2-135M-Instruct-Q4_0.gguf` in un `Server`
  in-process (CPU, deterministico) e lo serve su porta effimera:
  - `FullPipelineChatCompletion` — prompt → risposta: oggetto
    `chat.completion`, `finish_reason`, `usage` coerenti (1.1)
  - `MultiTurnChat` — history a 3 turni accettata, output valido (1.2,
    asserzione solo strutturale: un 135M non garantisce recall)
  - `AllEndpoints` — `/health` (con `model_loaded`), `/v1/models`,
    `/v1/models/{id}`, `/v1/completions`, `/v1/embeddings`, `/metrics` (1.3)
  - `StreamingSSE` — parse riga-per-riga degli eventi `data:`, ogni chunk
    JSON valido con `choices[0].delta`, `[DONE`] finale, contenuto
    accumulato non vuoto
  - `ErrorHandling` — JSON malformato → 400, rotta sconosciuta → 404
- `E2ECli.ServeBinaryLifecycle` — spawn reale di `serius.exe serve
  --model models/smollm2-q4 --gpu cpu` via `CreateProcess`, attesa
  `/health` (fino a 90s per il load CPU), una chat completion, poi
  `TerminateProcess` (1.3 completo, inclusa la parte CLI che il design
  POSIX del piano non copriva)
- Registrazione in `tests/CMakeLists.txt` (`test_e2e`,
  `gtest_discover_tests`, define `SERIUS_EXE_PATH`/`SERIOUS_REPO_ROOT`).

**Note di design:**

- Il piano originale prevedeva `fork`/`SIGTERM`/`waitpid`: su Windows si
  usa `CreateProcessA` + `TerminateProcess` (il test è `#ifdef _WIN32`,
  skip altrove).
- Skip pulito (`GTEST_SKIP`) se `models/smollm2-q4` non è presente.

**Fatto — scenario 1.4 (Convert → Infer):**

- `serius convert` implementato (`src/cli/cmd_convert.cpp` +
  `src/io/gguf_export.cpp`): esporta un GGUF come directory HF —
  `model.safetensors` FP32 dequantizzato (tensor rinominati
  `blk.N.*` → `model.layers.N.*`, shape `{in,out}` → `{out,in}`,
  unpermute RoPE Q/K per arch `llama`), `config.json` generato dai
  metadati GGUF, copia `tokenizer.json` e affini dalla directory
  sorgente.
- `E2ECli.ConvertThenInfer` in `tests/test_e2e.cpp`: spawn reale di
  `serius.exe convert` → verifica artefatti → `Server` carica la
  directory convertita → chat completion 200 con contenuto non vuoto.
  7/7 test E2E verdi.
- `serius generate` non esiste come comando a parte: `serius chat
  --prompt "..." --max-tokens N` è il equivalente non interattivo
  (usato anche nel test manuale di verifica).
- Scenario "import" coperto a livello di asset locali: `serius import`
  esiste ed è limitato al download HF/URL (la parte "convert" e
  "infer" è ora reale); un E2E di download richiede rete e non è
  nel test.

**Rimasto:**
- 1.5 Safety pipeline E2E: i guardrails non sono cablati nel path HTTP
  del server.
- Recall semantico multi-turno ("Marco") e verifica "output contiene 4"
  per "2+2": asserzioni troppo fragili su un 135M Q4 — da rivalutare con
  un modello più capace o mock deterministici.
