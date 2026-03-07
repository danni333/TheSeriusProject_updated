# Fase 63 — AI Swarm (Sciame di Agenti Autonomi + OpenRouter)

> Un supervisor delega compiti (Fase 40). Uno **swarm** è diverso:
> agenti autonomi che si coordinano come uno sciame, si dividono il lavoro
> dinamicamente, e possono essere sia **locali** che **cloud** (OpenRouter).
> Ogni agente decide cosa fare, chiede aiuto agli altri, e collabora.

---

## 1. Differenza tra Supervisor (Fase 40) e Swarm

| | Supervisor (Fase 40) | Swarm (Fase 63) |
| --- | --- | --- |
| **Controllo** | Centralizzato (1 boss) | Decentralizzato (pari tra pari) |
| **Piano** | Il supervisor pianifica tutto | Ogni agente pianifica autonomamente |
| **Comunicazione** | Hub & spoke (boss ↔ worker) | Peer-to-peer (tutti ↔ tutti) |
| **Errore** | Se il supervisor fallisce, tutto fallisce | Resiliente: gli altri compensano |
| **Modelli** | Solo locali | **Locali + Cloud (OpenRouter)** |

---

## 2. Architettura Swarm

```text
┌─────────────────────────────────────────────────┐
│                  SWARM BUS                      │
│          (message queue in-process)             │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ Agent A   │  │ Agent B   │  │ Agent C   │   │
│  │ Serius 7B │  │ Serius 1B │  │ OpenRouter│   │
│  │ (locale)  │  │ (locale)  │  │ GPT-4/etc │   │
│  │ Ruolo:    │  │ Ruolo:    │  │ Ruolo:    │   │
│  │ Planner   │  │ Coder     │  │ Reviewer  │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  │
│        │              │              │          │
│        └──────────────┼──────────────┘          │
│                       │                         │
│                 ┌─────┴─────┐                   │
│                 │ Agent D   │                   │
│                 │ Serius 3B │                   │
│                 │ (locale)  │                   │
│                 │ Ruolo:    │                   │
│                 │ Tester    │                   │
│                 └───────────┘                   │
└─────────────────────────────────────────────────┘
```

---

## 3. OpenRouter Integration

### Configurazione agenti misti (locali + cloud)

```json
// config/swarm.json
{
    "swarm_name": "dev_team",
    "max_agents": 6,
    "agents": [
        {
            "name": "planner",
            "role": "Decompone il problema in sotto-task",
            "backend": "local",
            "model": "models/SeriusModel-7B-Q4.gguf"
        },
        {
            "name": "coder",
            "role": "Scrive codice C++ e Python",
            "backend": "local",
            "model": "models/SeriusModel-3B-Q4.gguf"
        },
        {
            "name": "reviewer",
            "role": "Revisiona codice e trova bug",
            "backend": "openrouter",
            "model": "anthropic/claude-3.5-sonnet",
            "api_key_env": "OPENROUTER_API_KEY"
        },
        {
            "name": "researcher",
            "role": "Cerca informazioni e documentazione",
            "backend": "openrouter",
            "model": "google/gemini-2.0-flash",
            "api_key_env": "OPENROUTER_API_KEY"
        },
        {
            "name": "tester",
            "role": "Scrive e esegue test",
            "backend": "local",
            "model": "models/SeriusModel-1B-Q8.gguf"
        }
    ],
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "fallback_model": "meta-llama/llama-3-8b-instruct",
        "max_tokens_per_request": 4096,
        "budget_limit_usd": 5.00
    }
}
```

### Client OpenRouter

```cpp
namespace serius::swarm {

class OpenRouterClient {
    string base_url_ = "https://openrouter.ai/api/v1";
    string api_key_;
    float budget_spent_ = 0;
    float budget_limit_;

public:
    void init(const string& api_key, float budget_limit_usd) {
        api_key_ = api_key;
        budget_limit_ = budget_limit_usd;
    }

    // Compatibile OpenAI → funziona con qualsiasi modello su OpenRouter
    string chat(const string& model, const vector<Message>& messages,
                float temperature = 0.7, int max_tokens = 1024) {
        if (budget_spent_ >= budget_limit_) {
            spdlog::warn("OpenRouter budget limit reached: ${:.2f}", budget_spent_);
            return "[BUDGET EXCEEDED — switching to local fallback]";
        }

        json body = {
            {"model", model},
            {"messages", format_messages(messages)},
            {"temperature", temperature},
            {"max_tokens", max_tokens}
        };

        httplib::Client cli(base_url_);
        cli.set_bearer_token(api_key_);
        auto res = cli.Post("/chat/completions", body.dump(), "application/json");

        auto resp = json::parse(res->body);
        // Traccia costi
        if (resp.contains("usage")) {
            budget_spent_ += estimate_cost(model, resp["usage"]);
        }
        return resp["choices"][0]["message"]["content"];
    }

    // Lista modelli disponibili
    vector<ModelInfo> list_models();

    // Costo stimato
    float get_budget_spent() const { return budget_spent_; }
};

} // namespace serius::swarm
```

---

## 4. Swarm Engine

```cpp
namespace serius::swarm {

// Messaggio tra agenti
struct SwarmMessage {
    string from;         // nome agente mittente
    string to;           // "all" per broadcast, o nome specifico
    string type;         // "task", "result", "question", "review"
    string content;
    json metadata;
};

// Singolo agente nello swarm
class SwarmAgent {
    string name_;
    string role_;
    // Backend: locale (Model*) o cloud (OpenRouterClient*)
    variant<Model*, OpenRouterClient*> backend_;
    string model_id_;   // per OpenRouter: "anthropic/claude-3.5-sonnet"
    queue<SwarmMessage> inbox_;

public:
    // L'agente decide autonomamente cosa fare
    SwarmMessage think(const SwarmMessage& incoming) {
        string prompt = format(
            "Sei l'agente '{}' con ruolo: {}\n"
            "Messaggio ricevuto da '{}': {}\n\n"
            "Decidi cosa fare:\n"
            "1. Rispondi con il risultato\n"
            "2. Chiedi aiuto a un altro agente\n"
            "3. Esegui un tool\n",
            name_, role_, incoming.from, incoming.content);

        string response = generate(prompt);
        return parse_agent_response(response);
    }
};

// Orchestratore dello swarm
class SwarmEngine {
    vector<unique_ptr<SwarmAgent>> agents_;
    queue<SwarmMessage> bus_;        // message bus condiviso
    OpenRouterClient openrouter_;
    int max_rounds_ = 10;

public:
    void load_config(const string& config_path);

    string run(const string& user_task) {
        // Broadcast del task a tutti gli agenti
        bus_.push({"user", "all", "task", user_task, {}});

        for (int round = 0; round < max_rounds_; round++) {
            // Ogni agente processa i messaggi nella sua inbox
            for (auto& agent : agents_) {
                while (auto msg = agent->poll_inbox()) {
                    auto response = agent->think(*msg);
                    route_message(response);  // invia a destinatario
                }
            }
            // Check: qualcuno ha prodotto il risultato finale?
            if (auto result = check_final_result())
                return *result;
        }
        return synthesize_partial_results();
    }
};

} // namespace serius::swarm
```

---

## 5. CLI

```bash
# Avvia swarm con configurazione
serius swarm --config config/swarm.json \
    --task "Scrivi un HTTP server in C++ con endpoint /hello"

# Output:
# 🐝 Swarm "dev_team" avviato (5 agenti: 3 locali, 2 OpenRouter)
# [planner]    → Decomposizione task in 3 sotto-task
# [coder]      → Scrittura server.cpp (locale, SeriusModel-3B)
# [reviewer]   → Code review (OpenRouter, Claude 3.5)
# [coder]      → Fix basati sul review
# [tester]     → Test di compilazione e funzionamento
# ✅ Task completato in 4 round | Costo OpenRouter: $0.08

# Swarm interattivo (chat con lo sciame)
serius swarm --config config/swarm.json --interactive

# Budget check
serius swarm --budget
# OpenRouter: $0.08 / $5.00 spesi
```

---

## 6. Strategie di Routing

| Strategia | Quando usarla |
| --- | --- |
| **Local-first** | Usa modelli locali, cloud solo se serve qualità alta |
| **Cost-optimized** | Instrada ai modelli più economici che possano risolvere il task |
| **Quality-first** | Usa i modelli migliori (cloud) per task critici |
| **Fallback chain** | Locale → OpenRouter free tier → OpenRouter paid |

```cpp
// Esempio: routing intelligente
string SwarmRouter::route(const string& task, float complexity) {
    if (complexity < 0.3f)
        return "local:SeriusModel-1B";     // task semplice → modello piccolo
    else if (complexity < 0.7f)
        return "local:SeriusModel-7B";     // task medio → modello locale grande
    else if (budget_remaining() > 0.10f)
        return "openrouter:claude-3.5";    // task complesso → cloud
    else
        return "local:SeriusModel-7B";     // budget esaurito → fallback locale
}
```

---

## 7. Struttura File

| File | Contenuto |
| --- | --- |
| `src/swarm/swarm_engine.h/.cpp` | Orchestratore swarm |
| `src/swarm/swarm_agent.h/.cpp` | Agente autonomo |
| `src/swarm/message_bus.h/.cpp` | Message queue in-process |
| `src/swarm/openrouter_client.h/.cpp` | Client OpenRouter (OpenAI-compat) |
| `src/swarm/router.h/.cpp` | Routing intelligente local/cloud |
| `src/swarm/budget.h/.cpp` | Budget tracking e limiti |
| `src/cli/cmd_swarm.cpp` | Comando `serius swarm` |
| `config/swarm.json` | Configurazione swarm |

---

## 8. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_swarm_local` | 3 agenti locali collaborano e producono risultato |
| `test_openrouter_client` | Chiamata a OpenRouter ritorna risposta valida |
| `test_mixed_swarm` | Agenti locali + OpenRouter collaborano |
| `test_budget_limit` | Supera budget → fallback a locale |
| `test_peer_to_peer` | Agente A chiede aiuto ad Agente B direttamente |
| `test_fault_tolerance` | 1 agente fallisce → gli altri compensano |
| `test_routing` | Task semplice → locale, task complesso → cloud |
