# Fase 60 — Tool Server (MCP Locale & Remoto)

> Il modello da solo risponde a domande. Con i tool diventa un **agente**.
> Supporto per tool server locali (processi sulla stessa macchina),
> remoti (via HTTP/WebSocket), e compatibilità con il protocollo MCP.

---

## 1. Architettura

```text
Serius LLM
    │
    ├── Tool Router (sceglie quale server chiamare)
    │
    ├── Local Tool Servers (processi separati, stdio/pipe)
    │   ├── calculator-server      → Calcoli matematici
    │   ├── filesystem-server      → Lettura/scrittura file
    │   └── browser-server         → Web scraping
    │
    ├── Remote Tool Servers (HTTP/WebSocket)
    │   ├── https://api.weather.com → Meteo
    │   ├── https://search.api     → Ricerca web
    │   └── wss://db-server:9090   → Database queries
    │
    └── Built-in Tools (compilati nel binario)
        ├── python_exec            → Esecuzione codice Python
        ├── shell_exec             → Esecuzione comandi shell
        └── web_fetch              → HTTP GET/POST
```

---

## 2. Protocollo MCP (Model Context Protocol)

### Configurazione Tool Server

```json
// config/tools.json
{
    "tool_servers": [
        {
            "name": "filesystem",
            "type": "local",
            "command": ["node", "servers/filesystem-server/index.js"],
            "transport": "stdio"
        },
        {
            "name": "weather",
            "type": "remote",
            "url": "https://weather-mcp.example.com",
            "transport": "http",
            "auth": { "type": "bearer", "token_env": "WEATHER_API_KEY" }
        },
        {
            "name": "database",
            "type": "remote",
            "url": "wss://db-server:9090/mcp",
            "transport": "websocket"
        }
    ]
}
```

### Client MCP

```cpp
namespace serius::tools {

class MCPClient {
public:
    // Connessione a tool server locale (stdio)
    void connect_local(const string& command, const vector<string>& args);

    // Connessione a tool server remoto (HTTP/SSE)
    void connect_remote(const string& url, const string& auth_token = "");

    // Discovery: lista dei tool disponibili
    vector<ToolSchema> list_tools();

    // Invocazione tool
    json call_tool(const string& tool_name, const json& arguments);

    // Risorse (read-only data)
    json read_resource(const string& uri);
};

class ToolRouter {
    vector<MCPClient> clients_;
    map<string, int> tool_to_client_;  // tool_name → client index

public:
    void load_config(const string& config_path);

    // Auto-discovery: connetti tutti i server, mappa i tool
    void discover_all();

    // Chiamata unificata: il router trova il server giusto
    json call(const string& tool_name, const json& args);

    // Schema unificato per il modello (tutti i tool da tutti i server)
    vector<ToolSchema> all_tools();
};

} // namespace serius::tools
```

---

## 3. Tool Server Custom (come crearne uno)

```cpp
// Esempio: tool server custom in C++ con Serius SDK
#include <serius/tools/server.h>

int main() {
    serius::tools::MCPServer server;

    server.register_tool({
        .name = "translate",
        .description = "Traduce testo tra lingue",
        .parameters = {
            {"text", "string", "Testo da tradurre", true},
            {"target_lang", "string", "Lingua target (es. 'it', 'en')", true}
        },
        .handler = [](const json& args) -> json {
            string text = args["text"];
            string lang = args["target_lang"];
            return {{"translated", do_translate(text, lang)}};
        }
    });

    server.run_stdio();  // Comunica via stdin/stdout (MCP standard)
}
```

---

## 4. Struttura File

| File | Contenuto |
| --- | --- |
| `src/tools/mcp_client.h/.cpp` | Client MCP (stdio + HTTP + WebSocket) |
| `src/tools/mcp_server.h/.cpp` | SDK per creare tool server custom |
| `src/tools/tool_router.h/.cpp` | Router che unifica tutti i server |
| `src/tools/builtin/` | Tool built-in (python_exec, web_fetch) |
| `config/tools.json` | Configurazione tool server |
| `docs/tool-server-guide.md` | Guida per creare tool server custom |
| `examples/tool-server-translate/` | Esempio tool server completo |

---

## 5. Piano di Verifica

| Test | Verifica |
| --- | --- |
| `test_mcp_stdio` | Client ↔ Server locale via stdio |
| `test_mcp_http` | Client ↔ Server remoto via HTTP |
| `test_tool_discovery` | list_tools() ritorna schema corretto |
| `test_tool_call` | call_tool() ritorna risultato atteso |
| `test_tool_router` | Router instrada al server giusto |
| `test_custom_server` | Server creato con SDK risponde correttamente |
