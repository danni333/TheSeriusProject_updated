# Fase 32 — MCP Client Implementation (Il Connettore Universale)

> L'LLM diventa un "Client" nel protocollo MCP (Model Context Protocol),
> capace di scoprire e usare tool esposti da server MCP esterni.

---

## 1. Protocollo JSON-RPC

### Formato messaggi

```json
// Richiesta
{"jsonrpc":"2.0", "id":1, "method":"tools/list", "params":{}}

// Risposta
{"jsonrpc":"2.0", "id":1, "result":{"tools":[
  {"name":"read_file","description":"Legge un file","inputSchema":{...}},
  {"name":"run_query","description":"Esegue query SQL","inputSchema":{...}}
]}}
```

### Trasporti supportati

| Trasporto | Uso |
|---|---|
| **stdio** | Processo locale (stdin/stdout) |
| **HTTP + SSE** | Server remoto |

### Implementazione Client

```cpp
class MCPClient {
    enum Transport { STDIO, HTTP_SSE };
    Transport transport_;
    // stdio
    FILE* proc_stdin_ = nullptr;
    FILE* proc_stdout_ = nullptr;
    // http
    HttpClient http_;
    string endpoint_;
    int next_id_ = 1;

    json send_request(const string& method, const json& params = {}) {
        json request = {
            {"jsonrpc", "2.0"},
            {"id", next_id_++},
            {"method", method},
            {"params", params}
        };
        if (transport_ == STDIO) {
            string msg = request.dump() + "\n";
            fwrite(msg.c_str(), 1, msg.size(), proc_stdin_);
            fflush(proc_stdin_);
            // Leggi risposta
            char buf[65536];
            fgets(buf, sizeof(buf), proc_stdout_);
            return json::parse(buf);
        } else {
            auto resp = http_.post(endpoint_, request);
            return json::parse(resp.body);
        }
    }
};
```

---

## 2. Tool Discovery

```cpp
struct MCPTool {
    string name;
    string description;
    json input_schema;
};

vector<MCPTool> MCPClient::list_tools() {
    auto resp = send_request("tools/list");
    vector<MCPTool> tools;
    for (auto& t : resp["result"]["tools"])
        tools.push_back({t["name"], t["description"], t["inputSchema"]});
    return tools;
}

json MCPClient::call_tool(const string& name, const json& args) {
    return send_request("tools/call", {{"name", name}, {"arguments", args}});
}
```

### Integrazione con il modello

```cpp
// Converti tool MCP in formato function calling per il modello
string format_tools_for_prompt(const vector<MCPTool>& tools) {
    string desc = "Available tools:\n";
    for (auto& t : tools)
        desc += format("- {}: {} | Args: {}\n",
                        t.name, t.description, t.input_schema.dump());
    return desc;
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/mcp/client.h/.cpp` | MCPClient (JSON-RPC, stdio/HTTP) |
| `src/mcp/transport_stdio.h/.cpp` | Trasporto via stdin/stdout |
| `src/mcp/transport_http.h/.cpp` | Trasporto via HTTP + SSE |
| `src/mcp/discovery.h/.cpp` | Tool discovery e formatting |
| `tests/test_mcp_client.cpp` | Mock server, list_tools, call_tool |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_jsonrpc_format` | Messaggi conformi a JSON-RPC 2.0 |
| `test_list_tools` | Tool discovery ritorna lista corretta |
| `test_call_tool` | Chiamata tool ritorna risultato |
| `test_stdio_transport` | Comunicazione via stdin/stdout funziona |
| `test_error_handling` | Errori JSON-RPC gestiti correttamente |
