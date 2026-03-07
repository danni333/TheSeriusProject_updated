# Fase 34 — Sicurezza dei Permessi MCP (Sandboxing)

> Far eseguire comandi a un LLM tramite MCP è rischioso.
> Serve autorizzazione umana e scrubbing dei dati sensibili.

---

## 1. Human-in-the-Loop

### Classificazione rischio tool

```cpp
enum RiskLevel { SAFE, MODERATE, DANGEROUS };

map<string, RiskLevel> tool_risk = {
    {"read_file",    SAFE},
    {"search",       SAFE},
    {"write_file",   MODERATE},
    {"run_command",  DANGEROUS},
    {"send_email",   DANGEROUS},
    {"delete_file",  DANGEROUS},
};
```

### Interceptor

```cpp
class MCPGuard {
    function<bool(const string&, const json&)> approval_callback;

    json guarded_call(MCPClient& mcp, const string& tool, const json& args) {
        RiskLevel risk = get_risk(tool);
        if (risk == DANGEROUS) {
            // Chiedi approvazione all'utente
            bool approved = approval_callback(tool, args);
            if (!approved) return {{"error", "Operazione rifiutata dall'utente"}};
        }
        if (risk >= MODERATE) {
            log_action(tool, args);  // audit trail
        }
        return mcp.call_tool(tool, args);
    }
};
```

---

## 2. Token Scrubbing

### Pattern da rimuovere dall'output MCP

```cpp
class Scrubber {
    vector<regex> patterns = {
        regex(R"((?:api[_-]?key|token|secret)\s*[:=]\s*\S+)", regex::icase),
        regex(R"((?:password|passwd|pwd)\s*[:=]\s*\S+)", regex::icase),
        regex(R"(\b[A-Za-z0-9+/]{40,}={0,2}\b)"),  // base64 lunghi
        regex(R"(sk-[a-zA-Z0-9]{32,})"),              // OpenAI keys
        regex(R"(ghp_[a-zA-Z0-9]{36})"),              // GitHub tokens
    };

    string scrub(const string& text) {
        string result = text;
        for (auto& p : patterns)
            result = regex_replace(result, p, "[REDACTED]");
        return result;
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/mcp/guard.h/.cpp` | MCPGuard con approval e risk classification |
| `src/mcp/scrubber.h/.cpp` | Token/password scrubbing |
| `src/mcp/audit.h/.cpp` | Audit trail di tutte le azioni |
| `tests/test_guard.cpp` | Tool pericolosi bloccati senza approvazione |
| `tests/test_scrubber.cpp` | API key e password rimosse |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_safe_tool` | Tool SAFE eseguito senza approvazione |
| `test_dangerous_blocked` | Tool DANGEROUS bloccato senza approval |
| `test_scrub_apikey` | `api_key=sk-abc123` → `[REDACTED]` |
| `test_scrub_password` | `password: secret` → `[REDACTED]` |
| `test_audit_log` | Ogni azione registrata con timestamp |
