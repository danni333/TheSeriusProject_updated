# Fase 33 — Orchestrazione dei Tool e Loop di Ragionamento

> Non basta vedere i tool; il modello deve sapere **quando** usarli.
> ReAct (Reasoning + Acting) e gestione intelligente del contesto.

---

## 1. ReAct Pattern

### Loop: Pensiero → Azione → Osservazione → Risposta

```
[USER] Quanti abitanti ha la capitale del Giappone?

[THOUGHT] Devo trovare la capitale del Giappone, poi cercare la popolazione.
[ACTION] {"tool":"search","args":{"query":"capitale Giappone"}}
[OBSERVATION] Tokyo è la capitale del Giappone.
[THOUGHT] Ora cerco la popolazione di Tokyo.
[ACTION] {"tool":"search","args":{"query":"popolazione Tokyo 2025"}}
[OBSERVATION] Tokyo ha circa 13.96 milioni di abitanti.
[ANSWER] La capitale del Giappone è Tokyo, con circa 13.96 milioni di abitanti.
```

### Implementazione

```cpp
class ReActEngine {
    Model& model;
    MCPClient& mcp;
    int max_steps = 5;

    string run(const string& user_query) {
        string context = format_system_prompt(mcp.list_tools());
        context += "\n[USER] " + user_query + "\n";

        for (int step = 0; step < max_steps; step++) {
            string output = model.generate(context, {.stop_tokens={"[ACTION]","[ANSWER]"}});
            context += output;

            if (contains(output, "[ANSWER]")) {
                return extract_after(output, "[ANSWER]");
            }
            if (contains(output, "[ACTION]")) {
                auto call = parse_action(output);
                json result = mcp.call_tool(call.name, call.args);
                string obs = truncate(result.dump(), MAX_OBS_LENGTH);
                context += "\n[OBSERVATION] " + obs + "\n[THOUGHT] ";
            }
        }
        return model.generate(context + "\n[ANSWER] ", {.max_tokens=512});
    }
};
```

---

## 2. Context Injection (Window Management)

### Problema
Un tool MCP potrebbe restituire 1000 righe. Il contesto ha un limite.

```cpp
string truncate_observation(const string& obs, int max_tokens, Model& model) {
    auto ids = model.tokenizer().encode(obs);
    if (ids.size() <= max_tokens) return obs;

    // Strategia 1: Tronca e segnala
    auto truncated = vector(ids.begin(), ids.begin() + max_tokens);
    return model.tokenizer().decode(truncated) + "\n[...TRUNCATED...]";
}

// Strategia 2: Summarize con il modello
string summarize_observation(const string& obs, Model& model) {
    return model.generate("Riassumi in 200 parole:\n" + obs, {.max_tokens=256});
}

// Strategia 3: Sliding window sul contesto totale
string manage_context(string& context, int max_ctx_tokens) {
    auto ids = tokenizer.encode(context);
    if (ids.size() > max_ctx_tokens) {
        // Mantieni system prompt + ultimi N token
        int keep_start = system_prompt_length;
        int keep_end = max_ctx_tokens - keep_start;
        // Tronca il mezzo
        context = decode(ids[0:keep_start]) + "\n[...]\n" + decode(ids[-keep_end:]);
    }
    return context;
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/agent/react.h/.cpp` | ReAct loop engine |
| `src/agent/context_manager.h/.cpp` | Truncation, summarization, sliding window |
| `src/agent/action_parser.h/.cpp` | Parse [ACTION] JSON dall'output |
| `tests/test_react.cpp` | Loop completo con mock MCP server |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_react_loop` | Pensiero → Azione → Osservazione → Risposta |
| `test_max_steps` | Dopo N step senza risposta, forza [ANSWER] |
| `test_truncation` | Osservazione lunga troncata correttamente |
| `test_context_window` | Contesto non supera max_ctx_tokens |
