# Fase 22 — Agentic RAG & Structured Tool Use (GraphRAG, MCP Tools)

> Il modello conosce solo ciò che ha studiato. RAG aggiunge conoscenza
> aggiornata, Function Calling gli permette di agire nel mondo reale.

---

## 1. Vector Database Integration (RAG)

### Retrieval-Augmented Generation

```
Domanda utente
     │
     ▼
Embedding della domanda (tramite il modello)
     │
     ▼
Ricerca vettoriale nel DB (cosine similarity)
     │
     ▼
Top-K documenti rilevanti
     │
     ▼
Prompt = [System] + [Documenti trovati] + [Domanda]
     │
     ▼
LLM genera risposta basata sui documenti
```

### Client C++ per Vector DB

```cpp
class VectorDBClient {
    // Supporta: Qdrant, Milvus, Weaviate, o DB locale
    string endpoint;
    HttpClient http;

    // Cerca i documenti più simili
    vector<SearchResult> search(const vector<float>& query_embedding,
                                 int top_k = 5, float threshold = 0.7) {
        json body = {
            {"vector", query_embedding},
            {"limit", top_k},
            {"score_threshold", threshold}
        };
        auto response = http.post(endpoint + "/search", body);
        return parse_results(response);
    }

    // Indicizza un nuovo documento
    void upsert(const string& id, const vector<float>& embedding,
                 const json& metadata) {
        json body = {{"id", id}, {"vector", embedding}, {"payload", metadata}};
        http.put(endpoint + "/points", body);
    }
};
```

### Embedding locale (senza chiamate esterne)

```cpp
// Usa il modello stesso per generare embeddings
vector<float> compute_embedding(Model& model, const string& text) {
    auto ids = tokenizer.encode(text);
    Tensor hidden = model.forward_hidden(ids);  // ultimo hidden state
    // Mean pooling
    Tensor embedding = mean(hidden, /*dim=*/0);  // (d_model,)
    // Normalizza L2
    float norm = l2_norm(embedding);
    for (auto& v : embedding) v /= norm;
    return embedding.to_vector();
}
```

### Prompt con contesto RAG

```cpp
string build_rag_prompt(const string& question,
                         const vector<SearchResult>& docs) {
    string context = "";
    for (int i = 0; i < docs.size(); i++)
        context += format("[Doc {}] {}\n\n", i+1, docs[i].text);

    return format(
        "[SYSTEM] Rispondi basandoti SOLO sui documenti forniti. "
        "Cita le fonti.\n\n"
        "[DOCUMENTI]\n{}\n"
        "[DOMANDA] {}\n"
        "[RISPOSTA]", context, question);
}
```

---

## 2. Function Calling (Tool Use)

### Concetto

Il modello genera JSON strutturato che il codice C++ intercetta per eseguire azioni reali.

### Formato output del modello

```json
{
    "function": "smart_home.set_light",
    "arguments": {
        "room": "cucina",
        "state": "off"
    }
}
```

### Tool Registry

```cpp
struct ToolDefinition {
    string name;
    string description;
    json parameters_schema;  // JSON Schema
    function<json(const json&)> handler;
};

class ToolRegistry {
    map<string, ToolDefinition> tools;

    void register_tool(ToolDefinition tool) {
        tools[tool.name] = tool;
    }

    json execute(const string& name, const json& args) {
        if (!tools.count(name))
            return {{"error", "Tool not found: " + name}};
        // Valida gli argomenti contro lo schema
        if (!validate_schema(args, tools[name].parameters_schema))
            return {{"error", "Invalid arguments"}};
        return tools[name].handler(args);
    }
};
```

### Detection nel loop di generazione

```cpp
void generate_with_tools(Model& model, ToolRegistry& tools,
                          const string& prompt) {
    string output = model.generate(prompt);

    // Cerca pattern di function call nell'output
    if (auto call = detect_function_call(output)) {
        json result = tools.execute(call->name, call->args);
        // Reinserisci il risultato nel contesto
        string new_prompt = prompt + output +
            "\n[TOOL_RESULT] " + result.dump() + "\n[ASSISTANT]";
        // Continua la generazione con il risultato
        model.generate(new_prompt);
    }
}
```

### Esempio: registrazione tool

```cpp
registry.register_tool({
    "weather.get_current",
    "Ottieni il meteo corrente per una città",
    {{"type","object"}, {"properties", {
        {"city", {{"type","string"}}},
        {"unit", {{"type","string"}, {"enum",{"celsius","fahrenheit"}}}}
    }}},
    [](const json& args) -> json {
        string city = args["city"];
        // Chiamata HTTP a API meteo
        auto resp = http_get("api.weather.com/current?q=" + city);
        return {{"temperature", resp["temp"]}, {"condition", resp["desc"]}};
    }
});
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/rag/vector_client.h/.cpp` | Client per vector database |
| `src/rag/embeddings.h/.cpp` | Embedding locale + mean pooling |
| `src/rag/retriever.h/.cpp` | Pipeline RAG completa |
| `src/tools/registry.h/.cpp` | Tool registry + execution |
| `src/tools/parser.h/.cpp` | Detection function call nell'output |
| `tests/test_rag.cpp` | Embedding, search, prompt building |
| `tests/test_tools.cpp` | Registry, validation, execution |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_embedding` | Output normalizzato (norma L2 = 1) |
| `test_similarity` | Testi simili → cosine similarity > 0.8 |
| `test_rag_prompt` | Prompt contiene documenti e domanda |
| `test_tool_register` | Tool registrato e ritrovabile |
| `test_tool_execute` | Handler chiamato con argomenti corretti |
| `test_tool_validate` | Argomenti invalidi → errore |
| `test_function_detect` | Output JSON → function call rilevata |
