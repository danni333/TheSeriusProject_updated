# Fase 21 — Mechanistic Interpretability (SAE, Probing, Circuit Analysis)

> Se il modello viene usato in ambiti critici (medicina, legge),
> non può essere una "scatola nera". Servono strumenti per capire
> **cosa** guarda e **perché** decide.

---

## 1. Attention Visualization

### Obiettivo

Esportare i pesi dell'attention in tempo reale per vedere quali token il modello sta "guardando" mentre genera.

### Cattura delle Attention Weights

```cpp
struct AttentionCapture {
    // Per ogni layer, per ogni head: matrice (seq × seq)
    vector<vector<Tensor>> attention_weights;
    bool capturing = false;

    void start() { capturing = true; attention_weights.clear(); }
    void stop()  { capturing = false; }

    void record(int layer, int head, const Tensor& weights) {
        if (!capturing) return;
        if (attention_weights.size() <= layer)
            attention_weights.resize(layer + 1);
        if (attention_weights[layer].size() <= head)
            attention_weights[layer].resize(head + 1);
        attention_weights[layer][head] = weights.copy();
    }
};
```

### Inserzione nel forward dell'Attention

```cpp
Tensor multi_head_attention(const Tensor& x, ..., AttentionCapture* capture) {
    // ... calcolo Q, K, V ...
    Tensor attn_weights = softmax(scores);  // (n_heads, seq, seq)

    if (capture && capture->capturing)
        for (int h = 0; h < n_heads; h++)
            capture->record(layer_id, h, attn_weights[h]);

    return matmul(attn_weights, V);
}
```

### Esportazione

```cpp
// JSON per visualizzazione web (D3.js, BertViz-style)
void export_attention_json(const AttentionCapture& cap,
                            const vector<string>& tokens,
                            const string& path) {
    json j;
    j["tokens"] = tokens;
    for (int l = 0; l < cap.attention_weights.size(); l++)
        for (int h = 0; h < cap.attention_weights[l].size(); h++)
            j["layers"][l]["heads"][h] = tensor_to_array(cap.attention_weights[l][h]);
    write_file(path, j.dump());
}

// Heatmap CSV per analisi rapida
void export_attention_csv(const Tensor& attn, const string& path);
```

---

## 2. Neuron Activation Mapping

### Obiettivo

Capire **quale neurone** o **quale esperto** (in MoE) si è attivato per una specifica parola/affermazione.

### Activation Probe

```cpp
class ActivationProbe {
    map<string, vector<Tensor>> activations;

    void hook(const string& layer_name, const Tensor& output) {
        activations[layer_name].push_back(output.copy());
    }

    // Trova i neuroni più attivi per un token specifico
    vector<pair<int, float>> top_neurons(const string& layer,
                                          int token_pos, int top_n = 10) {
        Tensor act = activations[layer][0];  // (seq, d_ff)
        vector<pair<int, float>> neurons;
        for (int i = 0; i < act.shape()[1]; i++)
            neurons.push_back({i, fabsf(act(token_pos, i))});
        sort(neurons.rbegin(), neurons.rend(),
             [](auto& a, auto& b) { return a.second < b.second; });
        neurons.resize(top_n);
        return neurons;
    }

    // Per MoE: quale esperto è stato scelto per ogni token
    void log_expert_routing(int layer, int token_pos,
                             const vector<int>& experts,
                             const vector<float>& weights) {
        printf("Layer %d, Token %d → Expert %d (%.2f), Expert %d (%.2f)\n",
               layer, token_pos, experts[0], weights[0],
               experts[1], weights[1]);
    }
};
```

### Logit Lens

Proietta le rappresentazioni intermedie direttamente sul vocabolario per vedere cosa il modello "pensa" ad ogni layer:

```cpp
vector<string> logit_lens(const Model& model, const Tensor& hidden,
                           int layer, int token_pos, int top_k = 5) {
    // Prendi l'hidden state del layer intermedio
    Tensor h = hidden;  // (d_model,)
    // Applica LayerNorm finale + linear head
    Tensor logits = matmul(model.lm_head, model.final_norm(h));
    // Top-K token predetti a questo layer
    auto [values, indices] = topk(logits, top_k);
    vector<string> predictions;
    for (int idx : indices)
        predictions.push_back(tokenizer.decode({idx}));
    return predictions;
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/interpret/attention_capture.h/.cpp` | Cattura attention weights |
| `src/interpret/activation_probe.h/.cpp` | Probe neuroni, expert routing |
| `src/interpret/logit_lens.h/.cpp` | Proiezione intermedia → vocabolario |
| `src/interpret/export.h/.cpp` | Esportazione JSON, CSV, heatmap |
| `tests/test_interpret.cpp` | Shape corretto, export funzionante |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_attn_capture` | Cattura shape (n_layers, n_heads, seq, seq) |
| `test_attn_sum` | Somma righe attention ≈ 1.0 |
| `test_top_neurons` | Top neuroni sono quelli con attivazione più alta |
| `test_logit_lens` | Output è lista di token leggibili |
| `test_export_json` | JSON valido, importabile in visualizzatore |
