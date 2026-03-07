# Fase 18 — Safety & Guardrails (Llama Guard, Jailbreak Defense, Content Filters)

> Un LLM senza controllo può allucinare o generare contenuti pericolosi.
> Filtri a livello di codice C++ sono più robusti di quelli via prompt.

---

## 1. Logit Bias / Masking

### Logit Bias

Modifica le probabilità di specifici token **prima** del sampling:

```cpp
void apply_logit_bias(Tensor& logits,
                       const map<int, float>& bias_map) {
    for (auto& [token_id, bias] : bias_map)
        logits[token_id] += bias;
    // bias = -100 → token quasi impossibile
    // bias = +5   → token fortemente favorito
}
```

### Banned Token Masking

Blocca completamente certi token o sequenze:

```cpp
void apply_token_mask(Tensor& logits,
                       const vector<int>& banned_tokens) {
    for (int id : banned_tokens)
        logits[id] = -INFINITY;  // softmax → 0
}
```

### Banned Sequence Detection

```cpp
class SequenceFilter {
    vector<vector<int>> banned_sequences;  // es. parolacce tokenizzate

    bool should_ban(const vector<int>& generated, int next_token) {
        for (auto& seq : banned_sequences) {
            int n = seq.size();
            if (next_token != seq.back()) continue;
            // Controlla se gli ultimi n-1 token + next_token = sequenza bannata
            if (generated.size() >= n - 1) {
                bool match = true;
                for (int i = 0; i < n - 1; i++)
                    if (generated[generated.size()-n+1+i] != seq[i])
                        { match = false; break; }
                if (match) return true;
            }
        }
        return false;
    }
};
```

---

## 2. Toxicity Classifier

### Architettura

Un modello lineare leggero che classifica l'output in tempo reale:

```
Embedding medio dell'output → Linear(d_model, 256) → ReLU
                             → Linear(256, 1) → Sigmoid → score ∈ [0,1]
```

### Integrazione nel loop di generazione

```cpp
void generate_safe(Model& model, ToxicityClassifier& guard,
                    vector<int>& output, float threshold = 0.8) {
    for (int step = 0; step < max_tokens; step++) {
        Tensor logits = model.forward(output, kv_cache);
        apply_token_mask(logits, banned_tokens);
        int token = sampler.sample(logits);
        output.push_back(token);

        // Check ogni N token
        if (step % 5 == 0) {
            float toxicity = guard.score(output);
            if (toxicity > threshold) {
                // Rollback ultimi token e rigenera
                output.resize(output.size() - 5);
                kv_cache.truncate(output.size());
                // Oppure: interrompi e rispondi con messaggio di sicurezza
                output = tokenizer.encode("[Mi dispiace, non posso rispondere.]");
                break;
            }
        }
    }
}
```

### Categorie di contenuto

| Categoria | Soglia | Azione |
|---|---|---|
| Violenza esplicita | 0.7 | Blocca e rispondi con messaggio safe |
| Contenuto sessuale | 0.8 | Blocca |
| Autolesionismo | 0.6 | Blocca + redirect a risorse di aiuto |
| Istruzioni pericolose | 0.7 | Blocca |
| Linguaggio offensivo | 0.9 | Avvisa ma non blocca |

---

## 3. Output Sanitization

```cpp
string sanitize_output(const string& text) {
    string result = text;
    // Rimuovi pattern pericolosi (es. codice eseguibile, SQL injection)
    for (auto& pattern : dangerous_patterns)
        result = regex_replace(result, pattern, "[REDACTED]");
    // Limita lunghezza risposta
    if (result.size() > MAX_RESPONSE_LENGTH)
        result = result.substr(0, MAX_RESPONSE_LENGTH) + "...";
    return result;
}
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/safety/logit_filter.h/.cpp` | Logit bias, token masking, sequence filter |
| `src/safety/toxicity.h/.cpp` | Toxicity classifier (modello leggero) |
| `src/safety/sanitizer.h/.cpp` | Output sanitization |
| `src/safety/config.h` | Soglie, banned lists, configurazione |
| `tests/test_safety.cpp` | Test: token bannati, toxicity detection |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_logit_bias` | Token con bias -100 → probabilità ≈ 0 |
| `test_banned_sequence` | Sequenza bannata mai generata |
| `test_toxicity_detect` | Input tossico noto → score > threshold |
| `test_safe_input` | Input normale → score < 0.3 |
| `test_rollback` | Dopo detection, output safe e KV cache consistente |
