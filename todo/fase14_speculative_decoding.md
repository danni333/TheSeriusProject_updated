# Fase 14 — Speculative Decoding (EAGLE-3, Medusa, Draft+Verify) (Velocità di Generazione Estrema)

> Gli LLM generano un token alla volta (autoregressive).
> Speculative Decoding rompe questo limite usando un modello piccolo
> per "indovinare" più token, verificati dal modello grande in parallelo.

---

## 1. Concetto

```
Senza Speculative Decoding:
  Big model: [tok1] → [tok2] → [tok3] → [tok4] → [tok5]
  Tempo:       1x      1x      1x      1x      1x  = 5 step

Con Speculative Decoding:
  Draft model:  [tok1, tok2, tok3, tok4, tok5]  → 5 token veloci
  Big model:    [verifica tutti e 5 in 1 forward pass]
  Accettati:    [tok1 ✓, tok2 ✓, tok3 ✓, tok4 ✗]  → 3 token in 1 step!
  Tempo:        2 step (1 draft + 1 verify) per 3+ token
```

---

## 2. Draft Model

### Caratteristiche

| Proprietà | Draft | Target |
|---|---|---|
| Parametri | 100M-500M | 7B-70B |
| Layer | 4-6 | 32-80 |
| Speed | ~50 tok/s | ~5 tok/s |
| Qualità | Mediocre | Alta |
| Vocabolario | **Stesso** del target | — |

> [!IMPORTANT]
> I due modelli **devono** condividere lo stesso tokenizer/vocabolario.
> Altrimenti non puoi confrontare le probabilità token per token.

### Opzioni per il draft model

1. **Modello separato** (es. GPT-125M per verificare GPT-7B)
2. **Early exit**: usa solo i primi N layer del modello grande
3. **Medusa heads**: teste aggiuntive leggere sul modello grande

---

## 3. Algoritmo di Speculative Decoding

### Step-by-step

```
Input: prompt, draft_model, target_model, gamma (num token da speculare)

Ripeti fino a max_tokens:
    1. DRAFT: genera gamma token con draft_model
       draft_tokens = [d₁, d₂, ..., dγ]
       draft_probs  = [p_d(d₁), p_d(d₂), ..., p_d(dγ)]

    2. VERIFY: forward pass del target_model su [prompt + draft_tokens]
       target_probs = [p_t(d₁), p_t(d₂), ..., p_t(dγ), p_t(next)]
       ← Il target model processa TUTTI i token in parallelo (1 forward)

    3. ACCEPT/REJECT: per ogni token i = 1..γ:
       r = random(0, 1)
       if r < p_t(dᵢ) / p_d(dᵢ):
           ACCETTA dᵢ
       else:
           RIFIUTA dᵢ
           Campiona un nuovo token dalla distribuzione corretta:
             p_corrected = normalize(max(0, p_t - p_d))
           Scarta tutti i token dopo i
           BREAK

    4. Se tutti γ accettati:
       Aggiungi anche un token bonus campionato da p_t(next)
```

### Perché funziona

- Se draft e target concordano (alta probabilità), si accettano molti token → speedup
- Se draft sbaglia, il target corregge → **la qualità è identica** al target puro
- È **matematicamente provato** che la distribuzione di output è esattamente quella del target model

---

## 4. Implementazione C++

```cpp
struct SpeculativeDecoder {
    Model& draft_model;
    Model& target_model;
    Sampler& sampler;
    int gamma;  // tipicamente 4-8

    vector<int> generate(vector<int> prompt, int max_tokens) {
        vector<int> output;
        KVCache draft_cache, target_cache;

        while (output.size() < max_tokens) {
            // 1. Draft: genera gamma token
            vector<int> draft_tokens;
            vector<Tensor> draft_probs_list;
            auto draft_ids = concat(prompt, output);

            for (int i = 0; i < gamma; i++) {
                Tensor logits = draft_model.forward(draft_ids, draft_cache);
                Tensor probs = softmax(logits / temperature);
                int token = categorical_sample(probs);
                draft_tokens.push_back(token);
                draft_probs_list.push_back(probs);
                draft_ids.push_back(token);
            }

            // 2. Verify: un singolo forward del target
            auto verify_ids = concat(prompt, output, draft_tokens);
            Tensor all_logits = target_model.forward_batch(
                verify_ids, target_cache);
            // all_logits: shape (gamma + 1, vocab_size)

            // 3. Accept/Reject
            int accepted = 0;
            for (int i = 0; i < gamma; i++) {
                Tensor target_probs = softmax(all_logits[i] / temperature);
                float p_t = target_probs[draft_tokens[i]];
                float p_d = draft_probs_list[i][draft_tokens[i]];

                float r = random_uniform(0, 1);
                if (r < fmin(1.0f, p_t / p_d)) {
                    output.push_back(draft_tokens[i]);
                    accepted++;
                } else {
                    // Campiona dalla distribuzione corretta
                    Tensor corrected = max(0, target_probs - draft_probs_list[i]);
                    corrected = normalize(corrected);
                    int corrected_token = categorical_sample(corrected);
                    output.push_back(corrected_token);
                    break;
                }
            }

            // 4. Se tutti accettati, bonus token
            if (accepted == gamma) {
                Tensor bonus_probs = softmax(all_logits[gamma] / temperature);
                int bonus = categorical_sample(bonus_probs);
                output.push_back(bonus);
            }

            // Aggiorna le KV cache (rollback per i token rifiutati)
            draft_cache.truncate(prompt.size() + output.size());
            target_cache.truncate(prompt.size() + output.size());

            if (output.back() == EOS_TOKEN) break;
        }
        return output;
    }
};
```

---

## 5. Gestione Memoria per Due Modelli

```
RAM/VRAM Budget:
┌────────────────────────────────────────┐
│  Target Model (7B × 2 bytes = 14 GB)  │  ← FP16/INT4
├────────────────────────────────────────┤
│  Draft Model  (125M × 2 = 250 MB)     │  ← trascurabile
├────────────────────────────────────────┤
│  KV Cache Target (2 × 32 × seq × d)   │
├────────────────────────────────────────┤
│  KV Cache Draft  (2 × 4 × seq × d)    │  ← piccola
├────────────────────────────────────────┤
│  Workspace tensors                     │
└────────────────────────────────────────┘
```

### Strategie di ottimizzazione

| Strategia | Dettaglio |
|---|---|
| **Draft su CPU** | Se la GPU è piena, il draft model gira su CPU (è piccolo) |
| **Shared embeddings** | Draft e target condividono la embedding table |
| **Async prefetch** | Mentre il target verifica, il draft prepara il prossimo batch |

---

## 6. Tuning del parametro γ (gamma)

| γ | Acceptance rate tipico | Speedup |
|---|---|---|
| 1 | ~90% | 1.5x |
| 4 | ~70% | 2.0-2.5x |
| 8 | ~50% | 2.5-3.0x |
| 16 | ~30% | diminuisce (overhead) |

> Il γ ottimale dipende dalla qualità del draft model.
> Più il draft è buono, più alto puoi andare.

---

## 7. Structured Output / JSON Mode (Constrained Decoding)

> Il modello non genera solo testo libero: con il JSON Mode forza l'output
> a rispettare uno schema JSON definito dall'utente. Usa grammar-guided
> sampling per garantire validità sintattica al 100%.

### Concetto

```text
Senza JSON Mode:
  Prompt: "Estrai nome e età" → "Il nome è Marco e ha 30 anni"  (testo libero)

Con JSON Mode:
  Prompt: "Estrai nome e età" → {"name": "Marco", "age": 30}    (JSON valido sempre!)
```

### Implementazione (Grammar-Guided Sampling)

```cpp
namespace serius::inference {

struct JsonSchema {
    string type;         // "object", "array", "string", "number"
    map<string, JsonSchema> properties;
    vector<string> required;
    vector<string> enum_values;
};

class ConstrainedDecoder {
    // Finite State Machine che traccia dove siamo nel JSON
    enum State { OBJECT_START, KEY, COLON, VALUE, COMMA_OR_END, ARRAY, STRING, NUMBER };
    State state_ = OBJECT_START;
    JsonSchema schema_;

public:
    // Maschera i token non validi dato lo stato corrente
    void apply_grammar_mask(Tensor& logits, const Tokenizer& tok) {
        // Se siamo dopo '{' → solo token che iniziano con '"' (chiavi)
        // Se siamo dopo ':' → token validi per il tipo del campo
        // Se siamo alla fine di un valore → solo ',' o '}'
        for (int i = 0; i < logits.size(); i++) {
            string token_str = tok.decode({i});
            if (!is_valid_next(token_str))
                logits[i] = -INFINITY;  // blocca token non validi
        }
    }
};

} // namespace serius::inference
```

### API Endpoint (compatibile OpenAI)

```json
// POST /v1/chat/completions
{
    "model": "default",
    "messages": [{"role": "user", "content": "Estrai nome e età da: Marco ha 30 anni"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }
}
```

### File aggiuntivi

| File | Contenuto |
| --- | --- |
| `src/inference/constrained_decoder.h/.cpp` | Grammar-guided sampling |
| `src/inference/json_schema.h/.cpp` | Parser e validatore JSON Schema |
| `tests/test_json_mode.cpp` | Output sempre JSON valido per vari schema |

---

## 8. Struttura File

| File | Contenuto |
|---|---|
| `src/inference/speculative.h/.cpp` | SpeculativeDecoder: draft + verify + accept/reject |
| `src/inference/draft_model.h/.cpp` | Wrapper per il draft model |
| `src/inference/acceptance.h/.cpp` | Logica di accettazione/rigetto + distribuzione corretta |
| `tests/test_speculative.cpp` | Test correttezza: output = distribuzione del target |

---

## 8. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_spec_correctness` | Distribuzione output = target puro (KL divergence ≈ 0) |
| `test_acceptance_rate` | Con draft = target: acceptance 100% |
| `test_rejection_resample` | Token rifiutato → nuovo token dalla distribuzione corretta |
| `test_kv_cache_rollback` | Dopo rigetto, KV cache troncata correttamente |
| `test_eos_handling` | EOS nel mezzo dei draft → generazione si ferma |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_spec_vs_vanilla` | Token/secondo: speculative vs autoregressive |
| `bench_gamma_sweep` | Speedup al variare di γ (1, 2, 4, 8, 16) |
| `bench_acceptance` | Tasso di accettazione per combinazione draft/target |
