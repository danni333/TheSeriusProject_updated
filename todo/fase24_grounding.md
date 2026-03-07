# Fase 24 — Anti-Hallucination Engine (Self-Consistency, NLI Verification, Citation)

> Il problema del "mentire con sicurezza".
> Self-Consistency e Citation Engine riducono le allucinazioni
> direttamente a livello di engine C++.

---

## 1. Self-Consistency

### Concetto

Genera **N risposte diverse** alla stessa domanda (con temperature alta) e scegli quella su cui il modello è più "d'accordo".

### Algoritmo

```
Input: prompt P, N = 4 risposte, temperature = 0.8

1. Genera N risposte R₁, R₂, R₃, R₄
2. Per ogni coppia (Rᵢ, Rⱼ): calcola similarità
3. Scegli la risposta con la media di similarità più alta
   → è la più "consistente" con le altre
```

### Implementazione

```cpp
string self_consistent_generate(Model& model, const string& prompt,
                                  int n_samples = 4, float temp = 0.8f) {
    vector<string> responses;
    vector<vector<float>> embeddings;

    // 1. Genera N risposte
    for (int i = 0; i < n_samples; i++) {
        SamplingParams p = {.temperature = temp, .top_p = 0.9f};
        string resp = model.generate(prompt, p);
        responses.push_back(resp);
        embeddings.push_back(compute_embedding(model, resp));
    }

    // 2. Matrice di similarità coseno
    vector<float> avg_sim(n_samples, 0);
    for (int i = 0; i < n_samples; i++)
        for (int j = 0; j < n_samples; j++)
            if (i != j)
                avg_sim[i] += cosine_similarity(embeddings[i], embeddings[j]);

    // 3. Scegli la più consistente
    int best = max_element(avg_sim.begin(), avg_sim.end()) - avg_sim.begin();
    return responses[best];
}
```

### Variante: Majority Voting (per risposte brevi/categoriche)

```cpp
string majority_vote(const vector<string>& responses) {
    map<string, int> counts;
    for (auto& r : responses) {
        string normalized = normalize(r);  // lowercase, trim, etc.
        counts[normalized]++;
    }
    return max_element(counts.begin(), counts.end(),
        [](auto& a, auto& b) { return a.second < b.second; })->first;
}
```

---

## 2. Citation Engine

### Concetto

Obbligare il modello a produrre **riferimenti ai documenti sorgente** trovati via RAG, integrati direttamente nella logica di generazione.

### Constraint durante la generazione

```cpp
void generate_with_citations(Model& model, const string& prompt,
                               const vector<Document>& rag_docs) {
    // Aggiungi istruzioni di citazione al system prompt
    string sys = "Cita le fonti usando [1], [2], ecc. "
                 "Ogni affermazione deve avere almeno una citazione.";

    // Logit bias per i token di citazione
    map<int, float> citation_bias;
    for (int i = 1; i <= rag_docs.size(); i++) {
        string cit = "[" + to_string(i) + "]";
        auto ids = tokenizer.encode(cit);
        for (int id : ids)
            citation_bias[id] = 2.0f;  // boost probabilità
    }

    // Genera con bias
    auto output = model.generate(prompt, {.logit_bias = citation_bias});

    // Verifica post-generazione
    auto citations = extract_citations(output);
    if (citations.empty())
        output += "\n\n[Nota: nessuna fonte trovata per verificare questa risposta]";
}
```

### Verifica delle citazioni

```cpp
struct CitationVerifier {
    vector<Document> sources;

    struct VerificationResult {
        string claim;              // affermazione del modello
        int cited_doc;             // documento citato
        float support_score;       // quanto il documento supporta il claim
        bool verified;
    };

    vector<VerificationResult> verify(const string& response) {
        auto claims = split_into_claims(response);
        vector<VerificationResult> results;

        for (auto& claim : claims) {
            auto cit = extract_citation(claim);
            if (cit.doc_id < 0 || cit.doc_id >= sources.size()) {
                results.push_back({claim.text, -1, 0, false});
                continue;
            }
            float score = compute_entailment(
                claim.text, sources[cit.doc_id].text);
            results.push_back({claim.text, cit.doc_id, score, score > 0.7f});
        }
        return results;
    }
};
```

---

## 3. Confidence Score

```cpp
float compute_confidence(Model& model, const string& prompt,
                          const string& response) {
    // Log-probability media dei token generati
    auto log_probs = model.compute_logprobs(prompt + response);
    float avg_lp = mean(log_probs);
    // Converti in probabilità [0, 1]
    return expf(avg_lp);
}
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/grounding/self_consistency.h/.cpp` | Multi-sample + voting/similarity |
| `src/grounding/citation_engine.h/.cpp` | Citazioni forzate + verifica |
| `src/grounding/confidence.h/.cpp` | Confidence score da log-probs |
| `tests/test_grounding.cpp` | Consistency, citations, confidence |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_self_consistency` | N risposte simili → scelta coerente |
| `test_majority_vote` | 3 "Parigi" + 1 "Roma" → output "Parigi" |
| `test_citation_present` | Con RAG, output contiene [1], [2] etc. |
| `test_citation_valid` | Citazione punta a documento esistente |
| `test_confidence` | Risposta certa → score > 0.8, incerta → < 0.5 |
