# Fase 46 — Semantic Hash Collision (L'Archiviazione Totale)

> Ogni domanda è una coordinata in una funzione matematica universale.
> Invece di cercare nei database, la conoscenza **è** la funzione.

---

## 1. Concetto

### Da retrieval a function evaluation

```
Approccio tradizionale (RAG):
  Domanda → embedding → ricerca nel DB → documenti → risposta

Semantic Hash:
  Domanda → hash semantico → lookup diretto → risposta
  (O(1) invece di O(log n))
```

### L'idea
Tutta la conoscenza viene compressa in una **tabella hash semantica**: una funzione che mappa embedding → informazione, senza bisogno di scansionare documenti.

---

## 2. Semantic Locality-Sensitive Hashing

```cpp
class SemanticHash {
    // Hash che preserva la similarità semantica
    vector<Tensor> hyperplanes;  // N piani casuali nello spazio embedding
    int hash_bits;

    uint64_t hash(const Tensor& embedding) {
        uint64_t h = 0;
        for (int i = 0; i < hash_bits; i++) {
            float dot = dot_product(embedding, hyperplanes[i]);
            if (dot > 0) h |= (1ULL << i);
        }
        return h;
    }

    // Domande "vicine" semanticamente → stesso hash
    // "Capitale della Francia" ≈ "Qual è la capitale francese" → hash uguale
};
```

---

## 3. Knowledge Function

```cpp
class KnowledgeFunction {
    // La "funzione universale": hash → informazione compressa
    unordered_map<uint64_t, CompressedKnowledge> table;

    struct CompressedKnowledge {
        Tensor latent;        // rappresentazione compressa dell'informazione
        float confidence;     // quanto è affidabile
        vector<string> sources;
    };

    // Costruzione: comprime l'intero corpus
    void build(const vector<Document>& corpus, Model& encoder) {
        for (auto& doc : corpus) {
            // Scomponi in fatti atomici
            auto facts = extract_facts(doc);
            for (auto& fact : facts) {
                Tensor emb = encoder.encode(fact.text);
                uint64_t h = hasher.hash(emb);
                if (table.count(h)) {
                    // Merge: combina conoscenza esistente
                    table[h].latent = merge(table[h].latent, encoder.encode(fact.text));
                    table[h].sources.push_back(doc.source);
                } else {
                    table[h] = {encoder.encode(fact.text), 1.0f, {doc.source}};
                }
            }
        }
    }

    // Query: O(1)
    string query(const string& question, Model& encoder, Model& decoder) {
        Tensor q_emb = encoder.encode(question);
        uint64_t h = hasher.hash(q_emb);
        if (!table.count(h))
            return "[Nessuna conoscenza trovata]";
        auto& knowledge = table[h];
        // Decodifica la conoscenza compressa in risposta
        return decoder.generate_from_latent(knowledge.latent, question);
    }
};
```

---

## 4. Multi-Resolution Hashing

```
Hash bits: 8  → 256 bucket     → bassa risoluzione (topic-level)
Hash bits: 16 → 65K bucket     → media risoluzione (concetto)
Hash bits: 32 → 4B bucket      → alta risoluzione (fatto specifico)

Query: cerca prima a 32 bit, poi fallback a 16, poi a 8
```

---

## 5. Struttura File

| File | Contenuto |
|---|---|
| `src/knowledge/semantic_hash.h/.cpp` | Locality-Sensitive Hashing semantico |
| `src/knowledge/knowledge_fn.h/.cpp` | Knowledge Function (build + query) |
| `src/knowledge/fact_extractor.h/.cpp` | Estrazione fatti atomici da documenti |
| `tests/test_knowledge.cpp` | Build, query, collision resolution |

---

## 6. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_semantic_hash` | Domande simili → stesso hash |
| `test_hash_collision` | Domande diverse → hash diversi (p > 0.95) |
| `test_build_corpus` | 1000 documenti indicizzati correttamente |
| `test_query_accuracy` | Risposta corretta per domande fattuali note |
| `test_multi_resolution` | Fallback da high-res a low-res funziona |

