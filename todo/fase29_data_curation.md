# Fase 29 — Data Curation (DCLM-class Quality Filtering, Synthetic Data)

> Il motore più veloce è inutile con benzina sporca.
> Deduplicazione ad alte prestazioni e generazione dati sintetici.

---

## 1. Deduplicazione (MinHash + LSH)

### Problema

Dataset di TB contengono miliardi di righe duplicate o quasi-duplicate. Farlo in Python è troppo lento.

### MinHash: firma compatta di ogni documento

```cpp
class MinHasher {
    int num_hashes;
    vector<pair<uint64_t, uint64_t>> hash_params; // (a, b) per ogni hash

    vector<uint32_t> compute_signature(const string& text, int ngram=5) {
        auto shingles = extract_ngrams(text, ngram);
        vector<uint32_t> signature(num_hashes, UINT32_MAX);
        for (auto& shingle : shingles) {
            uint64_t h = hash64(shingle);
            for (int i = 0; i < num_hashes; i++) {
                uint32_t val = (hash_params[i].first * h + hash_params[i].second) >> 32;
                signature[i] = min(signature[i], val);
            }
        }
        return signature;
    }
};
```

### LSH: raggruppamento per similarità

```cpp
class LSHIndex {
    int bands, rows_per_band;  // bands × rows = num_hashes

    // Inserisci firma → bucket
    void insert(int doc_id, const vector<uint32_t>& sig) {
        for (int b = 0; b < bands; b++) {
            size_t band_hash = hash_band(sig, b * rows_per_band, rows_per_band);
            buckets[b][band_hash].push_back(doc_id);
        }
    }

    // Trova candidati duplicati
    set<pair<int,int>> find_candidates() {
        set<pair<int,int>> pairs;
        for (auto& band : buckets)
            for (auto& [hash, docs] : band)
                if (docs.size() > 1)
                    for (int i = 0; i < docs.size(); i++)
                        for (int j = i+1; j < docs.size(); j++)
                            pairs.insert({docs[i], docs[j]});
        return pairs;
    }
};
```

### Pipeline completa

```
1. Leggi TB di testo (multi-thread)
2. Per ogni documento: calcola MinHash signature
3. LSH: trova coppie candidate
4. Verifica: Jaccard similarity > threshold (0.8)
5. Deduplicazione: mantieni solo 1 copia
```

---

## 2. Synthetic Data Generation (Self-Instruction)

```
Modello v1.0 (stabile)
     │
     ├──▶ Genera 1000 istruzioni diverse
     │    "Scrivi una poesia su...", "Spiega come..."
     │
     ├──▶ Genera risposte per ogni istruzione
     │
     ├──▶ Filtra: qualità > soglia (reward model o euristica)
     │
     ▼
Dataset sintetico per training modello v2.0
```

```cpp
void generate_synthetic_dataset(Model& teacher, RewardModel& judge,
                                 const string& output_path, int n_samples) {
    ofstream out(output_path);
    int accepted = 0;
    while (accepted < n_samples) {
        string instruction = teacher.generate(
            "Generate a unique, creative instruction:", {.temp=1.2});
        string response = teacher.generate(instruction, {.temp=0.7});
        float quality = judge.score(instruction + response);
        if (quality > 0.7f) {
            out << json({{"instruction", instruction},
                         {"output", response}}).dump() << "\n";
            accepted++;
        }
    }
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/data/minhash.h/.cpp` | MinHash signature computation |
| `src/data/lsh.h/.cpp` | LSH index + candidate finding |
| `src/data/dedup.h/.cpp` | Pipeline deduplicazione end-to-end |
| `src/data/synthetic.h/.cpp` | Generazione dati sintetici |
| `tests/test_dedup.cpp` | Documenti duplicati trovati, unici preservati |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_minhash_similar` | Documenti simili → signature simili |
| `test_lsh_candidates` | Duplicati noti trovati come candidati |
| `test_dedup_preserves` | Documenti unici non rimossi |
| `test_synthetic_quality` | Dati generati passano il filtro qualità |
