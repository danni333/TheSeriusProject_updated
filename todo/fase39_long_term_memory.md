# Fase 39 — Long-term Memory (MemoryLayers, ∞-former, Persistent KV)

> Gli LLM dimenticano tutto a fine sessione.
> DNC aggiunge una memoria esterna leggibile/scrivibile;
> State Compression permette di ricordare conversazioni passate.

---

## 1. Differentiable Neural Computers (DNC)

### Architettura

```
Input xₜ
  │
  ▼
Controller (Transformer/LSTM)
  │
  ├──▶ Read heads  ──▶ Legge dalla memoria esterna
  ├──▶ Write heads ──▶ Scrive nella memoria esterna
  │
  ▼
Output yₜ = f(controller_out, read_vectors)

Memoria esterna M: (N_slots × d_memory)
  = array persistente di vettori
```

### Operazioni di memoria

```cpp
struct DNCMemory {
    Tensor M;          // (N, d) - matrice di memoria
    Tensor usage;      // (N,) - quanto è usato ogni slot
    Tensor link;       // (N, N) - ordine temporale di scrittura
    Tensor precedence; // (N,) - precedenza di scrittura

    // Scrittura: content-based + allocation-based
    void write(const Tensor& write_key, const Tensor& write_vector,
               float write_strength, float free_gate) {
        // 1. Content addressing: trova slot più simile
        Tensor content_w = softmax(cosine_similarity(write_key, M) * write_strength);
        // 2. Allocation: trova slot meno usato
        Tensor alloc_w = allocation_weighting(usage);
        // 3. Combina
        Tensor w = gate * content_w + (1-gate) * alloc_w;
        // 4. Scrivi
        M += outer(w, write_vector);  // aggiorna memoria
        update_usage(w, free_gate);
    }

    // Lettura: content-based + temporal
    Tensor read(const Tensor& read_key, float read_strength) {
        Tensor w = softmax(cosine_similarity(read_key, M) * read_strength);
        return matmul(w, M);  // vettore letto
    }
};
```

---

## 2. State Compression

### Concetto

Comprimi vecchi contesti in "vettori di riassunto" per ricordare conversazioni passate.

```
Conversazione 1 (3 mesi fa): 2000 token → compresso in 32 vettori
Conversazione 2 (1 mese fa):  1500 token → compresso in 32 vettori
Conversazione 3 (oggi):       500 token → pieno (non compresso)

Contesto totale: 32 + 32 + 500 = 564 token (vs. 4000 originali)
```

```cpp
class StateCompressor {
    Model& model;
    int summary_tokens;  // es. 32

    Tensor compress(const vector<int>& conversation) {
        // Forward pass attraverso il modello
        Tensor hidden = model.encode(conversation);  // (seq, d)
        // Pool in summary_tokens vettori usando attention
        Tensor queries = learned_summary_queries;  // (32, d) parametri appresi
        Tensor compressed = cross_attention(queries, hidden, hidden);
        return compressed;  // (32, d)
    }

    // Memoria persistente su disco
    void save_memory(const string& user_id, const Tensor& state) {
        string path = format("memory/{}/session_{}.bin", user_id, timestamp());
        state.save(path);
    }

    // Ricostruisci contesto da sessioni passate
    Tensor build_context(const string& user_id) {
        auto sessions = list_sessions(user_id);
        vector<Tensor> memories;
        for (auto& s : sessions)
            memories.push_back(Tensor::load(s.path));
        return concat(memories, /*dim=*/0);
    }
};
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/memory/dnc.h/.cpp` | DNC: read/write heads, memory matrix |
| `src/memory/compressor.h/.cpp` | State compression con cross-attention |
| `src/memory/persistent.h/.cpp` | Storage su disco per sessioni passate |
| `tests/test_dnc.cpp` | Write → read: contenuto recuperato |
| `tests/test_compression.cpp` | 2000 token → 32 vettori → info preservata |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_dnc_write_read` | Scrivi un vettore, rileggilo: cosine sim > 0.95 |
| `test_dnc_capacity` | N slot: N scritture diverse recuperabili |
| `test_compression` | Domanda su contesto compresso → risposta corretta |
| `test_persistence` | Save → restart → load: memoria intatta |
