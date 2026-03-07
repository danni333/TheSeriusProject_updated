# Fase 36 — Native Sparse Attention (NSA, Dynamic Sparsity, Block-Sparse)

> I Transformer standard calcolano attention su **tutti** i token.
> Dynamic Sparsity salta i token irrilevanti; Block-Sparse raggruppa i calcoli.

---

## 1. Dynamic Sparsity

### Concetto

Per ogni query token, identifica al volo quali key token sono rilevanti e calcola attention solo su quelli.

```cpp
struct SparseAttentionResult {
    Tensor output;           // (seq, d_model)
    vector<vector<int>> active_keys;  // per ogni query: indici key attivi
};

Tensor dynamic_sparse_attention(const Tensor& Q, const Tensor& K,
                                  const Tensor& V, float threshold) {
    int seq = Q.shape()[0], d = Q.shape()[1];
    Tensor output({seq, d});
    
    for (int i = 0; i < seq; i++) {
        // 1. Calcola score approssimati velocemente (hash-based o top-k)
        vector<pair<int,float>> approx_scores;
        for (int j = 0; j <= i; j++) {  // causal
            float score = dot(Q[i], K[j]) / sqrtf(d);
            if (score > threshold)
                approx_scores.push_back({j, score});
        }
        // 2. Attention solo sui token selezionati
        Tensor sparse_scores({(int)approx_scores.size()});
        for (int k = 0; k < approx_scores.size(); k++)
            sparse_scores[k] = approx_scores[k].second;
        Tensor weights = softmax(sparse_scores);
        // 3. Weighted sum
        output[i].fill(0);
        for (int k = 0; k < approx_scores.size(); k++)
            output[i] += weights[k] * V[approx_scores[k].first];
    }
    return output;
}
```

---

## 2. Block-Sparse Attention

### Pattern di sparsità a blocchi

```
Full Attention (dense):     Block-Sparse (es. 4x4 blocks):
████████████████            ████░░░░░░░░████
████████████████            ████░░░░░░░░████
████████████████            ░░░░████░░░░░░░░
████████████████            ░░░░████░░░░░░░░
████████████████            ░░░░░░░░████░░░░
████████████████            ░░░░░░░░████░░░░
████████████████            ████░░░░░░░░████
████████████████            ████░░░░░░░░████

█ = calcolato   ░ = saltato (zero)
```

### CUDA Kernel Block-Sparse

```cpp
__global__ void block_sparse_attention(
    const float* Q, const float* K, const float* V, float* O,
    const int* block_mask,  // maschera: quali blocchi sono attivi
    int seq_len, int d, int block_size)
{
    int q_block = blockIdx.y;
    int k_block = blockIdx.x;
    
    // Controlla se questo blocco è attivo
    if (!block_mask[q_block * (seq_len/block_size) + k_block])
        return;  // Salta completamente
    
    // Carica Q e K block in shared memory
    // Calcola scores e output solo per questo blocco
    // ...
}
```

---

## 3. Struttura File

| File | Contenuto |
|---|---|
| `src/attention/sparse_attention.h/.cpp` | Dynamic sparsity (CPU) |
| `src/attention/block_sparse.cu` | Block-sparse CUDA kernel |
| `src/attention/sparsity_mask.h/.cpp` | Generazione pattern di sparsità |
| `tests/test_sparse.cpp` | Sparse vs dense: output ≈ uguale |

---

## 4. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_sparse_vs_dense` | Output simile (cosine sim > 0.99) con sparsity 50% |
| `test_block_pattern` | Solo blocchi attivi calcolati |
| `test_memory_saving` | Memoria usata proporzionale a sparsity ratio |
| `bench_sparse_speedup` | Speedup block-sparse vs dense per seq 4096+ |
