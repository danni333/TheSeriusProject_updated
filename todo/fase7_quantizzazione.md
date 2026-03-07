# Fase 7 — Quantizzazione SOTA (AWQ, GPTQ, FP8, GGUF INT4/INT8)

> Un modello 7B a float32 occupa ~28 GB di VRAM.
> La quantizzazione riduce la precisione per risparmiare memoria
> e aumentare la velocità, con perdita minima di qualità.

---

## 1. FP8 (E4M3/E5M2) e Half Precision (FP16/BF16)

### Obiettivo

Passare da 32 bit a 16 bit **dimezza** l'uso della memoria con perdita di precisione quasi nulla.

### Formati a confronto

| Formato | Bits | Esponente | Mantissa | Range | Uso tipico |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | Training standard |
| FP16 | 16 | 5 | 10 | ±65504 | Inference, mixed precision |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | Training mixed precision |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | Forward pass, tensori |
| FP8 E5M2 | 8 | 5 | 2 | ±57344 | Backward pass, gradienti |

> [!TIP]
> **FP8** è il nuovo standard di de facto su architetture Ada/Hopper (L40, H100). Usare E4M3 per i pesi/attivazioni (più precisione) e E5M2 per i gradienti (più range).

### FP8 Block-wise Scaling vs Tensor-wise Scaling

Piuttosto che calcolare uno scale-factor per l'intero tensore, calcolarlo per **blocchi** (es. 128 elementi) o vettori riduce drasticamente l'errore di quantizzazione e gli outliers. FP8 con MX6 block scaling abilita throughput elevatissimi tramite i Tensor Core.

### Mixed Precision Training

```
Forward pass:   pesi in FP16/BF16 (veloce)
Loss:           calcolata in FP32 (precisa)
Backward pass:  gradienti in FP16/BF16
Optimizer step: master weights in FP32, aggiornati, poi cast a FP16
```

### Implementazione C++

```cpp
// Conversione FP32 ↔ FP16 (IEEE 754)
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);

// BF16: troncamento semplice dei 16 bit bassi
uint16_t float_to_bf16(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    return static_cast<uint16_t>(bits >> 16);
}
```

---

## 2. Quantizzazione INT4

### Concetto

Comprimere ogni peso da 32 bit a **4 bit** → riduzione di **8x** nella memoria.

### Schema di quantizzazione (per blocco)

```
Blocco di 32/64/128 float → {scale, zero_point, 4-bit values[]}

quantize(x):
    x_min = min(blocco)
    x_max = max(blocco)
    scale = (x_max - x_min) / 15        // 15 = 2⁴ - 1
    zero_point = round(-x_min / scale)
    q = round(x / scale + zero_point)    // clamp a [0, 15]
    return pack_4bit(q), scale, zero_point

dequantize(q, scale, zero_point):
    return (q - zero_point) * scale
```

### AWQ (Activation-Aware Weight Quantization, 2023)

L'idea chiave di AWQ è proteggere i pesi **salienti** — quelli che corrispondono ad attivazioni grandi. Invece di quantizzare tutti i pesi allo stesso modo, AWQ scala i pesi salienti prima della quantizzazione per ridurre l'errore:

```text
1. Raccogli statistiche delle attivazioni (su piccolo calibration set)
2. Identifica i canali con |attivazione| alta ("salient channels")
3. Scala quei canali di un fattore s > 1 PRIMA della quantizzazione
4. Risultato: errore di quantizzazione ridotto dove conta di più
```

**AWQ vs GPTQ:** AWQ è più veloce da quantizzare (no Hessian) e generalizza meglio (meno overfitting sul calibration set).

### GPTQ (Gradient Post-Training Quantization, 2022)

GPTQ usa informazioni sulla matrice Hessiana per minimizzare l'errore di quantizzazione colonna per colonna, calibrando su un piccolo dataset:

```text
Per ogni colonna j del peso W:
    1. Quantizza w_j a INT4
    2. Calcola l'errore di quantizzazione δ
    3. Compensa l'errore distribuendo δ alle colonne ancora non quantizzate
       usando l'inverso della Hessiana: W_{j+1:} -= δ · H⁻¹_{j,j+1:} / H⁻¹_{j,j}
```

### Packing a 4 bit

Due valori INT4 vengono impacchettati in un singolo byte:

```cpp
uint8_t pack_two(uint8_t a, uint8_t b) {
    return (a & 0x0F) | ((b & 0x0F) << 4);
}

void unpack_two(uint8_t packed, uint8_t& a, uint8_t& b) {
    a = packed & 0x0F;
    b = (packed >> 4) & 0x0F;
}
```

### De-quantization Kernel (on-the-fly)

```cpp
// CPU: decomprime e moltiplica al volo
void dequant_matmul(const uint8_t* W_q4, const float* scales,
                     const uint8_t* zeros, const float* X,
                     float* Y, int M, int N, int K, int block_size) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                int block_idx = k / block_size;
                float s = scales[j * (K/block_size) + block_idx];
                uint8_t z = zeros[j * (K/block_size) + block_idx];
                uint8_t packed = W_q4[(j * K + k) / 2];
                uint8_t q = (k % 2 == 0) ? (packed & 0xF) : (packed >> 4);
                float w = (q - z) * s;
                sum += X[i * K + k] * w;
            }
            Y[i * N + j] = sum;
        }
    }
}
```

> [!IMPORTANT]
> La de-quantizzazione avviene **nei registri**, non in RAM.
> Non si espande mai l'intero tensore a FP32: solo i valori necessari per il calcolo corrente.

---

## 3. KV Cache

### Il problema

Nella generazione autoregressiva, ad ogni nuovo token il modello deve ricalcolare l'Attention su **tutta la sequenza precedente**. Senza cache, il costo cresce come O(n²) ad ogni step.

### Soluzione

Memorizzare i vettori **Key** e **Value** già calcolati e riutilizzarli.

```
Step 1: input "Il"     → calcola K₁, V₁ → salva in cache
Step 2: input "gatto"  → calcola K₂, V₂ → cache = [K₁K₂, V₁V₂]
                          Q₂ attende a K₁K₂ (non ricalcola K₁)
Step 3: input "è"      → calcola K₃, V₃ → cache = [K₁K₂K₃, V₁V₂V₃]
```

### Struttura dati

```cpp
struct KVCache {
    // Per ogni layer, per ogni head
    // shape: (n_layers, n_heads, max_seq_len, d_k)
    Tensor keys;
    Tensor values;
    int current_len = 0;  // quanti token sono in cache

    void append(int layer, const Tensor& new_k, const Tensor& new_v) {
        // Copia new_k e new_v alla posizione current_len
        keys.slice(layer, current_len) = new_k;
        values.slice(layer, current_len) = new_v;
    }

    // Per beam search: duplica le cache dei beam selezionati
    void reorder(const vector<int>& beam_indices);
};
```

### Gestione memoria della KV Cache

| Strategia | Descrizione |
|---|---|
| **Pre-allocazione** | Alloca `max_seq_len` slot all'inizio, riempi man mano |
| **Rolling buffer** | Per sequenze lunghissime, sovrascrive i token più vecchi |
| **Paged KV Cache** | Alloca pagine di 256 token alla volta (come PagedAttention) |

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/quant/fp16.h/.cpp` | Conversioni FP32↔FP16, FP32↔BF16 |
| `src/quant/int4.h/.cpp` | Quantizzazione/de-quantizzazione INT4 a blocchi |
| `src/quant/dequant_matmul.cpp` | Matmul con de-quantizzazione on-the-fly (CPU) |
| `src/quant/dequant_matmul.cu` | Versione CUDA del kernel (GPU) |
| `src/inference/kv_cache.h/.cpp` | KV Cache con append, reorder, paging |
| `tests/test_quant.cpp` | Test di precisione: errore max FP16, INT4 vs FP32 |
| `tests/test_kv_cache.cpp` | Test: append, lunghezza, rolling buffer |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_fp16_roundtrip` | `float → half → float`: errore < 0.1% per valori tipici |
| `test_bf16_roundtrip` | `float → bf16 → float`: range preservato |
| `test_int4_accuracy` | Matmul INT4 vs FP32: errore relativo < 1% su pesi reali |
| `test_int4_packing` | Pack/unpack di valori 0-15 corretto |
| `test_kv_append` | Dopo 10 append, cache contiene tutte le K e V |
| `test_kv_reorder` | Beam reorder mantiene consistenza |
| `test_kv_perplexity` | Perplexity con/senza KV cache ≈ identica |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_int4_vs_fp32` | Speedup matmul INT4 vs FP32 |
| `bench_kv_cache` | Token/secondo con vs senza KV cache (seq 512, 1024, 2048) |
| `bench_memory` | VRAM usata: FP32 vs FP16 vs INT4 per modello 125M params |
