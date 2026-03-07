# Fase 11 — FlashAttention-3 e RingAttention

> L'Attention standard cresce come $O(n^2)$ in memoria.
> Flash Attention riduce drasticamente gli accessi alla memoria lenta (HBM)
> calcolando l'Attention a blocchi direttamente nella SRAM veloce della GPU.

---

## 1. Il Problema: Memory-Bound Attention

### Attention standard

```
Q, K, V: shape (N, d)        N = lunghezza sequenza

1. S = Q · Kᵀ                 shape (N, N)  ← O(N²) memoria!
2. P = softmax(S)             shape (N, N)
3. O = P · V                  shape (N, d)
```

Per N = 8192, d = 128: la matrice S occupa **256 MB** in FP32. Con batch e multi-head, la VRAM esplode.

### Il bottleneck reale

Non è il calcolo (FLOPS), ma il **trasferimento dati** tra:

- **HBM** (VRAM, ~3 TB/s su H100): dove risiedono Q, K, V
- **SRAM** (on-chip, ~30+ TB/s, ma molto limitata): registri e shared memory

```
Attention standard: HBM → SRAM → HBM → SRAM → HBM  (3 passaggi)
FlashAttention:     HBM → SRAM → HBM                 (1 passaggio)
```

### Oltre FlashAttention-2: TMA e WGMMA (FlashAttention-3)

Sui Tensor Core Hopper (H100+), FlashAttention-3 sovrappone I/O e computation usando il **Tensor Memory Accelerator (TMA)** e le istruzioni **Warp Group MMA (WGMMA)**. Il fetch dei dati in HBM avviene in hardware asincrono mentre la compute lavora sul blocco precedente (Ping-Pong bufering), spingendo l'utilizzazione al 75-80% del massimo hardware.

---

## 2. Flash Attention: Tiling Algorithm

### Idea chiave

Non materializzare mai la matrice NxN completa. Calcola softmax e output **a blocchi** usando l'**online softmax trick**.

### Pseudocodice

```
Dividi Q in blocchi Q₁, Q₂, ... di dimensione Br × d
Dividi K, V in blocchi K₁, K₂, ... di dimensione Bc × d

Per ogni blocco Qᵢ:                              // loop esterno
    Carica Qᵢ in SRAM
    Inizializza Oᵢ = 0, mᵢ = -inf, lᵢ = 0       // accumulatori

    Per ogni blocco Kⱼ, Vⱼ:                       // loop interno
        Carica Kⱼ, Vⱼ in SRAM
        Sᵢⱼ = Qᵢ · Kⱼᵀ                           // (Br × Bc) in SRAM
        m_new = max(mᵢ, rowmax(Sᵢⱼ))
        P̃ᵢⱼ = exp(Sᵢⱼ - m_new)
        l_new = exp(mᵢ - m_new) · lᵢ + rowsum(P̃ᵢⱼ)
        Oᵢ = exp(mᵢ - m_new) · Oᵢ + P̃ᵢⱼ · Vⱼ     // aggiorna output
        mᵢ = m_new, lᵢ = l_new

    Oᵢ = Oᵢ / lᵢ                                 // normalizza
    Scrivi Oᵢ in HBM
```

### Online Softmax Trick

Il trucco è calcolare softmax **incrementalmente** senza conoscere tutti i valori in anticipo:

```
Stato: (m, l, O)
  m = max visto finora
  l = somma exp vista finora
  O = output pesato non normalizzato

Aggiornamento con nuovo blocco di scores S:
  m_new = max(m, max(S))
  correction = exp(m - m_new)
  l = correction * l + sum(exp(S - m_new))
  O = correction * O + exp(S - m_new) · V
```

---

## 3. CUDA Kernel Flash Attention

### Struttura del kernel

```cpp
__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Br, int Bc)
{
    // Ogni thread block processa un blocco Qᵢ
    int q_block = blockIdx.x;

    // Shared memory per i blocchi
    extern __shared__ float smem[];
    float* sQ  = smem;                    // Br × d
    float* sK  = smem + Br * d;           // Bc × d
    float* sV  = sK + Bc * d;             // Bc × d
    float* sS  = sV + Bc * d;             // Br × Bc

    // Carica Qᵢ in shared memory
    load_block(Q, sQ, q_block * Br, Br, d);

    // Accumulatori in registri
    float m_i = -INFINITY, l_i = 0;
    float O_acc[Br_per_thread][d];  // output parziale

    // Loop sui blocchi K, V
    for (int j = 0; j < N; j += Bc) {
        load_block(K, sK, j, Bc, d);
        load_block(V, sV, j, Bc, d);
        __syncthreads();

        // S = Q · Kᵀ in shared memory
        compute_scores(sQ, sK, sS, Br, Bc, d);

        // Online softmax update
        update_online_softmax(sS, sV, O_acc, &m_i, &l_i, Br, Bc, d);
        __syncthreads();
    }

    // Normalizza e scrivi in HBM
    normalize_and_store(O, O_acc, l_i, q_block * Br, Br, d);
}
```

### Dimensioni ottimali dei blocchi

| GPU | Architettura | Feature chiavi | Br × Bc consigliato |
|---|---|---|---|
| RTX 3090/A100 | Ampere | FlashAttn-2 (cp.async) | 128 × 64 |
| H100 / RTX 4090 | Hopper/Ada | FlashAttn-3 (TMA, WGMMA) | 128 × 128 |

---

## 3.5. Ring Attention

Flash Attention risolve la memoria rispetto ai pesi intermedi O(N²), ma in sequenze di 1M+ token (long-context), *distribuire* la sequenza su multiple GPU diventa critico.
Con **RingAttention**, i blocchi K e V ruotano attraverso un anello di GPU (via NVLink/NCCL), permettendo a ciascuna GPU di processare un chunk di Q sull'intero contesto asincronamente.

---

## 4. Memory Paging

### Concetto

Per sequenze lunghissime (es. 128K token), gestisci la memoria come un **sistema operativo** con pagine.

### Paged KV Cache

```
┌──────────────────────────────────────┐
│           Page Table                 │
│  Seq 0: [Page 3] [Page 7] [Page 1]  │
│  Seq 1: [Page 0] [Page 5] [Page 2]  │
│  Seq 2: [Page 4] [Page 6]    -      │
└──────────────────────────────────────┘

Physical Pages (ognuna contiene BLOCK_SIZE token):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │
└────┴────┴────┴────┴────┴────┴────┴────┘
```

### Vantaggi

- **Nessuna frammentazione**: pagine allocate in un pool contiguo
- **Allocation on-demand**: alloca solo le pagine necessarie
- **Condivisione**: sequenze con prefisso comune condividono le pagine (copy-on-write)

---

## 5. Backward Pass (Flash Attention)

Per il training serve anche il backward. Flash Attention salva solo i softmax statistics (`m`, `l`) e **ricalcola** S e P nel backward:

```
Backward:
  Ricarica Q, K, V a blocchi (come forward)
  Ricalcola S = Q · Kᵀ, P = softmax(S) al volo
  dV = Pᵀ · dO
  dP = dO · Vᵀ
  dS = P ⊙ (dP - rowsum(dP ⊙ P))
  dQ = dS · K
  dK = dSᵀ · Q
```

> [!IMPORTANT]
> Flash Attention usa **più FLOPS** dell'attention standard (ricalcola nel backward),
> ma è **più veloce** perché fa molti meno accessi a HBM.
> Il rapporto FLOPS/IO ne fa un trade-off vincente.

---

## 6. Struttura File

| File | Contenuto |
|---|---|
| `src/attention/flash_attn.h` | Interfaccia Flash Attention |
| `src/attention/flash_attn_fwd.cu` | Kernel CUDA forward |
| `src/attention/flash_attn_bwd.cu` | Kernel CUDA backward |
| `src/attention/flash_attn_cpu.cpp` | Versione CPU (tiled, senza GPU) |
| `src/memory/paged_kv_cache.h/.cpp` | KV Cache con paging |
| `tests/test_flash_attn.cpp` | Correttezza: Flash vs Naive attention |

---

## 7. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_flash_vs_naive` | Output identico (atol 1e-3) a attention standard |
| `test_flash_backward` | Gradienti dQ, dK, dV vs numerico |
| `test_flash_causal` | Causal mask rispettata nel tiling |
| `test_paged_cache` | Append, lookup, page allocation corretti |
| `test_long_sequence` | N=8192: Flash OK, naive OOM |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_flash_vs_naive` | Speedup e memoria: N = 512, 1024, 2048, 4096, 8192 |
| `bench_io_reads` | Conteggio accessi HBM: Flash vs standard |
