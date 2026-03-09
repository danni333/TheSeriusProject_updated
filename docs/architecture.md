# Architecture Overview

Questa pagina descrive ad alto livello l'architettura interna di **TheSeriusProject**.

---

## 1. Visione generale

L'obiettivo del progetto è implementare un motore LLM locale in C++ con:

- pipeline matematica controllabile,
- backend multipli (CPU, SIMD, CUDA),
- stack Transformer completo e modulare,
- supporto alla formazione (training) e all'inferenza.

La struttura dei sorgenti è pensata per essere leggibile e didattica, non solo performante.

---

## 2. Struttura delle directory principali

- `src/autograd/`
  - `engine.*`: gestione del grafo computazionale e della propagazione del gradiente.
  - `ops.*`: definizione delle operazioni di alto livello registrate nell'engine.

- `src/blas/`
  - `matmul_scalar.cpp`, `matmul_avx2.cpp`, `matmul_avx512.cpp`: varianti CPU di `matmul`.
  - `matmul_cuda.*`: implementazioni CUDA per la moltiplicazione di matrici.
  - `dispatch.cpp`: logica di selezione del backend in base alle capacità della macchina.

- `src/tensor/`
  - `tensor.*`: implementazione del tipo `Tensor`, gestione shape/stride, operazioni base.

- `src/memory/`
  - `pool_allocator.*`: gestione di un pool di memoria per ridurre le frammentazioni e le allocazioni frequenti.

- `src/transformer/`
  - `attention.*`, `ffn.*`, `layernorm.*`, `kv_cache.*`, `block.*`, `model.*`, `positional.*`:
    implementano i building block di un Transformer moderno, dal singolo blocco fino al modello completo.

- `src/train/`
  - `trainer.*`: ciclo di training, scheduling, logging.
  - `checkpoint.*`: salvataggio e ripristino dello stato del modello e dell'ottimizzatore.
  - `optim/adamw.*`: implementazione dell'ottimizzatore AdamW.

- `tests/`
  - Test unitari e benchmark per i vari moduli (autograd, tensor, transformer, CUDA, quantizzazione, ecc.).

---

## 3. Flow di esecuzione tipico (alto livello)

1. **Definizione del modello**  
   Le componenti `transformer/*` definiscono il modello (stack di blocchi, embedding, output head, ecc.).

2. **Inizializzazione dei tensori**  
   I tensori sono allocati tramite il `pool_allocator` (dove possibile) per ottimizzare l'uso della memoria.

3. **Esecuzione forward**  
   Il modello usa il layer di astrazione BLAS (`src/blas`) per eseguire matmul e altre operazioni in modo efficiente su CPU/GPU.

4. **(Opzionale) Backward & training**  
   L'engine di autograd costruisce il grafo computazionale durante il forward e calcola i gradienti in backward.
   L'ottimizzatore (es. AdamW) aggiorna i pesi usando i gradienti calcolati.

5. **Checkpointing**  
   Lo stato del modello e dell'allenamento viene salvato tramite il modulo `checkpoint`.

---

## 4. Backend e ottimizzazioni

### 4.1 CPU

- Implementazioni scalar e SIMD (AVX/AVX2/AVX‑512) di `matmul` e altre primitive.
- Selezione dinamica del percorso migliore in base alle capacità della CPU rilevate all'avvio.

### 4.2 GPU (CUDA)

- Kernel CUDA dedicati per operazioni pesanti (matmul, kernel fused, ecc.) in `src/blas/matmul_cuda.*` e `src/hal/fused_kernels.cu`.
- Meccanismi per minimizzare i trasferimenti host↔device dove possibile.

---

## 5. Testing & Benchmark

La cartella `tests/` include:

- **Test unitari** per blocchi logici (autograd, tensor, transformer, training).
- **Benchmark** focalizzati (matmul, memoria, mixed precision, kv cache, ecc.).

I risultati dei benchmark vengono spesso salvati in file `*_bench.txt` nella root del progetto per confronti successivi.

---

## 6. Estensioni future

La roadmap nel folder `todo/` prevede:

- Quantizzazione avanzata.
- Distribuzione multi‑GPU / multi‑node.
- Interfacce CLI e API compatibili con la specifica OpenAI.
- Sistema di plugin, strumenti di benchmarking e profiling avanzato.

Questa pagina verrà aggiornata man mano che l'architettura evolve e si stabilizza.

