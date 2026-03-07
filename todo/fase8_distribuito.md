# Fase 8 — Calcolo Distribuito (NCCL/RCCL, ZeRO-3, FSDP2, Multi-Node)

> Quando un modello non sta nella VRAM di una singola GPU,
> bisogna far comunicare più GPU o più macchine tra loro.

---

## 1. Data Parallelism

### Concetto

Ogni GPU ha una **copia completa** del modello ma vede **dati diversi** (batch diversi). Alla fine di ogni step, i gradienti vengono **mediati** tra tutte le GPU.

### Flusso

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  GPU 0   │  │  GPU 1   │  │  GPU 2   │
│ Batch 0  │  │ Batch 1  │  │ Batch 2  │
│ Forward  │  │ Forward  │  │ Forward  │
│ Backward │  │ Backward │  │ Backward │
│ grad_0   │  │ grad_1   │  │ grad_2   │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └──────── AllReduce ────────┘
                   │
          avg_grad = (g0+g1+g2)/3
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
  Update W      Update W      Update W
  (identici su tutte le GPU)
```

### AllReduce

Operazione collettiva dove ogni GPU contribuisce un valore e riceve la somma/media.

```cpp
// Con NCCL
ncclAllReduce(grad_buffer, grad_buffer,
              num_elements, ncclFloat, ncclAvg,
              comm, stream);
```

> [!IMPORTANT]
> **Ring AllReduce**: i dati vengono passati ad anello tra GPU.
> Il tempo dipende dalla dimensione del tensore, non dal numero di GPU.
> Bandwidth-optimal: trasferisce `2(N-1)/N * data_size` in totale.

---

## 2. Model Parallelism

### 2a. Tensor Parallelism

Divide singoli layer **orizzontalmente** tra più GPU.

```
Linear Layer: Y = X · W

      W (d_model × d_ff)
┌──────────┬──────────┐
│  W_part0 │  W_part1 │    → GPU 0 calcola Y₀ = X · W₀
│ (d×d_ff/2)│(d×d_ff/2)│   → GPU 1 calcola Y₁ = X · W₁
└──────────┴──────────┘
                              → AllGather: Y = [Y₀, Y₁]
```

### 2b. Pipeline Parallelism

Divide il modello **verticalmente** per layer.

```
                  Timeline →
GPU 0 (L0-L5):   [F₀] [F₁] [F₂] [F₃]  ───────  [B₃] [B₂] [B₁] [B₀]
GPU 1 (L6-L11):  ──── [F₀] [F₁] [F₂] [F₃] [B₃] [B₂] [B₁] [B₀] ────

F = Forward micro-batch, B = Backward micro-batch
```

#### Micro-batching

Per ridurre la "bolla" (GPU idle), il batch viene diviso in **micro-batch** che fluiscono nel pipeline:

| Micro-batch | GPU 0 | GPU 1 |
|---|---|---|
| μ₀ | Forward | — |
| μ₁ | Forward | Forward μ₀ |
| μ₂ | Forward | Forward μ₁ |
| — | Backward μ₂ | Forward μ₂ |
| — | Backward μ₁ | Backward μ₂ |

---

## 3. Librerie di Comunicazione

### NCCL (NVIDIA Collective Communications Library)

| Operazione | Descrizione |
|---|---|
| `ncclAllReduce` | Somma/media di tensori tra tutte le GPU |
| `ncclBroadcast` | Invia un tensore da una GPU a tutte |
| `ncclAllGather` | Ogni GPU contribuisce una parte, tutte ricevono il tutto |
| `ncclReduceScatter` | Reduce + distribuzione del risultato |

```cpp
// Inizializzazione
ncclComm_t comm;
ncclUniqueId id;
ncclGetUniqueId(&id);
ncclCommInitRank(&comm, num_gpus, id, rank);

// Operazione collettiva (asincrona)
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
cudaStreamSynchronize(stream);
```

### MPI (Message Passing Interface)

Per la comunicazione **tra nodi** (macchine diverse collegate in rete).

```cpp
MPI_Init(&argc, &argv);
int rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// AllReduce inter-node
MPI_Allreduce(MPI_IN_PLACE, grad_buffer, count,
              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

// Divide per world_size per ottenere la media
for (int i = 0; i < count; i++)
    grad_buffer[i] /= world_size;
```

### Architettura ibrida tipica

```
┌─── Nodo 0 ──────────────┐    ┌─── Nodo 1 ──────────────┐
│ GPU0 ◀──NCCL──▶ GPU1    │◀──MPI──▶│ GPU2 ◀──NCCL──▶ GPU3    │
│    (intra-nodo veloce)   │  (inter-nodo)  │    (intra-nodo veloce)   │
└─────────────────────────┘    └─────────────────────────┘
```

---

## 4. Struttura File

| File | Contenuto |
|---|---|
| `src/distributed/comm.h` | Interfaccia astratta `Communicator` |
| `src/distributed/nccl_comm.cpp` | Implementazione NCCL (intra-nodo) |
| `src/distributed/mpi_comm.cpp` | Implementazione MPI (inter-nodo) |
| `src/distributed/data_parallel.h/.cpp` | Wrapper Data Parallelism (AllReduce gradienti) |
| `src/distributed/pipeline.h/.cpp` | Pipeline Parallelism con micro-batching |
| `src/distributed/tensor_parallel.h/.cpp` | Split orizzontale dei linear layer |
| `tests/test_distributed.cpp` | Test multi-processo (AllReduce, Broadcast) |

---

## 5. Piano di Verifica

| Test | Verifica |
|---|---|
| `test_allreduce` | 4 processi: AllReduce di tensori → media corretta |
| `test_broadcast` | Rank 0 invia, tutti ricevono lo stesso tensore |
| `test_data_parallel` | 2 GPU × stesso batch → gradienti identici |
| `test_pipeline` | 2 GPU pipeline: output identico a single-GPU |
| `test_tensor_parallel` | Split linear layer → output riassemblato corretto |

### Benchmark

| Benchmark | Misura |
|---|---|
| `bench_allreduce_bw` | Bandwidth AllReduce: GB/s al variare del tensore (1MB-1GB) |
| `bench_scaling` | Throughput (tokens/s) al variare del numero di GPU (1,2,4,8) |
| `bench_pipeline_bubble` | % di tempo GPU idle nel pipeline |

---

## 6. Requisiti

| Requisito | Dettaglio |
|---|---|
| NCCL | 2.10+ (incluso con CUDA Toolkit) |
| MPI | OpenMPI 4.0+ o Intel MPI |
| Interconnect | NVLink (intra-nodo) o InfiniBand (inter-nodo) raccomandata |
| CMake | `find_package(MPI REQUIRED)`, `find_package(NCCL)` |

> [!CAUTION]
> Il calcolo distribuito aggiunge complessità significativa.
> Si consiglia di completare prima le Fasi 1-7 su singola GPU
> prima di implementare questa fase.
