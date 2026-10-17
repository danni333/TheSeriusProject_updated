# Fase 8+ — Stato Implementazione e Cose Mancanti (Calcolo distribuito)

Riferimento: `fase8_distribuito.md`

---

## 1. Data parallelism

| Elemento | Stato | Nota |
|----------|--------|-----|
| AllReduce (NCCL/MPI/Local) per mediare gradienti | ✅ | Implementato in `src/distributed/data_parallel.cpp/.h` con `comm->all_reduce` (Avg/Sum) |
| Ring AllReduce / Local simulation | ✅ | `LocalComm` con code thread-safe per simulare multi-rank su singolo processo |
| Multi-GPU / Multi-device intra-node (NCCL) | ✅ | `src/distributed/nccl_comm.cpp/.h` con stream CUDA e collective sincronizzate |
| Test suite distribuito | ✅ | `tests/test_distributed.cpp`, `tests/test_distributed_llama.cpp`, `tests/test_nccl_bootstrap.cpp` |

---

## 2. Model parallelism

| Elemento | Stato | Nota |
|----------|--------|-----|
| Tensor parallelism (Row/Col parallel Linear, AllGather, ReduceScatter) | ✅ | Implementato in `src/distributed/tensor_parallel.cpp/.h` |
| Pipeline parallelism (micro-batch, forward/backward staged 1F1B) | ✅ | Implementato in `src/distributed/pipeline.cpp/.h` e `tests/test_pipeline_llama.cpp` |
| ZeRO-3 / FSDP2 (sharding parametri/gradienti/stati) | [/] | Sharding checkpoint presente (Fase 15); ZeRO-3 runtime completo pianificato |

---

## 3. Multi-node & Communicator

| Elemento | Stato | Nota |
|----------|--------|-----|
| Comunicazione cross-node (MPI) | ✅ | Implementato in `src/distributed/mpi_comm.cpp/.h` |
| Bootstrap automatico NCCL via MPI | ✅ | `Communicator::create_nccl(world_size, rank)` distribuisce NCCL ID via MPI_COMM_WORLD |
| Configurazione rank, world_size, local_rank | ✅ | Integrato nell'interfaccia `Communicator` |

---

## 4. File sorgente presenti (fase 8)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/distributed/comm.h` | ✅ | Interfaccia astratta `Communicator`, `ReduceOp`, `CommBackend` |
| `src/distributed/local_comm.h/.cpp` | ✅ | Communicator in-memory multi-thread per testing locale |
| `src/distributed/mpi_comm.h/.cpp` | ✅ | Backend MPI per nodi multipli / CPU |
| `src/distributed/nccl_comm.h/.cpp` | ✅ | Backend NCCL per GPU multi-device con bootstrap |
| `src/distributed/data_parallel.h/.cpp` | ✅ | Wrapper DDP per accumulazione e media gradienti |
| `src/distributed/tensor_parallel.h/.cpp` | ✅ | Layer paralleli e primitive di tensor parallelism |
| `src/distributed/pipeline.h/.cpp` | ✅ | Stage di pipeline parallelism e scheduling micro-batch |

---

**Priorità rimanenti:** stress test multi-nodo su cluster dedicato (MPI+NCCL) ed estensione ZeRO-3 runtime.
