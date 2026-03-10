# Fase 8+ — Cose mancanti (Calcolo distribuito)

Riferimento: `fase8_distribuito.md`

---

## 1. Data parallelism

| Elemento | Stato | Nota |
|----------|--------|-----|
| AllReduce (NCCL/RCCL) per mediare gradienti | ❌ | Non presente |
| Ring AllReduce | ❌ | |
| Multi-GPU stesso nodo | ❌ | |
| test distribuito | ❌ | |

---

## 2. Model parallelism

| Elemento | Stato |
|----------|--------|
| Tensor parallelism (split layer orizzontale, AllGather) | ❌ |
| Pipeline parallelism (micro-batch, forward/backward staged) | ❌ |
| ZeRO-3 / FSDP2 (sharding parametri/gradienti/stati) | ❌ |

---

## 3. Multi-node

| Elemento | Stato |
|----------|--------|
| Comunicazione cross-node (NCCL, MPI) | ❌ |
| Configurazione rank, world_size, local_rank | ❌ |

---

## 4. File previsti (fase8)

| File | Stato |
|------|--------|
| src/distributed/ o simile (nccl_wrapper, zero, fsdp) | ❌ |
| Script/test multi-GPU | ❌ |

---

**Priorità:** fase avanzata; iniziare con data parallelism single-node (NCCL AllReduce) e wrapper per rank/world_size.
