# Fase 17+ — Stato Implementazione e Cose Mancanti (Mixture of Experts)

Riferimento: `fase17_moe.md`

---

## 1. Mixture of Experts & DeepSeek-MoE Style

| Elemento | Stato | Nota |
|----------|--------|-----|
| Router (softmax su gate, top-k experts) | ✅ | Implementato in `src/moe/router.cpp/.h` con pesi normalizzati e routing deterministico |
| Layer MoE (N experts routed, forward/backward) | ✅ | Implementato in `src/moe/moe_layer.cpp/.h` con parametri dedicati per esperto |
| Shared Experts (stile DeepSeek-MoE) | ✅ | Supportato tramite `config.moe.shared_experts` in `src/moe/moe_layer.cpp` |
| Load Balancing Auxiliary Loss | ✅ | Implementato in `src/moe/load_balance.cpp/.h` |
| Integrazione in TransformerBlock / Model | ✅ | `TransformerBlock::has_moe()`, parametri MoE registrati e forward autograd abilitato |
| Config MoE (`num_experts`, `top_k`, `shared_experts`) | ✅ | Supportato in `ModelConfig` e configs JSON (`configs/model_moe_1b.json`, `configs/model_moe_demo.json`) |
| Expert parallelism distribuito | [/] | Primitives di TensorParallel/DDP presenti (Fase 8); sharding specifico per esperto pianificato |

---

## 2. Test e Validazione

| Elemento | Stato | Nota |
|----------|--------|-----|
| Test Router Top-K e normalizzazione pesi | ✅ | `MoE.RouterSelectsTopKAndNormalizesWeights` in `tests/test_moe.cpp` |
| Test MoELayer output shape | ✅ | `MoE.MoELayerProducesCorrectOutputShape` |
| Test Backward pass gradiente router & esperti | ✅ | `MoE.BackwardUpdatesExpertAndRouterParameters` |
| Test TransformerModel end-to-end con MoE | ✅ | `MoE.TransformerModelRunsWithConfigEnabledMoE` |

---

## 3. File sorgente presenti (fase 17)

| File | Stato | Descrizione |
|------|--------|-------------|
| `src/moe/router.h/.cpp` | ✅ | Routing per token verso i top-k esperti con normalizzazione softmax |
| `src/moe/expert.h/.cpp` | ✅ | Modulo FFN per singolo esperto |
| `src/moe/load_balance.h/.cpp` | ✅ | Calcolo auxiliary loss per bilanciamento carico tra esperti |
| `src/moe/moe_layer.h/.cpp` | ✅ | Aggregazione esperti routed + shared experts |
| `tests/test_moe.cpp` | ✅ | Suite di test completa (router, shape, backward, full model) |
| `configs/model_moe_*.json` | ✅ | Configurazioni JSON pronte per modelli MoE |

---

**Priorità rimanenti:** EP (Expert Parallelism) su multi-GPU distribuito con all-to-all communication.
