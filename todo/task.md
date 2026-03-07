# LLM from Scratch in C++ — Master Task Tracker (Aggiornato)

## Documenti di Fase

### 🔧 Fondamenta (Fase 1–5)

- [x] **Fase 1** — [Motore Matematico](fase1_motore_matematico.md) (Tensori, BLAS, Memory Pooling)
- [x] **Fase 2** — [Autograd](fase2_autograd.md) (Computational Graph, Backprop, Attivazioni)
- [x] **Fase 3** — [Transformer](fase3_transformer.md) (MLA, FFN, LayerNorm, Positional Encoding, Dual-Pipe)
- [x] **Fase 4** — [Tokenizer](fase4_tokenizer.md) (BPE, DataLoader multi-thread)
- [x] **Fase 5** — [Training](fase5_training.md) (AdamW, LR Schedule, Checkpointing)

### ⚡ Accelerazione Hardware (Fase 6–8)

- [x] **Fase 6** — [Ibrido CPU+GPU](fase6_ibrido_cpu_gpu.md) (HAL, CUDA/ROCm/Metal, Unified Memory)
- [x] **Fase 7** — [Quantizzazione](fase7_quantizzazione.md) (FP8 MX6, BF16, INT4, KV Cache)
- [x] **Fase 8** — [Distribuito](fase8_distribuito.md) (Data/Model Parallelism, NCCL, MPI)

### 🛡️ Stabilità & Allineamento (Fase 9–10)

- [x] **Fase 9** — [Stabilità](fase9_stabilita.md) (Gradient Clipping, Weight Init, Loss Scaling, SWA)
- [x] **Fase 10** — [Fine-Tuning](fase10_finetuning.md) (SFT, Sampling: Top-K/P, Beam Search)

### 🚀 Ottimizzazioni Avanzate (Fase 11–15)

- [x] **Fase 11** — [Flash Attention](fase11_flash_attention.md) (FlashAttention-3, TMA, RingAttention)
- [x] **Fase 12** — [Profiling](fase12_profiling.md) (Tensor Logger, Nsight, NumPy Testing)
- [x] **Fase 13** — [Deployment](fase13_deployment.md) (Web Server, REST API, SSE, WebSocket)
- [x] **Fase 14** — [Speculative Decoding](fase14_speculative_decoding.md) (Draft + Target Model)
- [x] **Fase 15** — [Model Checkpoints](fase15_checkpoint_sharding.md) (GGUF Local Import, Safetensors, mmap)

### 🧠 Architettura Avanzata (Fase 16–17)

- [x] **Fase 16** — [Long Context](fase16_long_context.md) (RoPE, Sliding Window Attention)
- [x] **Fase 17** — [Mixture of Experts](fase17_moe.md) (DeepSeekMoE, Shared Experts, Fine-Grained)

### 🔒 Sicurezza & Guardrails (Fase 18)

- [x] **Fase 18** — [Sicurezza](fase18_sicurezza.md) (Logit Bias, Toxicity Classifier)

### 📚 Apprendimento Continuo (Fase 19)

- [x] **Fase 19** — [LoRA](fase19_lora.md) (Low-Rank Adaptation, Online Learning)

### 🌍 Multimodalità (Fase 20)

- [x] **Fase 20** — [Multimodale](fase20_multimodale.md) (Vision Encoder, Cross-Attention, Audio)

### 🔬 Interpretabilità (Fase 21)

- [x] **Fase 21** — [Interpretabilità](fase21_interpretabilita.md) (Attention Viz, Neuron Mapping, Logit Lens)

### 🔗 Tool Use & RAG (Fase 22)

- [x] **Fase 22** — [RAG & Tools](fase22_rag_tools.md) (Vector DB, Function Calling)

### 🎯 RLHF (Fase 23)

- [x] **Fase 23** — [RLHF](fase23_rlhf.md) (Reward Model, GRPO)

### ✅ Fact-Checking (Fase 24)

- [x] **Fase 24** — [Grounding](fase24_grounding.md) (Self-Consistency, Citation Engine)

### 📊 Valutazione & DevOps (Fase 25–28)

- [x] **Fase 25** — [Benchmarking](fase25_benchmarking.md) (MMLU, GSM8K, Perplexity)
- [x] **Fase 26** — [Containerizzazione](fase26_containerizzazione.md) (Docker, Kubernetes)
- [x] **Fase 27** — [CI/CD](fase27_cicd.md) (GPU Testing, Model Versioning)
- [x] **Fase 28** — [Monitoring](fase28_monitoring.md) (Prometheus, Continuous Batching)

### 🗄️ Dati & Ottimizzazione (Fase 29–30)

- [x] **Fase 29** — [Data Curation](fase29_data_curation.md) (MinHash/LSH Dedup, Synthetic Data)
- [x] **Fase 30** — [Kernel Fusion](fase30_kernel_fusion.md) (Operator Fusion, Static Graph)

### ⚖️ Etica & Sicurezza Avanzata (Fase 31)

- [x] **Fase 31** — [Ethical Bias](fase31_ethical_bias.md) (Red Teaming, Bias Scoring)

### 🔌 MCP Integration (Fase 32–34)

- [x] **Fase 32** — [MCP Client](fase32_mcp_client.md) (JSON-RPC, Tool Discovery)
- [x] **Fase 33** — [Orchestrazione Tool](fase33_orchestrazione_tool.md) (ReAct, Context Injection)
- [x] **Fase 34** — [Sicurezza MCP](fase34_sicurezza_mcp.md) (Human-in-the-Loop, Scrubbing)

### 🧪 Ragionamento Avanzato (Fase 35–36)

- [x] **Fase 35** — [Test-Time Compute](fase35_test_time_compute.md) (System 2 Reasoning, RL-based Search)
- [x] **Fase 36** — [Sparse Attention](fase36_sparse_attention.md) (Dynamic Sparsity, Block-Sparse)

### 📱 On-Device & Evoluzione (Fase 37–38)

- [x] **Fase 37** — [Federated Learning](fase37_federated_learning.md) (Local GGUF LoRA, Differential Privacy)
- [x] **Fase 38** — [Self-Evolution & NAS](fase38_self_evolution_nas.md) (SSM/Mamba, Auto-Tuning, Pruning)

### 🧠 Memoria & Multi-Agent (Fase 39–40)

- [x] **Fase 39** — [Long-term Memory](fase39_long_term_memory.md) (DNC, State Compression)
- [x] **Fase 40** — [Multi-Agent](fase40_multi_agent.md) (Supervisor, Consensus Algorithms)

### 🔮 Frontiera (Fase 41–46)

- [x] **Fase 41** — [Neuromorphic](fase41_neuromorphic.md) (SNN, Event-Driven Execution)
- [x] **Fase 42** — [Liquid Networks](fase42_liquid_networks.md) (Neural ODE, LTC)
- [x] **Fase 43** — [World Models](fase43_world_models.md) (Predictive Models, JEPA)
- [x] **Fase 44** — [Quantum](fase44_quantum.md) (Quantum-Classical Hybrid Kernels)
- [x] **Fase 45** — [Self-Replication](fase45_self_replication.md) (Auto-Refactoring, Hot-Swap)
- [x] **Fase 46** — [Semantic Hash](fase46_semantic_hash.md) (Knowledge Function, LSH)

---

**Totale: 47 documenti · 46 fasi · ~25.000 righe di pianificazione tecnica**
