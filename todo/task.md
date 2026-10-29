# LLM from Scratch in C++ — Master Task Tracker (Aggiornato)

## Documenti di Fase

### 🔧 Fondamenta (Fase 1–5)

- [x] **Fase 1** — [Motore Matematico](fase1_motore_matematico.md) (Tensori N-D contigui, BLAS CPU SIMD AVX2/AVX-512, Memory Pool con allocatore a blocchi)
- [x] **Fase 2** — [Autograd](fase2_autograd.md) (Computational Graph, Backprop topologico, nodi differenziabili e attivazioni)
- [x] **Fase 3** — [Transformer](fase3_transformer.md) (MLA, FFN con SwiGLU, LayerNorm/RMSNorm, Positional Encoding/RoPE, LLaMA e Qwen; Dual-Pipe pianificato)
- [x] **Fase 4** — [Tokenizer](fase4_tokenizer.md) (BPE con vocab.json/tokenizer.json, DataLoader multi-thread con batching e padding)
- [x] **Fase 5** — [Training](fase5_training.md) (AdamW, LR Schedule con warmup e cosine decay, Checkpointing binario)

### ⚡ Accelerazione Hardware (Fase 6–8)

- [x] **Fase 6** — [Ibrido CPU+GPU](fase6_ibrido_cpu_gpu.md) (HAL base con backend CPU e CUDA operativi; Metal e Vulkan stub, ROCm pianificato)
- [x] **Fase 7** — [Quantizzazione](fase7_quantizzazione.md) (INT4, FP8, BF16, AWQ, GPTQ kernel implementati; integrazione INT4 nel forward di inferenza pendente)
- [x] **Fase 8** — [Distribuito](fase8_distribuito.md) (LocalComm, MPI, NCCL bootstrap, DataParallel, TensorParallel, PipelineParallel)

### 🛡️ Stabilità & Allineamento (Fase 9–10)

- [x] **Fase 9** — [Stabilità](fase9_stabilita.md) (Gradient clipping, loss scaler dinamico AMP, pesi EMA, NaNDebugger, ciclo di 100 step completato)
- [x] **Fase 10** — [Fine-Tuning](fase10_finetuning.md) (Sampler probabilistico/deterministico, SFT preprocessing con loss mask, SFTTrainer con masked cross-entropy, AdamW, freeze/unfreeze, test convergenza 7/7)

### 🚀 Ottimizzazioni Avanzate (Fase 11–15)

- [x] **Fase 11** — [Flash Attention](fase11_flash_attention.md) (Integrazione kernel CUDA FlashAttention-2, online softmax e tiling completati; fallback CPU e dispatch dinamico pendenti)
- [x] **Fase 12** — [Profiling](fase12_profiling.md) (Tensor Logger completo con rilevamento anomalie NaN/Inf; profiling Nsight/Tracy pendente)
- [ ] **Fase 13** — [Deployment](fase13_deployment.md) (Parziale: Server HTTP cpp-httplib presente con rotte /health, /metrics e endpoint OpenAI; risposte attualmente stub, collegamento al forward del modello pendente)
- [x] **Fase 14** — [Speculative Decoding](fase14_speculative_decoding.md) (DraftModel, SpeculativeDecoder Leviathan, campionamento accettazione/rifiuto, gamma configurabile, metriche; integrazione CLI/server pendente)
- [x] **Fase 15** — [Model Checkpoints](fase15_checkpoint_sharding.md) (Sharded Safetensors con indice manifest JSON e loader HF LLaMA completi; GGUF parser raw tensors implementato, loader pesi GGUF su modello pendente)

### 🧠 Architettura Avanzata (Fase 16–17)

- [x] **Fase 16** — [Long Context](fase16_long_context.md) (Sliding Window Attention, Prefix Cache LRU, Paged KV Cache e RoPE NTK-aware implementati; modulazione frequenziale YaRN completa pendente)
- [x] **Fase 17** — [Mixture of Experts](fase17_moe.md) (DeepSeekMoE router top-k, shared experts, auxiliary load balance loss, MoELayer integrato in TransformerModel)

### 🔒 Sicurezza & Guardrails (Fase 18)

- [x] **Fase 18** — [Sicurezza](fase18_sicurezza.md) (ToxicityClassifier neurale, Guardrails e sanitizzazione completati e testati; fix EmptyTokens validato)

### 📚 Apprendimento Continuo (Fase 19)

- [x] **Fase 19** — [LoRA](fase19_lora.md) (LoRALinear matrici A/B, scaling alpha/r, LoRAAdapter multi-layer, salvataggio/caricamento binario)

### 🌍 Multimodalità (Fase 20)

- [x] **Fase 20** — [Multimodale](fase20_multimodale.md) (Vision Encoder/ViT patch embed, Audio Mel-spectrogram + 1D conv, Cross-Attention, InputBuilder; test validati)

### 🔬 Interpretabilità (Fase 21)

- [x] **Fase 21** — [Interpretabilità](fase21_interpretabilita.md) (AttentionCapture, ActivationProbe, LogitLens, JSON/CSV export e test validati)

### 🔗 Tool Use & RAG (Fase 22)

- [x] **Fase 22** — [RAG & Tools](fase22_rag_tools.md) (VectorDBClient in-memory, Embedding locale, RAG Retriever con prompt injection, Tool Registry, Function Call Parser e test validati)

### 🎯 RLHF (Fase 23)

- [x] **Fase 23** — [RLHF](fase23_rlhf.md) (RewardModel scalare con loss Bradley-Terry, utilità PPO/GRPO: GAE, KL divergenza, vantaggi, obiettivo clippato, test validati)

### ✅ Fact-Checking (Fase 24)

- [x] **Fase 24** — [Grounding](fase24_grounding.md) (Self-Consistency majority vote, Citation Engine con verifica claim/fonti, Confidence score da logit; test validati)

### 📊 Valutazione & DevOps (Fase 25–28)

- [x] **Fase 25** — [Benchmarking](fase25_benchmarking.md) (EvalSuite multiple-choice MMLU, perplexity, loss-curve/overfitting detection, report JSON/CSV; test validati)
- [x] **Fase 26** — [Containerizzazione](fase26_containerizzazione.md) (Dockerfile multi-stage CPU/CUDA, docker-compose.yml e manifest Kubernetes in k8s/ deployment, pvc, service)
- [x] **Fase 27** — [CI/CD](fase27_cicd.md) (GitHub Actions workflow .github/workflows/ci.yml con pipeline di test su CPU e GPU)
- [x] **Fase 28** — [Monitoring](fase28_monitoring.md) (Continuous Batcher con slot reuse, structured JSON logging, InferenceMetrics con exporter Prometheus text, dashboard Grafana; aggancio a server HTTP pendente)

### 🗄️ Dati & Ottimizzazione (Fase 29–30)

- [x] **Fase 29** — [Data Curation](fase29_data_curation.md) (MinHash signature, LSH index a bande, dedup pipeline con verifica Jaccard, synthetic data generation con quality judge; test validati)
- [x] **Fase 30** — [Kernel Fusion](fase30_kernel_fusion.md) (Fused CUDA kernels Linear+ReLU, LayerNorm+Residual, Graph Compiler automatico e test validati)

### ⚖️ Etica & Sicurezza Avanzata (Fase 31)

- [x] **Fase 31** — [Ethical Bias](fase31_ethical_bias.md) (Red Teaming con 4 categorie attacchi, ToxicityClassifier, BiasScorer con disparity analysis; test validati)

### 🔌 MCP Integration (Fase 32–34)

- [x] **Fase 32** — [MCP Client](fase32_mcp_client.md) (src/mcp/ client JSON-RPC 2.0 con trasporti stdio e http/sse, tests/test_mcp_client.cpp validati)
- [x] **Fase 33** — [Orchestrazione Tool](fase33_orchestrazione_tool.md) (src/agent/react.h, action parser, context manager; tests/test_react.cpp validati; adapter su LLM reale pendente)
- [x] **Fase 34** — [Sicurezza MCP](fase34_sicurezza_mcp.md) (src/mcp/ guard, scrubber, audit log operativi; inizializzazione statica tool_risk_ e isolamento test validati 11/11)

### 🧪 Ragionamento Avanzato (Fase 35–36)

- [x] **Fase 35** — [Test-Time Compute](fase35_test_time_compute.md) (src/reasoning/ CoT, MCTS reasoner, keyword/rule verifier, budget; tests/test_reasoning.cpp validati; adapter su LLM reale pendente)
- [x] **Fase 36** — [Sparse Attention](fase36_sparse_attention.md) (Dynamic Sparse Attention CPU, Block-Sparse CUDA kernel, Sparsity Mask Generator e test validati; integrazione in TransformerBlock pendente)

### 📱 On-Device & Evoluzione (Fase 37–38)

- [x] **Fase 37** — [Federated Learning](fase37_federated_learning.md) (Differential privacy clip+noise, secure aggregation, per-user LoRA adapter, privacy budget; test validati)
- [x] **Fase 38** — [Self-Evolution & NAS](fase38_self_evolution_nas.md) (src/automl/ AutoTune kernel search, Pruner magnitude/structured, TuneCache e test validati 11/11)

### 🧠 Memoria & Multi-Agent (Fase 39–40)

- [x] **Fase 39** — [Long-term Memory](fase39_long_term_memory.md) (src/memory/ DNCMemory con matrici e key-value addressing, State Compressor e PersistentMemory su disco; test validati 19/19)
- [x] **Fase 40** — [Multi-Agent](fase40_multi_agent.md) (src/agent/ Delphi Consensus, Supervisor, Specialist, Planner e test validati 9/9)

### 🔮 Frontiera (Fase 41–46)

- [x] **Fase 41** — [Neuromorphic](fase41_neuromorphic.md) (src/neuro/ LIFNeuron, SpikingLinear con STDP, EventEngine event-driven asincrono, SpikeEncoder rate/latency; test validati 10/10)
- [x] **Fase 42** — [Liquid Networks](fase42_liquid_networks.md) (src/liquid/ NeuralODE con RK4/Euler, LTCNetwork a costanti di tempo liquide, AdaptiveStepController per profondità computazionale; test validati 8/8)
- [x] **Fase 43** — [World Models](fase43_world_models.md) (src/world/ WorldModel predittivo con reward/uncertainty, ImaginationEngine simulatore multi-step e pianificazione MPC, JEPA con target encoder EMA; test validati 6/6)
- [x] **Fase 44** — [Quantum](fase44_quantum.md) (src/quantum/ QuantumState ampiezze complesse e gate unitari/CNOT, QuantumCircuit ansatz parametrizzato, gradiente parameter-shift esatto, VQE optimizer; test validati 5/5)
- [x] **Fase 45** — [Self-Replication](fase45_self_replication.md) (AutoRefactorer, Sandbox, HotSwapManager, GitManager, LLMClient implementati e test validati)
- [x] **Fase 46** — [Semantic Hash](fase46_semantic_hash.md) (SemanticHash LSH, KnowledgeFunction con build/query O(1), FactExtractor, multi-resolution fallback e test validati)

### 💻 Interfacce & Serving (Fase 47–49)

- [x] **Fase 47** — [CLI](fase47_cli.md) (Eseguibile serius: comando chat interattiva reale con KV-cache per LLaMA/Qwen/Transformer; comando serve operativo con server HTTP; CLI framework con parser argomenti, comandi chat/serve/doctor)
- [x] **Fase 48** — [API OpenAI](fase48_api_openai.md) (Endpoint OpenAI /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings implementati con streaming SSE; autenticazione API key e rate limiting non implementati ma endpoint funzionano)
- [x] **Fase 49** — [Chat UI](fase49_chat_ui.md) (Interfaccia web completa con HTML/CSS/JS, streaming SSE, selezione modello, parametri generazione, dark mode; servita come file statici dal server)

### 🌐 Model Hub & Estendibilità (Fase 50–53)

- [x] **Fase 50** — [Model Hub](fase50_model_hub.md) (Server HTTP base, metadata manager JSON, downloader per HuggingFace Hub, converter Safetensors->GGUF (stub), comandi CLI import/convert; autenticazione JWT e storage SQLite non implementati)
- [x] **Fase 51** — [Documentazione](fase51_documentazione.md) (Guide complete: Quick Start, CLI Reference, API Reference, Installation, Supported Models, Architecture, Troubleshooting, GPU Setup)
- [x] **Fase 52** — [Installer & Packaging](fase52_installer_packaging.md) (Script install.sh per Linux/macOS, install.ps1 per Windows, CI/CD GitHub Actions per build multi-piattaforma)
- [x] **Fase 53** — [Plugin System](fase53_plugin_system.md) (PluginManager con caricamento dinamico DLL/SO, capability e sandboxing risorse implementati)

### 🧪 Benchmarking & Qualità (Fase 54–59)

- [x] **Fase 54** — [Performance Benchmarks](fase54_performance_benchmarks.md)
- [x] **Fase 55** — [Stress Testing](fase55_stress_testing.md)
- [ ] **Fase 56** — [E2E Tests](fase56_e2e_tests.md)
- [ ] **Fase 57** — [Voice Interface](fase57_voice_interface.md)
- [ ] **Fase 58** — [Edge Deployment](fase58_edge_deployment.md)
- [ ] **Fase 59** — [A/B Testing](fase59_ab_testing.md)

### ⚙️ Piattaforma & Intelligenza Collettiva (Fase 60–64)

- [ ] **Fase 60** — [Tool Server](fase60_tool_server.md)
- [ ] **Fase 61** — [Skill System](fase61_skill_system.md)
- [ ] **Fase 62** — [Batch Processing](fase62_batch_processing.md)
- [ ] **Fase 63** — [AI Swarm](fase63_ai_swarm.md)
- [ ] **Fase 64** — [Riservata](fase64_riservata.md)

### 🚀 Ecosistema Avanzato & Frontiere Future (Fase 65–72)

- [ ] **Fase 65** — [Community & Ecosystem](fase65_community_ecosystem.md)
- [ ] **Fase 66** — [Edge AI Optimization](fase66_edge_ai_optimization.md)
- [ ] **Fase 67** — [Real-time Streaming](fase67+.md)
- [ ] **Fase 68** — [Advanced Security](fase68+.md)
- [ ] **Fase 69** — [Robotics Integration](fase69+.md)
- [ ] **Fase 70** — [Biomedical AI](fase70+.md)
- [ ] **Fase 71** — [Climate Science](fase71+.md)
- [ ] **Fase 72** — [Quantum ML](fase72+.md)

---

**Totale: 73 documenti · 72 fasi · oltre 35.000 righe di pianificazione tecnica**
