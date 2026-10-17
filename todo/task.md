# LLM from Scratch in C++ — Master Task Tracker (Aggiornato)

## Documenti di Fase

### 🔧 Fondamenta (Fase 1–5)

- [x] **Fase 1** — [Motore Matematico](fase1_motore_matematico.md) (Tensori, BLAS, Memory Pooling)
- [x] **Fase 2** — [Autograd](fase2_autograd.md) (Computational Graph, Backprop, Attivazioni)
- [x] **Fase 3** — [Transformer](fase3_transformer.md) (MLA, FFN, LayerNorm, Positional Encoding, Dual-Pipe)
- [x] **Fase 4** — [Tokenizer](fase4_tokenizer.md) (BPE, DataLoader multi-thread)
- [x] **Fase 5** — [Training](fase5_training.md) (AdamW, LR Schedule, Checkpointing)

### ⚡ Accelerazione Hardware (Fase 6–8)

- [x] **Fase 6** — [Ibrido CPU+GPU](fase6_ibrido_cpu_gpu.md) (HAL base presente, CPU/CUDA backends)
- [x] **Fase 7** — [Quantizzazione](fase7_quantizzazione.md) (INT4, FP8, BF16, GGUF quants supportati)
- [x] **Fase 8** — [Distribuito](fase8_distribuito.md) (LocalComm, MPI, NCCL bootstrap, DataParallel, TensorParallel, PipelineParallel)

### 🛡️ Stabilità & Allineamento (Fase 9–10)

- [x] **Fase 9** — [Stabilità](fase9_stabilita.md) (componenti core, ciclo di 100 step e smoke test AMP del Trainer su Transformer CPU completati; resta lo stress test lungo e su hardware dedicato)
- [x] **Fase 10** — [Fine-Tuning](fase10_finetuning.md) (Sampler probabilistico/deterministico, SFT preprocessing con loss mask, loader GGUF/Safetensors, SFTTrainer con masked cross-entropy, AdamW, freeze/unfreeze, test convergenza 7/7)

### 🚀 Ottimizzazioni Avanzate (Fase 11–15)

- [x] **Fase 11** — [Flash Attention](fase11_flash_attention.md) (Integrazione kernel CUDA, online softmax e tiling completati)
- [x] **Fase 12** — [Profiling](fase12_profiling.md) (Tensor Logger completo)
- [x] **Fase 13** — [Deployment](fase13_deployment.md) (Server HTTP cpp-httplib, endpoint OpenAI /v1/chat/completions con streaming SSE, /metrics, /health, /v1/models)
- [x] **Fase 14** — [Speculative Decoding](fase14_speculative_decoding.md) (DraftModel, SpeculativeDecoder, campionamento accettazione/rifiuto, gamma configurabile, metriche)
- [x] **Fase 15** — [Model Checkpoints](fase15_checkpoint_sharding.md) (Sharded Safetensors con indice manifest JSON, writer/loader multi-shard, GGUF loader completo)

### 🧠 Architettura Avanzata (Fase 16–17)

- [x] **Fase 16** — [Long Context](fase16_long_context.md) (RoPE scaling YaRN/NTK-aware, Sliding Window Attention, Prefix Cache e Paged KV Cache implementati)
- [x] **Fase 17** — [Mixture of Experts](fase17_moe.md) (DeepSeekMoE router top-k, shared experts, auxiliary load balance loss, MoELayer integrato in TransformerModel)

### 🔒 Sicurezza & Guardrails (Fase 18)

- [x] **Fase 18** — [Sicurezza](fase18_sicurezza.md) (Guardrails con token/phrase blocklist, Toxicity Classifier, Output Sanitization e response filtering implementati)

### 📚 Apprendimento Continuo (Fase 19)

- [x] **Fase 19** — [LoRA](fase19_lora.md) (LoRALinear A/B matrices, scaling alpha/r, LoRAAdapter multi-layer, salvataggio/caricamento binario)

### 🌍 Multimodalità (Fase 20)

- [x] **Fase 20** — [Multimodale](fase20_multimodale.md) (Vision Encoder/ViT, Audio/Mel, Cross-Attention, InputBuilder; test validati)

### 🔬 Interpretabilità (Fase 21)

- [x] **Fase 21** — [Interpretabilità](fase21_interpretabilita.md) (AttentionCapture, ActivationProbe, LogitLens, JSON/CSV export e test validati)

### 🔗 Tool Use & RAG (Fase 22)

- [x] **Fase 22** — [RAG & Tools](fase22_rag_tools.md) (VectorDBClient, Embedding locale, RAG Retriever, Tool Registry, Function Call Parser e test validati)

### 🎯 RLHF (Fase 23)

- [x] **Fase 23** — [RLHF](fase23_rlhf.md) (RewardModel con testa scalare e loss Bradley-Terry, utilità PPO/GRPO: KL divergenza, vantaggi, obiettivo clippato, test validati)

### ✅ Fact-Checking (Fase 24)

- [x] **Fase 24** — [Grounding](fase24_grounding.md) (Self-Consistency, Citation Engine con verifica supporto, Confidence score da log-prob; test validati)

### 📊 Valutazione & DevOps (Fase 25–28)

- [x] **Fase 25** — [Benchmarking](fase25_benchmarking.md) (EvalSuite multiple-choice, perplexity, loss-curve/overfitting detection, report JSON/CSV; test validati)
- [ ] **Fase 26** — [Containerizzazione](fase26_containerizzazione.md)
- [ ] **Fase 27** — [CI/CD](fase27_cicd.md)
- [x] **Fase 28** — [Monitoring](fase28_monitoring.md) (Prometheus metrics testo, Continuous Batcher con slot reuse, structured JSON logging, grafana dashboard; test validati)

### 🗄️ Dati & Ottimizzazione (Fase 29–30)

- [x] **Fase 29** — [Data Curation](fase29_data_curation.md) (MinHash signature, LSH index, dedup pipeline con verifica Jaccard, synthetic data generation con quality judge; test validati)
- [x] **Fase 30** — [Kernel Fusion](fase30_kernel_fusion.md) (Fused CUDA kernels presenti: Linear+ReLU, LayerNorm+Residual, Graph Compiler e test validati)

### ⚖️ Etica & Sicurezza Avanzata (Fase 31)

- [x] **Fase 31** — [Ethical Bias](fase31_ethical_bias.md) (Red Teaming con 4 categorie attacchi, ToxicityClassifier, BiasScorer con disparity analysis; test validati)

### 🔌 MCP Integration (Fase 32–34)

- [ ] **Fase 32** — [MCP Client](fase32_mcp_client.md)
- [ ] **Fase 33** — [Orchestrazione Tool](fase33_orchestrazione_tool.md)
- [ ] **Fase 34** — [Sicurezza MCP](fase34_sicurezza_mcp.md)

### 🧪 Ragionamento Avanzato (Fase 35–36)

- [ ] **Fase 35** — [Test-Time Compute](fase35_test_time_compute.md)
- [x] **Fase 36** — [Sparse Attention](fase36_sparse_attention.md) (Dynamic Sparse Attention CPU, Block-Sparse CUDA kernel, Sparsity Mask Generator e test validati)

### 📱 On-Device & Evoluzione (Fase 37–38)

- [x] **Fase 37** — [Federated Learning](fase37_federated_learning.md) (Differential privacy clip+noise, secure aggregation, per-user LoRA adapter, privacy budget; test validati)
- [ ] **Fase 38** — [Self-Evolution & NAS](fase38_self_evolution_nas.md)

### 🧠 Memoria & Multi-Agent (Fase 39–40)

- [ ] **Fase 39** — [Long-term Memory](fase39_long_term_memory.md)
- [ ] **Fase 40** — [Multi-Agent](fase40_multi_agent.md)

### 🔮 Frontiera (Fase 41–46)

- [ ] **Fase 41** — [Neuromorphic](fase41_neuromorphic.md)
- [ ] **Fase 42** — [Liquid Networks](fase42_liquid_networks.md)
- [ ] **Fase 43** — [World Models](fase43_world_models.md)
- [ ] **Fase 44** — [Quantum](fase44_quantum.md)
- [ ] **Fase 45** — [Self-Replication](fase45_self_replication.md)
- [ ] **Fase 46** — [Semantic Hash](fase46_semantic_hash.md)

### 💻 Interfacce & Serving (Fase 47–49)

- [x] **Fase 47** — [CLI](fase47_cli.md) (Eseguibile `serius`, comandi `serve` e `chat` interattiva con KV-cache per LLaMA/Qwen/Transformer)
- [x] **Fase 48** — [API OpenAI](fase48_api_openai.md) (Endpoints OpenAI compatibili: `/v1/chat/completions` con streaming SSE, `/v1/completions`, `/v1/models`, `/v1/embeddings`)
- [ ] **Fase 49** — [Chat UI](fase49_chat_ui.md) (Interfaccia web)

### 🌐 Model Hub & Estendibilità (Fase 50–53)

- [x] **Fase 50** — [Model Hub](fase50_model_hub.md) (Registry server, metadata manager, search engine e configs per 1B/7B/MoE)
- [ ] **Fase 51** — [Documentazione](fase51_documentazione.md)
- [ ] **Fase 52** — [Installer & Packaging](fase52_installer_packaging.md)
- [x] **Fase 53** — [Plugin System](fase53_plugin_system.md) (PluginManager, caricamento dinamico DLL/SO, capability, sandboxing e monitoraggio risorse)

### 🧪 Benchmarking & Qualità (Fase 54–59)

- [ ] **Fase 54** — [Performance Benchmarks](fase54_performance_benchmarks.md)
- [ ] **Fase 55** — [Stress Testing](fase55_stress_testing.md)
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
