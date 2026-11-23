# Audit Remediation — tracker fix

Ordine: CRITICO/ALTO prima, poi MEDIO. Ogni fix: finding ID → file → test di verifica → build+ctest.
Nota: fix applicati dall'utente manualmente segnati come [user].

## Stato

| ID | Fix | File | Stato |
|----|-----|------|-------|
| F7b.11 | DP noise seed 42 → CSPRNG/random_device | experimental/federated/dp_noise.cpp | DONE |
| F7b.23 | beam_width clamp <= vocab_size | experimental/inference/generate.cpp | DONE |
| F7b.22 | delete logits_node + input_node lifetime in generate/beam_search | experimental/inference/generate.cpp | DONE |
| F7b.26 | score() empty tokens guard | experimental/rlhf/reward_model.cpp | DONE |
| F7b.18 | NON-BUG: formula PPO surrogate corretta (min ratio*A, clip*A) | - | verified-ok |
| F7b.29 | PPO train_step no-op (old==new, no backward): modulo didattico, serve policy separata | experimental/rlhf/ppo_trainer.cpp | deferred-design |
| F7.1 | checkpoint shape/size validation | src/train/checkpoint.cpp | pending |
| F7.13 | MCP audit args.dump() plaintext → scrub | src/mcp/ | pending |
| F7.16 | ProcessGroup shared collective state | src/distributed/ | pending |
| F7.53 | CudaBackend::free pinned/managed tracking | src/hal/cuda_backend.cu | pending |
| F4.1 | embedding_lookup token_id validation → throw | src/transformer/model.cpp | DONE |
| F4.13/14 | speculative decoding position + early-stop | src/infer/speculative.cpp | pending |
| F4.3 | KVCache rolling order after wrap | src/transformer/kv_cache | pending |
| F5.1 | generation autograd graph leak | src/server/server_generation.cpp | pending |
| F5.2 | finish_reason "length" | src/server/ | pending |
| F5.17 | loader error → random weights published | src/server/server.cpp | pending |
| F6.1 | BPE bos/eos ids | src/tokenizer/ | pending |
| F6.13 | sharded safetensors weight_map | src/io/ | pending |
| F6.20 | safetensors loader return ignored | src/io/ | pending |
| F6.17 | GGUF writer Q4_0 decl vs F16 payload | src/io/gguf_writer | pending |
| downloader traversal | is_safe_repo_path | src/hub/downloader.cpp | [user] done |
| GGUF nested KV depth cap | skip_gguf_value | src/io/gguf_loader.cpp + test | [user] done |
| server payload max + embeddings cap | server.cpp | src/server/server.cpp | [user] done |
