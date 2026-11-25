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
| F7.1 | checkpoint validation | src/train/checkpoint.cpp | DONE (fix preesistente verificato) |
| F7.2 | save_checkpoint .at(p) → zero-state fallback + load materializza state assente | src/train/checkpoint.cpp | DONE |
| F7.13 | MCP audit args.dump() plaintext → scrub | src/mcp/ | pending |
| F7.16 | MPI count int-overflow guard (mpi_count) | src/distributed/mpi_comm.cpp | DONE |
| F7.53 | CudaBackend::free pinned tracking + thread-safe singleton (F7.52) | src/hal/cuda_backend.{cu,h} | DONE |
| F4.1 | embedding_lookup token_id validation → throw | src/transformer/model.cpp | DONE |
| F4.13/14 | speculative decoding position + early-stop | src/infer/speculative.cpp | pending |
| F4.3 | KVCache rolling: ordered_seq_() gather in ordine cronologico dopo wrap (get_keys/get_values) | src/transformer/kv_cache.{h,cpp} | DONE |
| F5.1 | free_graph() helper (topological_sort, params esclusi); input_node heap-allocated in generate_transformer/stream_transformer | src/server/server_generation.cpp | DONE |
| F5.2 | "stopped" flag in generate/stream llama+transformer; finish_reason stop/length propagato a server.cpp chat+completions | src/server/server_generation.cpp, src/server/server.cpp | DONE |
| F5.3 | embeddings zero su modelli GPU-offloaded: rimosso dtype guard, item() handle F16/BF16+strides | src/server/server.cpp | DONE |
| F5.17 | GGUF load fail → entry.llama reset (load_model returns false); dir load fail + weight files present → reject; config-only dirs inalterati | src/server/server.cpp | DONE |
| F6.1 | BPE bos/eos ids da bos_token/eos_token/added_tokens + fallback | src/tokenizer/bpe.cpp | DONE |
| F6.13 | sharded safetensors: accetta weight_map HF + formato interno | src/io/sharding.cpp | DONE |
| F6.20 | coverage counting in load_one_file_into_model; single-file >=90%, dir aggregato | src/io/hf_llama_loader.cpp | DONE |
| F6.17 | NON-BUG: file_type=1 = MOSTLY_F16 nello enum GGUF, coerente coi tensori F16 | - | verified-ok |
| downloader traversal | is_safe_repo_path | src/hub/downloader.cpp | [user] done |
| GGUF nested KV depth cap | skip_gguf_value | src/io/gguf_loader.cpp + test | [user] done |
| server payload max + embeddings cap | server.cpp | src/server/server.cpp | [user] done |
