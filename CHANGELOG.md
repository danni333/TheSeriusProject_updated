# Changelog

## Unreleased

- Web UI (`ui/`) served by `serius serve --ui`: streaming chat, model management panel, localStorage persistence.
- Multi-model runtime registry: `POST /v1/models/load` (local path or `hf://`), `DELETE /v1/models/{id}`, parallel loads, per-model KV cache.
- Persistent KV-cache prefix reuse for multi-turn chat; bounded history in UI payload.
- Fixed degenerate batched prefill: Tensor host/device sync bugs (`syncToHost` on read accessors, GPU invalidation on CPU writes). Batched prefill now opt-in via `SERIUS_PREFILL_BATCH=1`.
- Real BOS/EOS ids from GGUF metadata; correct stop on `<|im_end|>`.
- Integrate upstream changes and the Fase 9/10 feature work.
- Resolve build integration, CPU backend, and GGUF loader compatibility issues.
- Fix DataLoader batch generation and padding for incomplete sequences.
- Use compiler-specific optimization/debug flags and valid MSVC warning options in CMake.
- Validate Flash Attention inputs, configure forward tile sizes, benchmark CUDA configurations, and test numerical parity.
- Bootstrap NCCL unique IDs on rank 0 and distribute them across initialized MPI ranks.
- Validate the merged Debug build and CTest suite (200 tests, 0 failures).
