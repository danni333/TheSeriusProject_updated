# Changelog

## Unreleased

- Integrate upstream changes and the Fase 9/10 feature work.
- Resolve build integration, CPU backend, and GGUF loader compatibility issues.
- Fix DataLoader batch generation and padding for incomplete sequences.
- Use compiler-specific optimization/debug flags and valid MSVC warning options in CMake.
- Validate Flash Attention inputs, configure forward tile sizes, benchmark CUDA configurations, and test numerical parity.
- Bootstrap NCCL unique IDs on rank 0 and distribute them across initialized MPI ranks.
- Validate the merged Debug build and CTest suite (200 tests, 0 failures).
