# Audit Stub, Placeholder e Gap — 2026-11-22

Audit completo del codice sorgente (`src/`, ~50k righe C++, 505 file) con focus su
stub, placeholder, mock, codice morto e funzionalità non operative.

**Metodo**: grep sistematico per marker (`TODO`, `FIXME`, `stub`, `mock`,
`placeholder`, `simplified`, `not implemented`, `for now`, `#if 0`), verifica
manuale dei file sospetti, controllo CMake (cosa è davvero compilato), e
confronto con `MISSING_PARTS_REPORT.md` e `docs/competitive-gap-and-status.md`.

---

## 1. Cosa funziona davvero (verificato)

- **Nucleo matematico**: `tensor/`, `blas/` (scalar, AVX2, AVX-512, quantized GEMV),
  `autograd/`, `memory/` — implementazioni reali e testate.
- **Transformer**: `transformer/` (attention, FFN, LayerNorm, RoPE, KV cache,
  paged KV), `infer/` (LLaMA inference CPU+CUDA, GPU offload, speculative),
  `attention/` (flash attention CPU+CUDA).
- **I/O modelli**: GGUF loader con pesi packed quantizzati (Q4_0/Q4_1/Q8_0/
  Q4_K/Q5_K/Q6_K/Q5_0/BF16), safetensors + HF LLaMA loader, GGUF writer/export.
- **Server HTTP**: `/health`, `/metrics`, `/v1/models`, `POST /v1/models/load`,
  `DELETE /v1/models/{id}`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings` — con inferenza reale, SSE streaming, KV cache persistente
  multi-turn e web UI (`ui/`). **Non più stub** (la gap analysis di ottobre
  è superata).
- **CLI**: `serve`, `chat`, `generate`, `convert`, `import`, `doctor`, `bench`,
  `stress`, `mcp`.
- **Distributed**: local/data/pipeline/tensor parallel, NCCL+MPI opzionali con
  bootstrap NCCL via MPI.
- **MCP**: client, transport stdio+HTTP, guard con risk level, audit, scrubber.
- **Hub**: model hub server con auth file-based, search, metadata, downloader.

---

## 2. Stub e placeholder confermati

| File | Problema | Impatto |
|------|----------|---------|
| `src/hal/vulkan_backend.cpp` | `malloc`/`free`/copy TODO; `is_available()` ritorna `false` fisso; manca `malloc_pinned`/`malloc_managed`/`free_pinned` → la classe resta **astratta, non compila** se `SERIUS_USE_VULKAN=ON` | Backend Vulkan inutilizzabile |
| `src/hal/metal_backend.cpp` | Stessi TODO di Vulkan; disponibile solo su macOS/Metal | Backend Metal inutilizzabile |
| `src/optim/graph_compiler.cpp` | `trace(void*)` emetteva una sequenza **sintetica** (matmul→bias→gelu) solo per esercitare le ottimizzazioni; `ExecutionPlan::execute` copiava input→output — **RISOLTO 2026-11-22**: ora lancia `runtime_error` sulle op senza `forward_fn` (l'esecuzione reale è il replay autograd). Documentato in header. **Aggiornato 2026-11-23**: `trace(void*,shape)` rimosso del tutto (`void*` non può portare un forward callable); i test usano `trace_graph(Node*)` su grafi autograd reali; `fuse_elementwise_chains` fonde ora `add(bias rank-1)→gelu` reali in `fused_bias_gelu` | Onesto: niente esecuzione fasulla; `trace_graph(Node*)` è l'unico tracing |
| `src/bench/latency.cpp` | "realistic **simulated** metrics" anche con modello caricato — **RISOLTO 2026-11-22**: `BenchModel`/`load_bench_model`/`bench_decode` eseguono inferenza reale (GGUF/safetensors, tokenizer JSON o GGUF metadata, KV cache, sampler, TTFT/prefill/decode TPS) e restituiscono 0 se il modello non è caricato. **Verificato su SmolLM2-135M Q4_0**: GPU auto-offload ~595 tok/s decode, `SERIUS_NO_GPU=1` ~5 tok/s CPU | Benchmark latency reale |
| `src/agent/react.cpp` + `model_interface.h` | Tool result da **mappa precompilata (mock)**; nessun tool reale eseguito — **RISOLTO 2026-11-22**: `ReActEngine::run` accetta ora un `tools::ToolRegistry*` e munisce i tool registrati per davvero | ReAct può usare tool reali |
| `src/agent/` (tutto) | `agent::Model` era interfaccia astratta **senza adapter** ai modelli reali — **RISOLTO 2026-11-22**: nuovo `agent/llama_model_adapter.{h,cpp}` (`LlamaAgentModel`) che wrappa `LLaMAModel` + `BPETokenizer` e gira il path di inference di produzione (KV cache + sampler). Planner, Supervisor, Consensus possono ora usare modelli reali | Stack agent collegato al modello reale |
| `src/autoopt/refactorer.cpp` | Non sapeva eseguire il codice generato — **RISOLTO 2026-11-22**: `optimize_with_llm` ora carica la libreria compilata dal sandbox via `Sandbox::load_kernel` (dlopen/LoadLibrary, simbolo `kernel_entry` con ABI C) e la passa attraverso la stessa pipeline verify→benchmark→hot-swap di `optimize_hotspot` | Auto-optimize LLM end-to-end funzionale |
| `src/rag/embeddings.cpp` | Embedding **hash-based** "for testing" — **MITIGATO 2026-11-22**: `RAGRetriever::set_embedder(EmbedFn)` accetta backend reali (EmbeddingManager/NomicBert/remote); l'hash resta fallback onesto e dimensioni non corrispondenti ricadono sul fallback | RAG può usare embedding reali |
| `src/distributed/pipeline.cpp` | Ultimo stage senza `loss_gradients` usava **unit gradient** — **RISOLTO 2026-11-22**: ora lancia `invalid_argument` se mancano i gradienti reali | Backward PP onesto |
| `src/hub/downloader.cpp` | Progress callback `0/0`; `download_hf` con file list hardcoded — **RISOLTO 2026-11-22**: progresso reale (HEAD content-length + polling dimensione file durante download async); `download_hf` usa `list_hf_files` (API tree) e scarica config/tokenizer/safetensors | Download HF reale con progresso |
| `src/mcp/transport_http.cpp` | HTTPS non supportato senza OpenSSL | Solo HTTP/stdio in pratica |
| `src/hub/CMakeLists.txt` | `auth_manager.cpp`/JWT **disabilitati** (richiede OpenSSL) | Niente JWT; esiste auth file-based alternativa in `model_hub_server.cpp` |
| `src/inference/speculative.cpp` | File da **1 riga** (solo `#include`) — **RIMOSSO 2026-11-22** | — |
| `src/model/moe.cpp` | File da **1 riga** (solo `#include "moe.h"`) — **RIMOSSO 2026-11-22** | — |
| `src/core/dummy.cpp` | `dummy_function() { return 42; }` — **RIMOSSO 2026-11-22** (con `tests/test_dummy.cpp`) | — |
| `src/app/main.cpp` | Vecchia CLI da 412 righe, non referenziata in CMake — **RIMOSSA 2026-11-22** | — |

---

## 3. Implementazioni "giocattolo" compilate nel core

Tutte compilate in `serius_core` e coperte da test, ma a scala didattica:

| Directory | Righe | Nota |
|-----------|-------|------|
| `src/quantum/` | ~520 | Simulatore VQE con parameter-shift gradient; nessuna esecuzione quantistica reale |
| `src/liquid/` | ~370 | Neural ODE / LTC approssimati |
| `src/world/` | ~360 | World model / JEPA / imagination — prototype |
| `src/neuro/` | ~500 | SNN con LIF neuron, spiking layer, event engine |
| `src/ssm/` | ~110 | Mamba-like block minimale (proiezione, ricorrenza) |
| `src/grounding/` | ~200 | Citation engine, confidence, self-consistency |
| `src/eval/` | ~210 | Eval suite, perplexity, metrics, report |
| `src/federated/` | ~225 | DP noise, aggregator, user adapter |
| `src/multimodal/`, `src/vision/`, `src/audio/` | ~1.3k | Encoder didattici via autograd; nessun weight HF reale caricato |
| `src/reasoning/` | ~400 | CoT, MCTS, verifier, budget — logica su `Model` mock |
| `src/rlhf/` | ~540 | Reward model: `score()`/`compute_logprobs()` reali (forward autograd + testa scalare); `generate_with_logprobs` **ora usa decode reale** (`llama_forward_single_token` + KV cache, argmax greedy) — RISOLTO 2026-11-22. Resta: PPO con KL "same as old for now" |
| `src/interpret/` (ora `experimental/interpret/`) | ~800 | **Parzialmente reale**: `infer::ActivationHooks` (hook globali opt-in) catturano hidden states post-layer e distribuzioni attention softmaxate su CPU **e CUDA eager** (prefill batched + decode); `InferenceCapture` le memorizza; `LogitLens` + `logit_lens_head` producono predizioni reali; `serius inspect` le stampa (verificato `--gpu cuda` su modello reale). Limiti: CUDA-graph decode bypassato con hook attivi; attention_row assente nel branch fused-GQA FP16 |
| `src/knowledge/` | ~610 | Semantic hash, fact extractor |
| `src/automl/` | ~450 | Auto-tune, pruner, tune cache |

**Problema strutturale**: la roadmap `todo/` (74 fasi) marca quasi tutto `[x]`,
ma queste aree sono scaffolding. Il tracker non distingue "implementato" da
"implementato a scopo dimostrativo".

**Aggiornamento 2026-11-22**: tutti i moduli di questa sezione sono ora
etichettati `// EXPERIMENTAL` negli header e documentati in
`docs/experimental.md`. `rlhf` è parzialmente reale (decode con KV cache);
`eval` è in promozione a modulo reale (PerplexityRunner);
`interpret` è parzialmente reale (hook di attivazione + `serius inspect`,
CPU-only).

---

## 4. Backend hardware e distributed

- **CUDA**: `bench_results.txt` (marzo 2026) mostra `illegal memory access` a
  `src/hal/cuda_backend.cu:41` — **vecchio di 8 mesi, da riverificare** sul build
  attuale.
- **Vulkan/Metal**: vedi tabella stub — non compilano se abilitati.
- **NCCL**: richiede MPI per bootstrap automatico; senza MPI serve ID manuale.
- **MPI**: usa `MPI_COMM_WORLD` hardcoded, rank/size dal runtime.
- **GGUF**: dtype non supportati (Q2_K, Q3_K, IQ1–IQ4, TQ, ecc.) vengono
  **skippati con warning su stderr** — se un tensore critico ha un quant
  mancante, il modello si carica parziale/zconst. Verificare che il loader
  fallisca in modo esplicito per tensori mancanti.
- **`serius serve`**: `--type qwen` non supportato (`cmd_serve.cpp:107`).

---

## 5. Testing e CI

- **`test_cuda`, `test_distributed_llama`, `test_pipeline_llama`**: ~~mai registrati~~ — **REGISTRATI 2026-11-22** (`test_cuda` via `add_test` perché ha main custom non-gtest; gli altri via `gtest_discover_tests`). La registrazione ha **esposto bug reali preesistenti** (vedi sotto).
- **Bug CUDA reale — RISOLTO 2026-11-22**: la root cause non era in `matmul_cuda.cu` ma in `Tensor::free_data()` (`src/tensor/tensor.cpp`), che liberava `data_gpu_` **incondizionatamente** ignorando `owns_data_`. Le view non-mowners (`gpu_view`, `host_gpu_view`, `reshape`/`slice`/`permute`/`contiguous` alias) condividevano il puntatore GPU del proprietario e ogni temporanea eseguita `cudaFree` sul buffer altrui → use-after-free + double-free. I kernel cuBLAS scrivevano poi in memoria già riassegnata (OOB apparente) e `cudaFree` falliva con "invalid argument". Fix: nuovo flag `owns_gpu_` settato solo dove il tensor alloca davvero il buffer device (copy ctor, `gpu_only`, `syncToDevice`, `alloc_gpu`, `contiguous()` copy path); `free_data()` libera solo se `owns_gpu_`. Anche `expand()` ora propaga `data_gpu_` (prima la perdeva) e `syncToHost()` marca `owns_data_` per il buffer CPU allocato. Verificato: `compute-sanitizer --tool memcheck` riporta **0 errori** su `PrefillThenDecodeParity` + `SingleTokenWithKVCacheParity` (prima: decine di invalid write/free).
- **Bug distributed reale — RISOLTO 2026-11-22**: in `compute_head_attention` (`src/infer/llama_inference.cpp`), `npos = pos_offset + qi + 1` veniva usato anche **senza KV cache**: `key_buf`/`value_buf` contengono solo `seq_len` righe, quindi per `pos > 0` l'attenzione leggeva posizioni inesistenti (out-of-bounds → NaN nei logits). Fix: senza cache `npos = qi + 1` (solo le righe presenti nel buffer). Entrambi i test TP/PP equivalence ora passano.
- **Flaky sotto `ctest -j8`**: `E2EServer.*`, `PersistentMemoryTest.BuildUserContext`, `ModelHubServer.HealthAndEmptyList` falliscono per saturazione CPU/timeout — **passano tutti se eseguiti seriali**.
- **`bench_results.txt`, `matmul_bench.txt`, `out.txt`, `pp_test_*.txt`**: log di
  bench/build **committati** nella root (datati marzo 2026). `.gitignore` copre
  `.obj`/`.exe`/`build*/` ma non questi `.txt`; la root ha ~54 artefatti locali.
- **Nessun benchmark serio vs llama.cpp**: i bench misurano matmul isolato
  (~307 GFLOPS su 1024×1024 AVX-512); manca token/s su un 7B quantizzato.
- **CI**: `.github/workflows/ci.yml` esiste (CPU + job GPU self-hosted).

---

## 6. Debito tecnico minore

- `src/inference/` vs `src/infer/`: due path di inferenza paralleli (quello
  `inference/` usa l'autograd `TransformerModel`, quello `infer/` è la strada
  GGUF/produzione). `inference/speculative.cpp` è vuoto.
- Commenti stale in `src/transformer/ffn.cpp:78-91` ("we will add this to
  ops.h next") — il codice funziona (`fused_bias_gelu` esiste), ma i commenti
  descrivono un piano vecchio.
- `src/tensor/tensor.cpp` lancia `runtime_error` per dtype non supportati in
  `item()`/`set_item()`/conversion — ok, ma verificare copertura DType.
- `src/cli/cmd_chat.cpp`/`cmd_serve.cpp`: `hf://` resolve funziona, ma il
  progress callback del downloader è fittizio (0/0).
- `docs/competitive-gap-and-status.md` (2026-10-18) e `MISSING_PARTS_REPORT.md`
  sono **disallineati**: dicono ancora "chat stub", "GGUF pesi no", "rotte
  mancanti" — ora chat/completions/embeddings esistono e GGUF carica pesi.

---

## 7. Priorità consigliate

1. **Allineare o rimuovere le aree "giocattolo"**: o le si marca chiaramente
   come `experimental/didactic` nel CMake/docs, o si tolgono dal build di
   default. Oggi gonfiano `serius_core` e il tracker fasi.
2. ~~**Adapter `agent::Model` → modello reale**~~ — **FATTO** (`LlamaAgentModel`).
3. ~~**Esecuzione tool reali** in `react.cpp`~~ — **FATTO** (`ToolRegistry` in `ReActEngine::run`).
4. ~~**Fix o rimozione di `graph_compiler`, `autoopt`**~~ — **FATTO 2026-11-22**:
   `ExecutionPlan::execute` ora fallisce onestamente; `optimize_with_llm`
   esegue davvero le librerie compilate via `Sandbox::load_kernel`.
5. ~~**Registrare i 3 test mancanti in ctest**~~ — **FATTO**; ha esposto 2 bug
   CUDA e 2 bug distributed reali — **entrambi RISOLTI** (vedi sezione 5).
6. ~~**Verificare il CUDA illegal memory access**~~ — **FATTO E RISOLTO**: era
   un bug di ownership in `Tensor::free_data()` (vedi sezione 5), non un
   problema di `matmul_cuda.cu`. Sanitizer pulito, test verdi.
7. **Benchmark vs llama.cpp** (token/s, Q4_K_M, stesso hardware) per avere un
   numero onesto di competitività.
8. ~~**Pulire codice morto**~~ — **FATTO parziale**: rimossi `src/core/`,
   `src/app/`, `model/moe.cpp`, `inference/speculative.cpp`, `test_dummy.cpp`.
   Restano i log `.txt` di build committati nella root (decidere se ripulire).
9. **Decidere Vulkan/Metal**: implementare o rimuovere i backend (oggi non
   compilano se attivati — trappola per chi abilita il flag).
10. ~~**RAG con embedding reale**~~ — **FATTO 2026-11-22**: `RAGRetriever::set_embedder`
    accetta `hub::EmbeddingManager`/NomicBert/endpoint remoto; hash = fallback.
11. ~~**Aree "ricerca" da etichettare onestamente**~~ — **FATTO 2026-11-22**:
    tutti i moduli didattici ha il marker `// EXPERIMENTAL` negli header e sono
    documentati in `docs/experimental.md`.
12. ~~**`eval/` con metrica reale**~~ — **FATTO 2026-11-22**: nuovo
    `eval/perplexity_runner.{h,cpp}` (`evaluate_perplexity`) calcola perplexity
    teacher-forced reale via `llama_forward_single_token` + KV cache, con
    chunking a contesto scorrevole; esposto come `serius eval --task ppl`.
    Loader modello CLI estratto in `cli/llama_loader` (condiviso da chat+eval).
13. ~~**`grounding/` confidence reale**~~ — **FATTO 2026-11-22**: `serius chat`
    raccoglie il logprob di ogni token emesso durante il decode e stampa
    `mean token confidence` via `grounding::confidence_from_logprobs`.
14. ~~**`knowledge/` pipeline reale**~~ — **FATTO 2026-11-22**: nuovo
    `knowledge/knowledge_base.{h,cpp}` collega FactExtractor → embedder
    pluggabile → KnowledgeFunction (LSH) → query con fact texts.
15. ~~**`reasoning/` su modello reale**~~ — **FATTO 2026-11-22**: nuovi adapter
    `reasoning/agent_model_adapter.h` (CoT + MCTS) verso `agent::Model`, e
    comando `serius think` end-to-end.
