# Supported Models

List of compatible model architectures and quantization formats.

## Supported Architectures

| Architecture | Status | Notes |
|--------------|--------|-------|
| LLaMA 2/3 | ✅ Fully Supported | Including GQA, RoPE, SwiGLU |
| Qwen 2/2.5 | ✅ Fully Supported | Including attention variants |
| Generic Transformer | ✅ Supported | Configurable via JSON |
| MoE (Mixture of Experts) | ✅ Supported | DeepSeekMoE, Mixtral |

## Loading Formats

| Format | How | Status |
|--------|-----|--------|
| GGUF (.gguf) | `--model model.gguf` or `POST /v1/models/load` | ✅ Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K (decode packed + dense prefill) |
| Safetensors dir | `--model <dir>` (config.json + weights) | ✅ |
| HuggingFace | `--model hf://org/repo --quant q4_k_m` | ✅ GGUF download at runtime |
| JSON config | `--type transformer --config ...` | ✅ |

GGUF note: decode uses packed quantized weights (fast GEMV); batched prefill uses dense FP32 dequantized weights. Enable `SERIUS_PREFILL_BATCH=1` for faster prompt processing — outputs can differ slightly from the decode path on quantized models. Supported quantized types: F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q4_K, **Q5_K**, Q6_K.

## Embedding Models

The Hub `EmbeddingManager` runs a real BERT-style encoder for embedding GGUFs:

- **nomic-bert** (e.g. `nomic-embed-text-v1.5.Q4_K_M.gguf`) — ✅ full 12-layer encoder: WordPiece tokenizer, non-causal self-attention + RoPE, GELU FFN, mean pooling → 768-dim normalized embeddings. Quantized tensors Q4_K/Q5_K/Q6_K are dequantized on load. Projections run through `blas::matmul` (AVX2/AVX512, CUDA when dispatched) with weights pre-transposed at load: ~0.84s/embedding on CPU. Retrieval uses the model's task prefixes (`search_query:`/`search_document:`).
- Other GGUFs → fallback: mean-pooled `token_embd.weight` (weak semantics — for LLM token embeddings only).
- No usable model → deterministic hash embedding (word unigrams + signed char 3–5-grams, L2-normalized).

## Quantization Formats

| Format | VRAM Savings | Quality | Status |
|--------|-------------|---------|--------|
| FP16 | 0% | Baseline | ✅ Supported |
| FP8 | 50% | Excellent | ✅ Supported |
| INT8 | 50% | Good | ✅ Supported |
| INT4 (AWQ/GPTQ) | 75% | Acceptable | ✅ Supported |

## Model Size Guide

| Parameters | FP16 VRAM | INT4 VRAM | Use Case |
|------------|-----------|-----------|----------|
| 0.5B | 1 GB | 0.5 GB | Testing, embedded |
| 1B | 2 GB | 1 GB | Lightweight apps |
| 3B | 6 GB | 2 GB | General purpose |
| 7B | 14 GB | 4 GB | Desktop, small server |
| 14B | 28 GB | 8 GB | Server, GPU 24GB+ |
| 70B | 140 GB | 40 GB | Enterprise, multi-GPU |

## Recommended Models

### Small Models (0.5B - 3B)

- **Qwen2.5-0.5B-Instruct**: Fast, good quality
- **Qwen2.5-3B-Instruct**: Balanced speed/quality
- **Phi-3-mini**: 3.8B, excellent reasoning

### Medium Models (7B - 14B)

- **Qwen2.5-7B-Instruct**: Strong general purpose
- **Llama-3-8B-Instruct**: Industry standard
- **Mistral-7B-Instruct**: Good performance

### Large Models (30B+)

- **Qwen2.5-32B-Instruct**: High quality
- **Llama-3-70B-Instruct**: Best quality (requires multi-GPU)

## Conversion

Convert models from HuggingFace to Serius format:

```bash
serius import --source hf://Qwen/Qwen2.5-0.5B-Instruct --output ./models/
```

See [CLI Reference](cli-reference.md) for details.
