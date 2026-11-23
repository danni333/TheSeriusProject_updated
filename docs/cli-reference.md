# CLI Reference

Complete reference for all Serius CLI commands.

## Commands

### `serius chat`

Interactive chat with the model.

```bash
serius chat [OPTIONS]
```

**Options:**

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Path to model file | `./models/model.gguf` |
| `--threads` | `-t` | Number of threads | `4` |
| `--temperature` | | Sampling temperature | `0.7` |
| `--max-tokens` | | Maximum tokens to generate | `256` |

**Example:**

```bash
serius chat --model ./models/Qwen2.5-0.5B-Instruct --temperature 0.8
```

### `serius serve`

Start HTTP API server with OpenAI-compatible endpoints and the web UI.

```bash
serius serve [OPTIONS]
```

**Options:**

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Model path: dir, config.json, `.gguf`, or `hf://repo_id` | none |
| `--quant` | | GGUF quant tag for `hf://` models (e.g. `q4_k_m`) | auto |
| `--type` | | Model type: `llama` or `transformer` | `llama` |
| `--name` | | Model name reported by `/v1/models` | derived |
| `--port` | `-p` | Server port | `8080` |
| `--host` | | Server host | `0.0.0.0` |
| `--max-tokens` | | Default max tokens per request | `256` |
| `--temperature` | | Default sampling temperature | `0.7` |
| `--top-k` | | Default top-k sampling (0 = disabled) | `40` |
| `--top-p` | | Default top-p (nucleus) sampling | `0.9` |
| `--gpu` | | GPU backend (`auto`/`cpu`/`cuda`) | `auto` |
| `--ui` | | Web UI directory | `ui` |

More models can be loaded/unloaded at runtime via `POST /v1/models/load` and `DELETE /v1/models/{id}`.

**Examples:**

```bash
serius serve --model ./models/Qwen2.5-0.5B-Instruct --port 8080
serius serve -m models/model-q4_k_m.gguf --type llama -p 8080 --ui ui
serius serve -m hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q4_k_m --name qwen-small
serius serve   # no model: load everything later via API/UI
```

### `serius import`

Import model from HuggingFace Hub or direct URL.

```bash
serius import [OPTIONS]
```

**Options:**

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--source` | `-s` | Source (hf://repo_id or URL) | *required* |
| `--output` | `-o` | Output directory | `./models` |
| `--token` | `-t` | HuggingFace token (optional) | `` |
| `--format` | `-f` | Output format (gguf, safetensors) | `gguf` |

**Examples:**

```bash
# Import from HuggingFace
serius import --source hf://Qwen/Qwen2.5-0.5B-Instruct --output ./models/

# Import from URL
serius import --source https://example.com/model.gguf --output ./models/
```

### `serius generate`

Generate text from a single prompt (non-interactive).

```bash
serius generate -m <model> -p "Your prompt" [OPTIONS]
```

### `serius convert`

Convert between GGUF and HuggingFace safetensors formats (both directions).

```bash
# GGUF -> HF directory (safetensors + config.json)
serius convert -i <model.gguf> -o <output_dir>

# HF dir / .safetensors -> single .gguf (FP16 tensors, tokenizer embedded)
serius convert -i <model_dir_or_model.safetensors> -o <out.gguf>
serius convert -i models/qwen -o qwen.gguf --f32        # FP32 tensors
serius convert -i models/qwen -o qwen.gguf --no-tokenizer
```

Input may be a directory with `config.json` + `model.safetensors` (or a
sharded `*.safetensors.index.json`), or a `.safetensors` file directly.

### `serius bench`

Run performance benchmarks (matmul, memory, mixed precision, …).

```bash
serius bench [OPTIONS]
```

### `serius eval`

> **Sperimentale**: richiede la build con `-DSERIUS_EXPERIMENTAL_EVAL=ON`.
> Vedi [experimental.md](experimental.md).

Evaluate a model. Currently supports `--task ppl` (perplexity): teacher-forced
scoring through the real inference path (KV cache + `llama_forward_single_token`).

```bash
serius eval --model models/smollm2-q4 --file corpus.txt
serius eval --model models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --text "Some text to score" --max-tokens 1000 --context 512 --gpu cpu
```

Options: `--model/-m` (path or `hf://repo`), `--task/-t` (`ppl`), `--file/-f`
(corpus file), `--text/-x` (inline text), `--context/-c` (tokens per forward,
default 512), `--max-tokens/-n` (cap scored tokens, 0 = all), `--gpu/-g`
(auto/cpu/cuda), `--quant` (GGUF tag for `hf://`).

### `serius inspect`

> **Sperimentale**: richiede `-DSERIUS_EXPERIMENTAL_INTERPRET=ON`.

Per-layer interpretability on a real prompt: runs one instrumented forward
pass (`infer::ActivationHooks`, CPU path only) and reports, for every
transformer layer, the top-k next-token predictions via logit lens
(`interpret::LogitLens` on the captured hidden state + real LM head) and
attention statistics (entropy + top attended positions, averaged over heads).

```bash
serius inspect --model models/smollm2-q4 --prompt "The capital of France is"
```

Options: `--model/-m`, `--prompt/-p` (required), `--topk/-k` (lens top-k,
default 3), `--attn-top/-a` (top attended positions, default 3), `--gpu/-g`
(default `cpu` — CUDA kernels do not emit activation hooks), `--quant`.

### `serius think`

> **Sperimentale**: richiede `-DSERIUS_EXPERIMENTAL_REASONING=ON`.

Chain-of-thought reasoning: the model first produces a hidden "thinking"
buffer, then the final answer conditioned on it. Uses the real inference path
(`LlamaAgentModel` → `reasoning::AgentReasoningModel` → `ChainOfThought`).

```bash
serius think --model models/smollm2-q4 --prompt "What is 2+2?" \
    --max-thinking-tokens 512 --show-thinking --gpu cpu
```

Options: `--model/-m`, `--prompt/-p` (required), `--max-thinking-tokens`,
`--show-thinking` (print the thinking buffer), `--gpu/-g`, `--quant`.

### `serius stress`

Run stress tests against the server.

```bash
serius stress [OPTIONS]
```

### `serius mcp-serve`

Serve MCP (Model Context Protocol) tools to external clients. Default transport is stdio (newline-delimited JSON-RPC); `--http PORT` exposes `POST /mcp` instead.

```bash
serius mcp-serve [--audit PATH] [--http PORT]
```

Built-in tools: `echo`, `current_time`, `serius_info`. Every `tools/call` is recorded in the audit log (`mcp_audit.log` by default, override with `--audit`).

```bash
serius mcp-serve                          # stdio mode (for MCP clients)
serius mcp-serve --http 8091              # HTTP mode: POST http://host:8091/mcp
serius mcp-serve --audit /var/log/mcp.log
```

### `serius doctor`

System diagnostics and deployment readiness check.

```bash
serius doctor [--model <path>] [--port <n>] [--json]
```

| Flag | Description |
|---|---|
| `-m, --model` | Inspect a model file (`.gguf`) or directory (`config.json` + weights): file size, GGUF metadata, architecture, exact parameter count and bits/weight, and whether it fits in VRAM or RAM |
| `--port` | Check whether a TCP port (e.g. `8080`) is free on localhost for `serius serve` |
| `--json` | Emit the report as JSON |

**Output:**

```
Serius System Diagnostics
========================

Version: 0.1.0

System:
✅ CPU: 16 logical cores
✅ RAM: 23.8 GiB total, 13.8 GiB available
✅ Disk: 88.3 GiB free in current directory
✅ GPU: CUDA backend available — NVIDIA GeForce RTX 3060 Laptop GPU (1 device, 6.0 GiB VRAM, 5.0 GiB free)

Build:
ℹ️  Flags: C++20, CUDA enabled, NCCL disabled, MPI disabled

Readiness:
ℹ️  GPU budget: GPU can comfortably run up to ~8.6B params (Q4-class). ...
ℹ️  CPU budget: with 12.9 GiB free RAM, CPU inference suits up to ~9.2B params ...

Model:
✅ File: qwen2.5-0.5b-instruct-q4_k_m.gguf (469 MiB)
✅ GGUF: version 3, 291 tensors, 26 metadata entries
ℹ️  Architecture: qwen2
ℹ️  Dimensions: vocab=32000, hidden=896, layers=24, heads=14/2 (q/kv), ffn=4864, ctx=32768
ℹ️  Weights: ~630M params (exact, from tensor shapes), payload 463 MiB, ~6.1 bits/weight → Q4-class quantization
✅ Fit: file fits in GPU VRAM (6.0 GiB)

Network:
✅ Port: 8080 is free on 127.0.0.1 — ready for `serius serve`
```

### `serius --help`

Show help message.

```bash
serius --help
```

### `serius --version`

Show version information.

```bash
serius --version
```
