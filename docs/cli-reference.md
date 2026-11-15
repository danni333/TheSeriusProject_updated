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

Convert a GGUF model to a safetensors directory.

```bash
serius convert -m <model.gguf> -o <output_dir>
```

### `serius bench`

Run performance benchmarks (matmul, memory, mixed precision, …).

```bash
serius bench [OPTIONS]
```

### `serius stress`

Run stress tests against the server.

```bash
serius stress [OPTIONS]
```

### `serius doctor`

System diagnostics.

```bash
serius doctor
```

**Output:**

```
Serius System Diagnostics
========================

✅ CPU: Detected
✅ RAM: Available
⚠️  GPU: Detection pending (CUDA/Metal support)
✅ Build: C++20 compatible
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
