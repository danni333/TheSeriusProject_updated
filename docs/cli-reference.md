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

Start HTTP API server with OpenAI-compatible endpoints.

```bash
serius serve [OPTIONS]
```

**Options:**

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Path to model file | `./models/model.gguf` |
| `--port` | `-p` | Server port | `8080` |
| `--host` | | Server host | `0.0.0.0` |
| `--gpu` | | Use GPU (auto/cpu/cuda) | `auto` |

**Example:**

```bash
serius serve --model ./models/Qwen2.5-0.5B-Instruct --port 8080
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
