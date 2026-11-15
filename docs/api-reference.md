# API Reference

OpenAI-compatible HTTP API reference.

## Base URL

```
http://localhost:8080
```

## Endpoints

> **Web UI**: `GET /` serve l'interfaccia chat (se `--ui` punta a una directory valida).
>
> **Multi-modello**: il server può tenere più modelli caricati contemporaneamente. Il campo `model` nelle request viene risolto per nome; un nome sconosciuto restituisce `404 {"error":{"message":"model not found: ...","type":"invalid_request_error"}}`. `model` vuoto/assente usa il modello primario.

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen2.5-0.5B-Instruct"
}
```

### `GET /metrics`

Prometheus-format metrics.

**Response:**

```
# TYPE serius_requests_total counter
serius_requests_total 42
# TYPE serius_tokens_generated counter
serius_tokens_generated 1337
```

### `GET /v1/models`

List available models.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1709654400,
      "owned_by": "serius",
      "type": "llama"
    },
    {
      "id": "smollm2",
      "object": "model",
      "created": 1709654401,
      "owned_by": "serius",
      "type": "llama"
    }
  ]
}
```

### `GET /v1/models/{id}`

Get model details.

**Response:**

```json
{
  "id": "Qwen2.5-0.5B-Instruct",
  "object": "model",
  "created": 1709654400,
  "owned_by": "serius"
}
```

### `POST /v1/models/load`

Load a model at runtime (parallel to ongoing inference).

**Request:**

```json
{
  "name": "qwen-small",
  "path": "models/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
  "type": "llama"
}
```

| Field | Type | Description |
|-------|------|-------------|
| name | string | Optional model ID; derived from path if omitted |
| path | string | **Required.** Local dir / config.json / `.gguf`, or `hf://org/repo` |
| type | string | `llama` or `transformer` (optional, auto-detected; `hf://` forces `llama`) |
| quant | string | GGUF quant tag for `hf://` models (e.g. `q4_k_m`) |

**Response:** the created model object (same shape as `GET /v1/models/{id}`). Errors: `400` missing/invalid path or unsupported type, `409` name already loaded.

### `DELETE /v1/models/{id}`

Unload a model, freeing weights, KV cache and VRAM. Waits for any in-flight generation on that model. `404` if unknown.

### `POST /v1/chat/completions`

Generate chat completion.

**Request:**

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "stream": false
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| model | string | primary | Model ID (must be loaded; empty → primary) |
| messages | array | required | Array of {role, content} messages |
| temperature | float | 0.7 | Sampling temperature (0.0-2.0) |
| max_tokens | int | 256 | Maximum tokens to generate |
| top_p | float | 0.9 | Nucleus sampling (0.0-1.0) |
| top_k | int | 40 | Top-k sampling (0 = disabled) |
| stream | bool | false | Enable SSE streaming |

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709654400,
  "model": "Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  }
}
```

**Response (streaming):**

```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"!"},"index":0}]}

data: [DONE]
```

### `POST /v1/completions`

Text completion (legacy format).

**Request:**

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "prompt": "The capital of France is",
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false
}
```

**Response:**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1709654400,
  "model": "Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "text": " Paris.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 2,
    "total_tokens": 10
  }
}
```

### `POST /v1/embeddings`

Generate embeddings for text.

**Request:**

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "input": ["Hello world", "Test sentence"]
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, 0.3, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.4, 0.5, 0.6, ...],
      "index": 1
    }
  ],
  "model": "Qwen2.5-0.5B-Instruct",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## Example: Using curl

```bash
# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```
