// Prompt building, generation and SSE streaming helpers for the HTTP
// server — split from server.cpp. Internal helpers shared with the routes
// are declared in server_internal.h.
#include "server_internal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
namespace serius {
namespace server {
/// Decode one UTF-8 sequence at p[0..n).
/// Returns the sequence length (1-4) when valid and complete, 0 when invalid,
/// and -1 when the sequence is truncated by the end of the buffer but all
/// present bytes are valid continuations (may complete with more input).
int utf8_seq_len(const char *p, size_t n)
{
    const auto u = reinterpret_cast<const unsigned char *>(p);
    if (n == 0) {
        return 0;
    }
    size_t len;
    uint32_t cp;
    if (u[0] < 0x80) {
        return 1;
    }
    if ((u[0] & 0xE0) == 0xC0) {
        len = 2; cp = u[0] & 0x1F;
    } else if ((u[0] & 0xF0) == 0xE0) {
        len = 3; cp = u[0] & 0x0F;
    } else if ((u[0] & 0xF8) == 0xF0) {
        len = 4; cp = u[0] & 0x07;
    } else {
        return 0;
    }
    if (n < len) {
        for (size_t j = 1; j < n; ++j) {
            if ((u[j] & 0xC0) != 0x80) {
                return 0;
            }
        }
        return -1;
    }
    for (size_t j = 1; j < len; ++j) {
        if ((u[j] & 0xC0) != 0x80) {
            return 0;
        }
        cp = (cp << 6) | (u[j] & 0x3F);
    }
    if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
        (len == 4 && cp < 0x10000) || cp > 0x10FFFF ||
        (cp >= 0xD800 && cp <= 0xDFFF)) {
        return 0;
    }
    return static_cast<int>(len);
}

/// Replace every invalid UTF-8 byte sequence with U+FFFD so the result is
/// always valid UTF-8 (required by nlohmann::json serialization).
std::string sanitize_utf8(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const int l = utf8_seq_len(s.data() + i, s.size() - i);
        if (l > 0) {
            out.append(s, i, static_cast<size_t>(l));
            i += static_cast<size_t>(l);
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}

/// Streaming variant: emit the longest prefix of `pending` made of complete
/// UTF-8 characters (invalid bytes become U+FFFD), keeping a possibly
/// incomplete trailing sequence in `pending` for the next call.
std::string utf8_stream_emit(std::string &pending)
{
    std::string out;
    size_t i = 0;
    while (i < pending.size()) {
        const int l = utf8_seq_len(pending.data() + i, pending.size() - i);
        if (l == -1) {
            break;
        }
        if (l == 0) {
            out += "\xEF\xBF\xBD";
            ++i;
            continue;
        }
        out.append(pending, i, static_cast<size_t>(l));
        i += static_cast<size_t>(l);
    }
    pending.erase(0, i);
    return out;
}

/// Flush any bytes still held back by utf8_stream_emit at end of stream.
std::string utf8_stream_flush(std::string &pending)
{
    return sanitize_utf8(pending);
}

std::string extract_user_prompt(const nlohmann::json &body)
{
    const auto messages = body.value("messages", nlohmann::json::array());
    if (!messages.is_array()) {
        return body.value("prompt", "Serius API is running. Load a model to get real completions.");
    }

    std::string last_user_text;
    for (const auto &msg : messages) {
        if (!msg.is_object()) {
            continue;
        }

        const std::string role = msg.value("role", "");
        if (role == "user") {
            const auto content = msg.value("content", nlohmann::json());
            if (content.is_string()) {
                last_user_text = content.get<std::string>();
            }
        }
    }

    if (!last_user_text.empty()) {
        return last_user_text;
    }

    for (const auto &msg : messages) {
        if (!msg.is_object()) {
            continue;
        }

        const auto content = msg.value("content", nlohmann::json());
        if (content.is_string()) {
            const std::string candidate = content.get<std::string>();
            if (!candidate.empty()) {
                return candidate;
            }
        }
    }

    return body.value("prompt", "Serius API is running. Load a model to get real completions.");
}

/// Extract plain text from a chat message content field (string or array of
/// content parts like OpenAI's [{"type":"text","text":"..."}]).
std::string message_text(const nlohmann::json &msg)
{
    const auto content = msg.value("content", nlohmann::json());
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_array()) {
        std::string out;
        for (const auto &part : content) {
            if (part.is_object() && part.value("type", "") == "text") {
                out += part.value("text", "");
            }
        }
        return out;
    }
    return "";
}

/// EOS_ID==2 is the GPT-2/LLaMA convention; GGUF models carry their real
/// eos in the tokenizer (Qwen ends on <|im_end|>, where id 2 is a normal
/// token). The <|im_end|> string check is kept as a safety net for ChatML
/// models whose metadata eos differs.
bool is_stop_token(const BPETokenizer &tokenizer, int next_id, const std::string &tok_str)
{
    return next_id == static_cast<int>(tokenizer.eos_id()) ||
           tok_str == "<|im_end|>";
}

size_t common_prefix_len(const std::vector<TokenId> &a, const std::vector<TokenId> &b)
{
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

/// Truncate every layer's KV cache to `len` entries. Entries beyond `len`
/// are simply ignored — attention only reads [0, current_len).
void truncate_kv_cache(infer::KVCache &cache, size_t len)
{
    for (auto &layer : cache.layers) {
        layer.current_len = len;
    }
}

/// Prefill `ids` into `cache`, reusing the longest prefix already present
/// (tracked via `kv_ids`, the token sequence the cache currently holds).
/// Only the suffix is forwarded, one token at a time — the batched prefill
/// path is disabled while it produces degenerate output, and the
/// single-token path dispatches to the CUDA decode graph for GPU models.
/// Returns the logits row for the last prompt position.
core::Tensor prefill_llama(const model::LLaMAModel &model,
                           const std::vector<TokenId> &ids,
                           infer::KVCache &cache,
                           std::vector<TokenId> *kv_ids)
{
    size_t prefix = 0;
    if (kv_ids && !kv_ids->empty() && !cache.layers.empty()) {
        prefix = common_prefix_len(ids, *kv_ids);
    }
    // Always forward at least the last prompt token: its logits are needed
    // to sample the first generated token, and they are not stored.
    const size_t start = std::min(prefix, ids.size() - 1);
    truncate_kv_cache(cache, start);

    core::Tensor last_logits({0});
    for (size_t i = start; i < ids.size(); ++i) {
        last_logits = infer::llama_forward_single_token(model, static_cast<int>(ids[i]), i, &cache);
    }
    return last_logits;
}

/// Build the full prompt for /v1/chat/completions from the messages array.
/// With chatml=true the whole transcript is rendered as ChatML turns
/// (<|im_start|>role ... <|im_end|>) ending with an open assistant turn, so
/// the model sees system prompt and prior history. Otherwise a plain-text
/// transcript is used. Falls back to the "prompt" field when no usable
/// messages are present.
std::string build_chat_prompt(const nlohmann::json &body, bool chatml)
{
    const auto messages = body.value("messages", nlohmann::json::array());
    if (!messages.is_array() || messages.empty()) {
        return body.value("prompt", "Serius API is running. Load a model to get real completions.");
    }

    std::string prompt;
    if (chatml) {
        for (const auto &msg : messages) {
            if (!msg.is_object()) {
                continue;
            }
            const std::string role = msg.value("role", "");
            if (role != "system" && role != "user" && role != "assistant") {
                continue;
            }
            const std::string text = message_text(msg);
            if (text.empty()) {
                continue;
            }
            prompt += "<|im_start|>" + role + "\n" + text + "<|im_end|>\n";
        }
        if (!prompt.empty()) {
            prompt += "<|im_start|>assistant\n";
        }
    } else {
        for (const auto &msg : messages) {
            if (!msg.is_object()) {
                continue;
            }
            const std::string role = msg.value("role", "");
            const std::string text = message_text(msg);
            if (text.empty()) {
                continue;
            }
            std::string label = role == "system" ? "System" : (role == "assistant" ? "Assistant" : "User");
            prompt += label + ": " + text + "\n";
        }
        if (!prompt.empty()) {
            prompt += "Assistant: ";
        }
    }

    if (prompt.empty()) {
        return body.value("prompt", "Serius API is running. Load a model to get real completions.");
    }
    return prompt;
}

GenerationResult generate_llama(const model::LLaMAModel &model,
                                BPETokenizer &tokenizer,
                                const std::string &prompt,
                                size_t max_new_tokens,
                                const infer::Sampler &sampler,
                                infer::KVCache &cache,
                                InferenceMetrics *metrics,
                                std::vector<TokenId> *kv_ids)
{
    GenerationResult res;
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty() || ids[0] != tokenizer.bos_id()) {
        ids.insert(ids.begin(), tokenizer.bos_id());
    }
    res.prompt_tokens = ids.size();

    core::Tensor logits = prefill_llama(model, ids, cache, kv_ids);
    const size_t vocab_size = logits.shape().back();

    int next_id = 0;
    if (!infer::sample_gpu_candidates(logits, 0, sampler, next_id)) {
        next_id = sampler.sample(logits);
    }

    for (size_t gen = 0; gen < max_new_tokens; ++gen) {
        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
        if (is_stop_token(tokenizer, next_id, tok_str)) {
            break;
        }
        res.text += tok_str;
        ++res.completion_tokens;
        if (metrics) {
            metrics->on_token_generated(0.0f);
        }

        if (gen + 1 == max_new_tokens) {
            break;
        }

        size_t pos = ids.size();
        ids.push_back(static_cast<TokenId>(next_id));
        core::Tensor next_logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
        if (!infer::sample_gpu_candidates(next_logits, 0, sampler, next_id)) {
            next_id = sampler.sample(next_logits);
        }
    }

    if (kv_ids) {
        *kv_ids = ids;
    }
    res.text = sanitize_utf8(res.text);
    return res;
}

void stream_llama(const model::LLaMAModel &model,
                  BPETokenizer &tokenizer,
                  const std::string &prompt,
                  size_t max_new_tokens,
                  const infer::Sampler &sampler,
                  infer::KVCache &cache,
                  const std::string &req_id,
                  const std::string &model_name,
                  httplib::DataSink &sink,
                  InferenceMetrics *metrics,
                  std::vector<TokenId> *kv_ids)
{
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty() || ids[0] != tokenizer.bos_id()) {
        ids.insert(ids.begin(), tokenizer.bos_id());
    }

    const auto created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    nlohmann::json intro = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", {{"role", "assistant"}}},
            {"finish_reason", nullptr}
        }}}
    };
    std::string intro_str = "data: " + intro.dump() + "\n\n";
    sink.write(intro_str.data(), intro_str.size());

    core::Tensor logits = prefill_llama(model, ids, cache, kv_ids);

    int next_id = 0;
    if (!infer::sample_gpu_candidates(logits, 0, sampler, next_id)) {
        next_id = sampler.sample(logits);
    }

    std::string pending;
    for (size_t gen = 0; gen < max_new_tokens; ++gen) {
        if (sink.is_writable && !sink.is_writable()) {
            break;
        }

        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
        if (is_stop_token(tokenizer, next_id, tok_str)) {
            break;
        }
        if (metrics) {
            metrics->on_token_generated(0.0f);
        }

        pending += tok_str;
        std::string emit = utf8_stream_emit(pending);
        if (!emit.empty()) {
            nlohmann::json chunk = {
                {"id", req_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model_name},
                {"choices", {{
                    {"index", 0},
                    {"delta", {{"content", emit}}},
                    {"finish_reason", nullptr}
                }}}
            };
            std::string chunk_str = "data: " + chunk.dump() + "\n\n";
            sink.write(chunk_str.data(), chunk_str.size());
        }

        if (gen + 1 == max_new_tokens) {
            break;
        }

        size_t pos = ids.size();
        ids.push_back(static_cast<TokenId>(next_id));
        core::Tensor next_logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
        if (!infer::sample_gpu_candidates(next_logits, 0, sampler, next_id)) {
            next_id = sampler.sample(next_logits);
        }
    }

    const std::string tail = utf8_stream_flush(pending);
    if (!tail.empty()) {
        nlohmann::json tail_chunk = {
            {"id", req_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"choices", {{
                {"index", 0},
                {"delta", {{"content", tail}}},
                {"finish_reason", nullptr}
            }}}
        };
        std::string tail_str = "data: " + tail_chunk.dump() + "\n\n";
        sink.write(tail_str.data(), tail_str.size());
    }

    nlohmann::json stop_chunk = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", nlohmann::json::object()},
            {"finish_reason", "stop"}
        }}}
    };
    std::string stop_str = "data: " + stop_chunk.dump() + "\n\n";
    sink.write(stop_str.data(), stop_str.size());

    const std::string done_str = "data: [DONE]\n\n";
    sink.write(done_str.data(), done_str.size());

    if (kv_ids) {
        *kv_ids = ids;
    }
}

GenerationResult generate_transformer(const model::TransformerModel &model,
                                      BPETokenizer &tokenizer,
                                      const std::string &prompt,
                                      size_t max_new_tokens,
                                      const infer::Sampler &sampler,
                                      InferenceMetrics *metrics)
{
    GenerationResult res;
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty()) {
        ids.push_back(tokenizer.bos_id());
    }
    res.prompt_tokens = ids.size();

    for (size_t gen = 0; gen < max_new_tokens; ++gen) {
        core::Tensor input_tensor({ids.size()});
        for (size_t i = 0; i < ids.size(); ++i) {
            input_tensor[i] = static_cast<float>(ids[i]);
        }

        core::autograd::Node input_node(input_tensor, false);
        core::autograd::Node *logits_node = model.forward(&input_node);
        const core::Tensor &logits = logits_node->data;

        size_t vocab_size = logits.shape().back();
        size_t last_row = logits.shape()[0] - 1;
        core::Tensor last_logits({vocab_size});
        const float *src = logits.data() + (last_row * vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            last_logits[i] = src[i];
        }

        int next_id = sampler.sample(last_logits);
        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
        if (is_stop_token(tokenizer, next_id, tok_str)) {
            break;
        }

        res.text += tok_str;
        ++res.completion_tokens;
        if (metrics) {
            metrics->on_token_generated(0.0f);
        }

        ids.push_back(static_cast<TokenId>(next_id));
    }

    res.text = sanitize_utf8(res.text);
    return res;
}

void stream_transformer(const model::TransformerModel &model,
                        BPETokenizer &tokenizer,
                        const std::string &prompt,
                        size_t max_new_tokens,
                        const infer::Sampler &sampler,
                        const std::string &req_id,
                        const std::string &model_name,
                        httplib::DataSink &sink,
                        InferenceMetrics *metrics)
{
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty()) {
        ids.push_back(tokenizer.bos_id());
    }

    const auto created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    nlohmann::json intro = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", {{"role", "assistant"}}},
            {"finish_reason", nullptr}
        }}}
    };
    std::string intro_str = "data: " + intro.dump() + "\n\n";
    sink.write(intro_str.data(), intro_str.size());

    std::string pending;
    for (size_t gen = 0; gen < max_new_tokens; ++gen) {
        if (sink.is_writable && !sink.is_writable()) {
            break;
        }

        core::Tensor input_tensor({ids.size()});
        for (size_t i = 0; i < ids.size(); ++i) {
            input_tensor[i] = static_cast<float>(ids[i]);
        }

        core::autograd::Node input_node(input_tensor, false);
        core::autograd::Node *logits_node = model.forward(&input_node);
        const core::Tensor &logits = logits_node->data;

        size_t vocab_size = logits.shape().back();
        size_t last_row = logits.shape()[0] - 1;
        core::Tensor last_logits({vocab_size});
        const float *src = logits.data() + (last_row * vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            last_logits[i] = src[i];
        }

        int next_id = sampler.sample(last_logits);
        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
        if (is_stop_token(tokenizer, next_id, tok_str)) {
            break;
        }
        if (metrics) {
            metrics->on_token_generated(0.0f);
        }

        pending += tok_str;
        std::string emit = utf8_stream_emit(pending);
        if (!emit.empty()) {
            nlohmann::json chunk = {
                {"id", req_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model_name},
                {"choices", {{
                    {"index", 0},
                    {"delta", {{"content", emit}}},
                    {"finish_reason", nullptr}
                }}}
            };
            std::string chunk_str = "data: " + chunk.dump() + "\n\n";
            sink.write(chunk_str.data(), chunk_str.size());
        }

        ids.push_back(static_cast<TokenId>(next_id));
    }

    const std::string tail = utf8_stream_flush(pending);
    if (!tail.empty()) {
        nlohmann::json tail_chunk = {
            {"id", req_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"choices", {{
                {"index", 0},
                {"delta", {{"content", tail}}},
                {"finish_reason", nullptr}
            }}}
        };
        std::string tail_str = "data: " + tail_chunk.dump() + "\n\n";
        sink.write(tail_str.data(), tail_str.size());
    }

    nlohmann::json stop_chunk = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", nlohmann::json::object()},
            {"finish_reason", "stop"}
        }}}
    };
    std::string stop_str = "data: " + stop_chunk.dump() + "\n\n";
    sink.write(stop_str.data(), stop_str.size());

    const std::string done_str = "data: [DONE]\n\n";
    sink.write(done_str.data(), done_str.size());
}

GenerationResult fallback_generation(const std::string &prompt)
{
    GenerationResult res;
    res.prompt_tokens = 0;
    res.completion_tokens = 1;
    res.text = prompt.empty()
                   ? "Serius API is running. Load a model to get live completions."
                   : prompt;
    return res;
}

void stream_fallback(const std::string &content,
                     const std::string &req_id,
                     const std::string &model_name,
                     httplib::DataSink &sink)
{
    const auto created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    nlohmann::json intro = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", {{"role", "assistant"}}},
            {"finish_reason", nullptr}
        }}}
    };
    std::string s1 = "data: " + intro.dump() + "\n\n";
    sink.write(s1.data(), s1.size());

    nlohmann::json chunk = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", {{"content", content}}},
            {"finish_reason", nullptr}
        }}}
    };
    std::string s2 = "data: " + chunk.dump() + "\n\n";
    sink.write(s2.data(), s2.size());

    nlohmann::json stop_chunk = {
        {"id", req_id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model_name},
        {"choices", {{
            {"index", 0},
            {"delta", nlohmann::json::object()},
            {"finish_reason", "stop"}
        }}}
    };
    std::string s3 = "data: " + stop_chunk.dump() + "\n\n";
    sink.write(s3.data(), s3.size());

    const std::string done_str = "data: [DONE]\n\n";
    sink.write(done_str.data(), done_str.size());
}

}  // namespace server
}  // namespace serius

