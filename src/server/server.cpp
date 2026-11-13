#include "server.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "metrics.h"
#include "static_files.h"
#include "../infer/llama_inference.h"
#include "../infer/sampler.h"
#include "../io/config_loader.h"
#include "../io/gguf_loader.h"
#include "../io/hf_llama_loader.h"
#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"
#include "../transformer/model.h"
#include "../transformer/qwen_model.h"

namespace serius {
namespace server {

namespace {

static std::chrono::steady_clock::time_point g_start_time = std::chrono::steady_clock::now();

/// Resolve the sampling seed for a request: honor an explicit "seed" field,
/// otherwise draw a random one so repeated requests don't replay identically.
uint64_t resolve_seed(const nlohmann::json &body)
{
    const uint64_t seed = body.value("seed", 0ULL);
    if (seed != 0) {
        return seed;
    }
    return std::random_device{}();
}

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

struct GenerationResult
{
    std::string text;
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
};

GenerationResult generate_llama(const model::LLaMAModel &model,
                                BPETokenizer &tokenizer,
                                const std::string &prompt,
                                size_t max_new_tokens,
                                const infer::Sampler &sampler,
                                infer::KVCache &cache,
                                InferenceMetrics *metrics)
{
    GenerationResult res;
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty() || ids[0] != BPETokenizer::BOS_ID) {
        ids.insert(ids.begin(), BPETokenizer::BOS_ID);
    }
    res.prompt_tokens = ids.size();

    cache.clear();
    core::Tensor prompt_input({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) {
        prompt_input[i] = static_cast<float>(ids[i]);
    }

    core::Tensor logits = infer::llama_forward_inference_last(model, prompt_input, &cache);
    size_t vocab_size = logits.shape().back();
    size_t last_row = logits.shape()[0] - 1;

    int next_id = 0;
    if (!infer::sample_gpu_candidates(logits, last_row, sampler, next_id)) {
        core::Tensor last_logits({vocab_size});
        const float *src = logits.data() + (last_row * vocab_size);
        std::copy(src, src + vocab_size, last_logits.data());
        next_id = sampler.sample(last_logits);
    }

    for (size_t gen = 0; gen < max_new_tokens; ++gen) {
        if (next_id == BPETokenizer::EOS_ID) {
            break;
        }

        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
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
                  InferenceMetrics *metrics)
{
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty() || ids[0] != BPETokenizer::BOS_ID) {
        ids.insert(ids.begin(), BPETokenizer::BOS_ID);
    }
    cache.clear();

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

    core::Tensor prompt_input({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) {
        prompt_input[i] = static_cast<float>(ids[i]);
    }

    core::Tensor logits = infer::llama_forward_inference_last(model, prompt_input, &cache);
    const size_t last_row = logits.shape()[0] - 1;

    int next_id = 0;
    if (!infer::sample_gpu_candidates(logits, last_row, sampler, next_id)) {
        const size_t vocab_size = logits.shape().back();
        core::Tensor last_logits({vocab_size});
        const float *src = logits.data() + (last_row * vocab_size);
        std::copy(src, src + vocab_size, last_logits.data());
        next_id = sampler.sample(last_logits);
    }

    std::string pending;
    for (size_t gen = 0; gen < max_new_tokens; ++gen) {
        if (sink.is_writable && !sink.is_writable()) {
            break;
        }

        if (next_id == BPETokenizer::EOS_ID) {
            break;
        }

        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
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
        ids.push_back(BPETokenizer::BOS_ID);
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
        if (next_id == BPETokenizer::EOS_ID) {
            break;
        }

        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
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
        ids.push_back(BPETokenizer::BOS_ID);
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
        if (next_id == BPETokenizer::EOS_ID) {
            break;
        }

        std::string tok_str = tokenizer.decode({static_cast<TokenId>(next_id)});
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

}  // namespace

Server::Server(uint16_t port)
    : Server(ServerConfig{port})
{
}

Server::Server(const ServerConfig &config)
    : port_(config.port),
      host_(config.host),
      config_(config),
      model_name_(config.model_name),
      model_type_(config.model_type),
      metrics_(std::make_unique<InferenceMetrics>()),
      llama_cache_(std::make_unique<infer::KVCache>())
{
    if (!config_.model_path.empty()) {
        load_model(config_.model_path, config_.model_type);
    }
}

Server::~Server() = default;

bool Server::load_llama_from_gguf(const std::string &gguf_path)
{
    tokenizer_ = std::make_unique<BPETokenizer>();

    // Prefer a JSON tokenizer next to the GGUF; fall back to the
    // tokenizer metadata embedded in the GGUF itself.
    const std::string dir = std::filesystem::path(gguf_path).parent_path().string();
    std::string vocab_path = dir + "/tokenizer.json";
    if (!std::filesystem::exists(vocab_path)) {
        vocab_path = dir + "/vocab.json";
    }
    if (std::filesystem::exists(vocab_path)) {
        tokenizer_->load(vocab_path);
    }
    if (tokenizer_->vocab_size() <= 260) {
        auto tok_data = io::load_gguf_tokenizer(gguf_path);
        if (tok_data.valid) {
            tokenizer_->load_from_gguf(tok_data);
        }
    }

    model::LLamaConfig config;
    const auto mc = io::gguf_to_model_config(gguf_path);
    if (mc.vocab_size > 0) {
        config.vocab_size = mc.vocab_size;
        config.hidden_size = mc.d_model;
        config.num_hidden_layers = mc.n_layers;
        config.num_attention_heads = mc.n_heads;
        config.num_key_value_heads = mc.n_kv_heads;
        config.intermediate_size = mc.d_ff;
        config.max_position_embeddings = mc.max_seq_len;
        config.rope_theta = mc.rope_theta;
        config.rms_norm_eps = mc.rms_norm_eps;
        if (tokenizer_->vocab_size() > 0) {
            config.vocab_size = tokenizer_->vocab_size();
        }
    }
    if (config.vocab_size == 0) {
        config.vocab_size = tokenizer_->vocab_size() > 0 ? tokenizer_->vocab_size() : 32000;
    }

    llama_model_ = std::make_unique<model::LLaMAModel>(config);
    io::load_llama_weights_from_gguf_file(gguf_path, llama_model_.get());

    maybe_offload_to_gpu();

    if (model_name_ == "default") {
        model_name_ = std::filesystem::path(gguf_path).filename().string();
    }
    return true;
}

void Server::maybe_offload_to_gpu()
{
    if (!llama_model_ || config_.gpu_mode == ServerConfig::GpuMode::kCpu) {
        return;
    }
    const auto mode = (config_.gpu_mode == ServerConfig::GpuMode::kCuda)
        ? infer::LlamaGpuMode::kCuda : infer::LlamaGpuMode::kAuto;
    const auto dec = infer::resolve_llama_gpu(*llama_model_, mode);
    if (!dec.use_gpu) {
        if (dec.fit == infer::LlamaGpuFit::kInsufficientVram) {
            spdlog::warn("Model ~{:.1f}B params needs ~{:.1f} GiB VRAM but only "
                         "{:.1f} GiB free — staying on CPU ({})",
                         dec.est_params / 1e9, dec.need_bytes / 1073741824.0,
                         dec.free_bytes / 1073741824.0, dec.reason);
        } else {
            spdlog::info("LLaMA model stays on CPU ({})", dec.reason);
        }
        return;
    }
    if (dec.fit == infer::LlamaGpuFit::kInsufficientVram) {
        spdlog::warn("GPU forced but estimate needs ~{:.1f} GiB vs {:.1f} GiB free — "
                     "attempting offload anyway", dec.need_bytes / 1073741824.0,
                     dec.free_bytes / 1073741824.0);
    }
    try {
        if (infer::offload_llama_model_to_gpu(*llama_model_)) {
            spdlog::info("LLaMA model ~{:.1f}B params offloaded to GPU "
                         "(~{:.1f} GiB VRAM, {})",
                         dec.est_params / 1e9, dec.need_bytes / 1073741824.0,
                         dec.reason);
        }
    } catch (const std::exception &e) {
        spdlog::error("GPU offload failed ({}): {} — staying on CPU", dec.reason, e.what());
    }
}

bool Server::load_model(const std::string &model_dir_or_config, const std::string &type)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    model_type_ = type;

    if (type == "llama") {
        // Direct .gguf file path → GGUF-native loading.
        if (!model_dir_or_config.empty() &&
            model_dir_or_config.size() >= 5 &&
            model_dir_or_config.compare(model_dir_or_config.size() - 5, 5, ".gguf") == 0 &&
            std::filesystem::is_regular_file(model_dir_or_config)) {
            return load_llama_from_gguf(model_dir_or_config);
        }

        std::string model_dir = ".";
        std::string actual_config_path = model_dir_or_config;

        if (!model_dir_or_config.empty()) {
            if (std::filesystem::is_directory(model_dir_or_config)) {
                model_dir = model_dir_or_config;
                actual_config_path = model_dir_or_config + "/config.json";
            } else {
                size_t pos = model_dir_or_config.find_last_of("/\\");
                if (pos != std::string::npos) {
                    model_dir = model_dir_or_config.substr(0, pos);
                } else {
                    model_dir = ".";
                }
            }
        }

        // First .gguf inside the model dir (GGUF-only distributions have no
        // config.json / safetensors; it provides config, tokenizer and weights).
        std::string dir_gguf;
        if (!model_dir.empty() && std::filesystem::is_directory(model_dir)) {
            for (const auto &entry : std::filesystem::directory_iterator(model_dir)) {
                std::string p = entry.path().string();
                if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0) {
                    dir_gguf = p;
                    break;
                }
            }
        }

        model::LLamaConfig config;
        bool vocab_known = false;
        const bool has_config_json =
            actual_config_path.size() >= 5 &&
            actual_config_path.compare(actual_config_path.size() - 5, 5, ".json") == 0 &&
            std::filesystem::exists(actual_config_path);
        if (has_config_json) {
            config = io::load_llama_config(actual_config_path);
            vocab_known = config.vocab_size > 0;
        } else if (!dir_gguf.empty()) {
            const auto mc = io::gguf_to_model_config(dir_gguf);
            if (mc.vocab_size > 0) {
                config.vocab_size = mc.vocab_size;
                config.hidden_size = mc.d_model;
                config.num_hidden_layers = mc.n_layers;
                config.num_attention_heads = mc.n_heads;
                config.num_key_value_heads = mc.n_kv_heads;
                config.intermediate_size = mc.d_ff;
                config.max_position_embeddings = mc.max_seq_len;
                config.rope_theta = mc.rope_theta;
                config.rms_norm_eps = mc.rms_norm_eps;
                vocab_known = true;
            }
        }
        if (config.vocab_size == 0) {
            config.vocab_size = 32000;
        }

        tokenizer_ = std::make_unique<BPETokenizer>();
        std::string vocab_path = model_dir + "/vocab.json";
        if (!std::filesystem::exists(vocab_path)) {
            vocab_path = model_dir + "/tokenizer.json";
        }
        if (std::filesystem::exists(vocab_path)) {
            tokenizer_->load(vocab_path);
        }
        if (tokenizer_->vocab_size() <= 260 && !dir_gguf.empty()) {
            auto tok_data = io::load_gguf_tokenizer(dir_gguf);
            if (tok_data.valid) {
                tokenizer_->load_from_gguf(tok_data);
            }
        }
        if (!vocab_known && tokenizer_->vocab_size() > 0) {
            config.vocab_size = tokenizer_->vocab_size();
        }

        llama_model_ = std::make_unique<model::LLaMAModel>(config);
        bool weights_loaded = false;
        if (!model_dir.empty() && std::filesystem::is_directory(model_dir)) {
            weights_loaded = io::load_llama_weights_from_dir(model_dir, llama_model_.get());
        }
        if (!weights_loaded && !dir_gguf.empty()) {
            weights_loaded = io::load_llama_weights_from_gguf_file(dir_gguf, llama_model_.get());
        }
        (void)weights_loaded;

        maybe_offload_to_gpu();

        if (model_name_ == "default" && !model_dir.empty() && model_dir != ".") {
            model_name_ = std::filesystem::path(model_dir).filename().string();
        }
        return true;
    } else if (type == "transformer") {
        model::ModelConfig config;
        if (!model_dir_or_config.empty() && std::filesystem::exists(model_dir_or_config)) {
            config = io::load_model_config(model_dir_or_config);
        } else {
            config.vocab_size = 320;
            config.d_model = 64;
            config.n_heads = 2;
            config.n_layers = 2;
            config.d_ff = 128;
            config.max_seq_len = 64;
        }
        config.normalize_moe();
        transformer_model_ = std::make_unique<model::TransformerModel>(config);
        tokenizer_ = std::make_unique<BPETokenizer>();
        return true;
    }

    return false;
}

void Server::set_llama_model(std::unique_ptr<model::LLaMAModel> model, std::unique_ptr<BPETokenizer> tokenizer)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    llama_model_ = std::move(model);
    if (tokenizer) {
        tokenizer_ = std::move(tokenizer);
    } else if (!tokenizer_) {
        tokenizer_ = std::make_unique<BPETokenizer>();
    }
    model_type_ = "llama";
}

void Server::set_transformer_model(std::unique_ptr<model::TransformerModel> model, std::unique_ptr<BPETokenizer> tokenizer)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    transformer_model_ = std::move(model);
    if (tokenizer) {
        tokenizer_ = std::move(tokenizer);
    } else if (!tokenizer_) {
        tokenizer_ = std::make_unique<BPETokenizer>();
    }
    model_type_ = "transformer";
}

bool Server::has_model() const
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    return (llama_model_ != nullptr || qwen_model_ != nullptr || transformer_model_ != nullptr);
}

void Server::set_model_name(const std::string &name)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    model_name_ = name;
}

std::string Server::model_name() const
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    return model_name_;
}

InferenceMetrics &Server::metrics()
{
    return *metrics_;
}

void Server::setup_routes()
{
    g_start_time = std::chrono::steady_clock::now();

    // Log handler exceptions and return an OpenAI-style error payload instead
    // of httplib's bare 500 + EXCEPTION_WHAT header.
    server_.set_exception_handler([](const httplib::Request &, httplib::Response &res,
                                     std::exception_ptr ep) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception &e) {
            spdlog::error("Request handler exception: {}", e.what());
            res.status = 500;
            res.set_content(
                nlohmann::json{{"error", {{"message", e.what()}, {"type", "internal_error"}}}}.dump(),
                "application/json");
        } catch (...) {
            spdlog::error("Request handler exception: unknown");
            res.status = 500;
            res.set_content("{\"error\":{\"message\":\"internal error\",\"type\":\"internal_error\"}}",
                            "application/json");
        }
    });

    // Setup static file serving for web UI
    setup_static_routes(server_, config_.ui_dir);

    server_.Get("/health", [this](const httplib::Request &, httplib::Response &res) {
        nlohmann::json j;
        j["status"] = "ok";
        j["model_loaded"] = has_model();
        j["model_name"] = model_name();
        res.set_content(j.dump(), "application/json");
    });

    server_.Get("/metrics", [this](const httplib::Request &req, httplib::Response &res) {
        const std::string accept = req.get_header_value("Accept");
        if (accept.find("application/json") != std::string::npos) {
            const auto now = std::chrono::steady_clock::now();
            const auto uptime_sec = std::chrono::duration<double>(now - g_start_time).count();
            nlohmann::json j;
            j["uptime_seconds"] = uptime_sec;
            j["model_loaded"] = has_model();
            res.set_content(j.dump(), "application/json");
        } else {
            res.set_content(metrics_->prometheus_text(), "text/plain; version=0.0.4; charset=utf-8");
        }
    });

    server_.Get("/v1/models", [this](const httplib::Request &, httplib::Response &res) {
        nlohmann::json out;
        out["object"] = "list";
        out["data"] = nlohmann::json::array();
        out["data"].push_back({
            {"id", model_name()},
            {"object", "model"},
            {"created", static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(g_start_time.time_since_epoch()).count())},
            {"owned_by", "serius"}
        });
        res.set_content(out.dump(), "application/json");
    });

    server_.Get(R"(/v1/models/([^/]+))", [this](const httplib::Request &req, httplib::Response &res) {
        std::string requested = req.matches[1];
        nlohmann::json out;
        out["id"] = requested;
        out["object"] = "model";
        out["created"] = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(g_start_time.time_since_epoch()).count());
        out["owned_by"] = "serius";
        res.set_content(out.dump(), "application/json");
    });

    server_.Post("/v1/chat/completions", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        const std::string prompt = extract_user_prompt(body);
        const float temp = body.value("temperature", config_.default_temperature);
        const int top_k = body.value("top_k", config_.default_top_k);
        const float top_p = body.value("top_p", config_.default_top_p);
        const size_t max_new = body.value("max_tokens", body.value("max_new_tokens", config_.max_new_tokens));
        const bool stream = body.value("stream", false);
        const std::string requested_model = body.value("model", model_name());
        const std::string req_id = "chatcmpl-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        infer::Sampler sampler(temp, top_k, top_p, resolve_seed(body));

        if (stream) {
            res.status = 200;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, max_new, sampler, req_id, requested_model](size_t offset, httplib::DataSink &sink) -> bool {
                    if (offset > 0) {
                        sink.done();
                        return true;
                    }

                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    auto t_start = std::chrono::steady_clock::now();
                    metrics_->on_request_start();

                    try {
                        if (llama_model_ && tokenizer_) {
                            stream_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, *llama_cache_, req_id, requested_model, sink, metrics_.get());
                        } else if (transformer_model_ && tokenizer_) {
                            stream_transformer(*transformer_model_, *tokenizer_, prompt, max_new, sampler, req_id, requested_model, sink, metrics_.get());
                        } else {
                            stream_fallback("Serius API is running without a loaded model. Load a model to get live completions.", req_id, requested_model, sink);
                        }
                    } catch (const std::exception &e) {
                        std::string err = std::string("data: {\"error\":\"") + e.what() + "\"}\n\ndata: [DONE]\n\n";
                        sink.write(err.data(), err.size());
                    } catch (...) {
                    }

                    auto t_end = std::chrono::steady_clock::now();
                    metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());
                    sink.done();
                    return true;
                });
            return;
        }

        // Non-streaming
        std::lock_guard<std::mutex> lock(inference_mutex_);
        auto t_start = std::chrono::steady_clock::now();
        metrics_->on_request_start();

        GenerationResult gen_res;
        if (llama_model_ && tokenizer_) {
            gen_res = generate_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, *llama_cache_, metrics_.get());
        } else if (transformer_model_ && tokenizer_) {
            gen_res = generate_transformer(*transformer_model_, *tokenizer_, prompt, max_new, sampler, metrics_.get());
        } else {
            gen_res = fallback_generation(prompt);
        }

        auto t_end = std::chrono::steady_clock::now();
        metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());

        const auto created = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

        nlohmann::json out;
        out["id"] = req_id;
        out["object"] = "chat.completion";
        out["created"] = created;
        out["model"] = requested_model;
        out["choices"] = nlohmann::json::array();
        out["choices"].push_back({
            {"index", 0},
            {"message", {{"role", "assistant"}, {"content", gen_res.text}}},
            {"finish_reason", "stop"}
        });
        out["usage"] = {
            {"prompt_tokens", gen_res.prompt_tokens},
            {"completion_tokens", gen_res.completion_tokens},
            {"total_tokens", gen_res.prompt_tokens + gen_res.completion_tokens}
        };
        res.set_content(out.dump(), "application/json");
    });

    server_.Post("/v1/completions", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        const std::string prompt = body.value("prompt", "");
        const float temp = body.value("temperature", config_.default_temperature);
        const int top_k = body.value("top_k", config_.default_top_k);
        const float top_p = body.value("top_p", config_.default_top_p);
        const size_t max_new = body.value("max_tokens", body.value("max_new_tokens", config_.max_new_tokens));
        const bool stream = body.value("stream", false);
        const std::string requested_model = body.value("model", model_name());
        const std::string req_id = "cmpl-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        infer::Sampler sampler(temp, top_k, top_p, resolve_seed(body));

        if (stream) {
            res.status = 200;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, max_new, sampler, req_id, requested_model](size_t offset, httplib::DataSink &sink) -> bool {
                    if (offset > 0) {
                        sink.done();
                        return true;
                    }

                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    auto t_start = std::chrono::steady_clock::now();
                    metrics_->on_request_start();

                    if (llama_model_ && tokenizer_) {
                        stream_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, *llama_cache_, req_id, requested_model, sink, metrics_.get());
                    } else if (transformer_model_ && tokenizer_) {
                        stream_transformer(*transformer_model_, *tokenizer_, prompt, max_new, sampler, req_id, requested_model, sink, metrics_.get());
                    } else {
                        stream_fallback("Serius API is running without a loaded model. Load a model to get live completions.", req_id, requested_model, sink);
                    }

                    auto t_end = std::chrono::steady_clock::now();
                    metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());
                    sink.done();
                    return true;
                });
            return;
        }

        std::lock_guard<std::mutex> lock(inference_mutex_);
        auto t_start = std::chrono::steady_clock::now();
        metrics_->on_request_start();

        GenerationResult gen_res;
        if (llama_model_ && tokenizer_) {
            gen_res = generate_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, *llama_cache_, metrics_.get());
        } else if (transformer_model_ && tokenizer_) {
            gen_res = generate_transformer(*transformer_model_, *tokenizer_, prompt, max_new, sampler, metrics_.get());
        } else {
            gen_res = fallback_generation(prompt);
        }

        auto t_end = std::chrono::steady_clock::now();
        metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());

        const auto created = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

        nlohmann::json out;
        out["id"] = req_id;
        out["object"] = "text_completion";
        out["created"] = created;
        out["model"] = requested_model;
        out["choices"] = nlohmann::json::array();
        out["choices"].push_back({
            {"text", gen_res.text},
            {"index", 0},
            {"finish_reason", "stop"}
        });
        out["usage"] = {
            {"prompt_tokens", gen_res.prompt_tokens},
            {"completion_tokens", gen_res.completion_tokens},
            {"total_tokens", gen_res.prompt_tokens + gen_res.completion_tokens}
        };
        res.set_content(out.dump(), "application/json");
    });

    server_.Post("/v1/embeddings", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        std::vector<std::string> inputs;
        if (body.contains("input")) {
            if (body["input"].is_string()) {
                inputs.push_back(body["input"].get<std::string>());
            } else if (body["input"].is_array()) {
                for (const auto &item : body["input"]) {
                    if (item.is_string()) {
                        inputs.push_back(item.get<std::string>());
                    }
                }
            }
        }
        if (inputs.empty()) {
            inputs.push_back("");
        }

        std::lock_guard<std::mutex> lock(inference_mutex_);
        metrics_->on_request_start();

        nlohmann::json out;
        out["object"] = "list";
        out["data"] = nlohmann::json::array();
        out["model"] = body.value("model", model_name_);

        size_t total_prompt_tokens = 0;

        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            const std::string &inp = inputs[idx];
            std::vector<float> emb_vector;

            if (llama_model_ && tokenizer_) {
                std::vector<TokenId> ids = tokenizer_->encode(inp);
                if (ids.empty()) ids.push_back(BPETokenizer::BOS_ID);
                total_prompt_tokens += ids.size();

                size_t hs = llama_model_->config().hidden_size;
                emb_vector.assign(hs, 0.0f);

                core::autograd::Node *embed_node = llama_model_->embed_tokens();
                if (embed_node && embed_node->data.size() > 0) {
                    const auto &weights = embed_node->data;
                    size_t vocab = llama_model_->config().vocab_size;
                    for (auto id : ids) {
                        size_t tid = static_cast<size_t>(id) % vocab;
                        for (size_t d = 0; d < hs; ++d) {
                            emb_vector[d] += weights.at(tid, d);
                        }
                    }
                    for (size_t d = 0; d < hs; ++d) {
                        emb_vector[d] /= static_cast<float>(ids.size());
                    }
                }
            } else {
                emb_vector = {0.0f, 0.0f, 0.0f};
            }

            out["data"].push_back({
                {"object", "embedding"},
                {"embedding", emb_vector},
                {"index", idx}
            });
        }

        out["usage"] = {
            {"prompt_tokens", total_prompt_tokens},
            {"total_tokens", total_prompt_tokens}
        };
        metrics_->on_request_end(0.0f);
        res.set_content(out.dump(), "application/json");
    });
}

int Server::run()
{
    setup_routes();
    spdlog::info("Serius server listening on {}:{}", host_, port_);
    const bool ok = server_.listen(host_.c_str(), static_cast<int>(port_));
    if (!ok) {
        spdlog::error("Failed to listen on {}:{}", host_, port_);
        return 1;
    }
    return 0;
}

void Server::stop()
{
    server_.stop();
}

std::string extract_last_user_message(const std::string &json_body)
{
    try {
        auto body = nlohmann::json::parse(json_body.empty() ? "{}" : json_body);
        return extract_user_prompt(body);
    } catch (...) {
        return "";
    }
}

std::string build_completion_text(const std::string &prompt)
{
    return prompt.empty() ? "Serius API is running." : prompt;
}

}  // namespace server
}  // namespace serius
