#include "server.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "metrics.h"
#include "static_files.h"
#include "../infer/llama_inference.h"
#include "../infer/sampler.h"
#include "../io/config_loader.h"
#include "../io/hf_llama_loader.h"
#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"
#include "../transformer/model.h"
#include "../transformer/qwen_model.h"

namespace serius {
namespace server {

namespace {

static std::chrono::steady_clock::time_point g_start_time = std::chrono::steady_clock::now();

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
                                InferenceMetrics *metrics)
{
    GenerationResult res;
    std::vector<TokenId> ids = tokenizer.encode(prompt);
    if (ids.empty() || ids[0] != BPETokenizer::BOS_ID) {
        ids.insert(ids.begin(), BPETokenizer::BOS_ID);
    }
    res.prompt_tokens = ids.size();

    infer::KVCache cache;
    core::Tensor prompt_input({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) {
        prompt_input[i] = static_cast<float>(ids[i]);
    }

    core::Tensor logits = infer::llama_forward_inference(model, prompt_input, &cache);
    size_t vocab_size = logits.shape().back();
    size_t last_row = logits.shape()[0] - 1;
    core::Tensor last_logits({vocab_size});
    const float *src = logits.data() + (last_row * vocab_size);
    std::copy(src, src + vocab_size, last_logits.data());

    int next_id = sampler.sample(last_logits);

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
        next_id = sampler.sample(next_logits);
    }

    return res;
}

void stream_llama(const model::LLaMAModel &model,
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
    if (ids.empty() || ids[0] != BPETokenizer::BOS_ID) {
        ids.insert(ids.begin(), BPETokenizer::BOS_ID);
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

    infer::KVCache cache;
    core::Tensor prompt_input({ids.size()});
    for (size_t i = 0; i < ids.size(); ++i) {
        prompt_input[i] = static_cast<float>(ids[i]);
    }

    core::Tensor logits = infer::llama_forward_inference(model, prompt_input, &cache);
    size_t vocab_size = logits.shape().back();
    size_t last_row = logits.shape()[0] - 1;
    core::Tensor last_logits({vocab_size});
    const float *src = logits.data() + (last_row * vocab_size);
    std::copy(src, src + vocab_size, last_logits.data());

    int next_id = sampler.sample(last_logits);

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

        nlohmann::json chunk = {
            {"id", req_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"choices", {{
                {"index", 0},
                {"delta", {{"content", tok_str}}},
                {"finish_reason", nullptr}
            }}}
        };
        std::string chunk_str = "data: " + chunk.dump() + "\n\n";
        sink.write(chunk_str.data(), chunk_str.size());

        if (gen + 1 == max_new_tokens) {
            break;
        }

        size_t pos = ids.size();
        ids.push_back(static_cast<TokenId>(next_id));
        core::Tensor next_logits = infer::llama_forward_single_token(model, next_id, pos, &cache);
        next_id = sampler.sample(next_logits);
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

        nlohmann::json chunk = {
            {"id", req_id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model_name},
            {"choices", {{
                {"index", 0},
                {"delta", {{"content", tok_str}}},
                {"finish_reason", nullptr}
            }}}
        };
        std::string chunk_str = "data: " + chunk.dump() + "\n\n";
        sink.write(chunk_str.data(), chunk_str.size());

        ids.push_back(static_cast<TokenId>(next_id));
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
      metrics_(std::make_unique<InferenceMetrics>())
{
    if (!config_.model_path.empty()) {
        load_model(config_.model_path, config_.model_type);
    }
}

Server::~Server() = default;

bool Server::load_model(const std::string &model_dir_or_config, const std::string &type)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    model_type_ = type;

    if (type == "llama") {
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

        model::LLamaConfig config;
        if (!actual_config_path.empty() && std::filesystem::exists(actual_config_path)) {
            config = io::load_llama_config(actual_config_path);
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

        llama_model_ = std::make_unique<model::LLaMAModel>(config);
        if (!model_dir.empty() && std::filesystem::exists(model_dir)) {
            io::load_llama_weights_from_dir(model_dir, llama_model_.get());
        }

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

        infer::Sampler sampler(temp, top_k, top_p, 42u);

        if (stream) {
            res.status = 200;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, max_new, sampler, req_id, requested_model](size_t offset, httplib::DataSink &sink) -> bool {
                    if (offset > 0) {
                        return false;
                    }

                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    auto t_start = std::chrono::steady_clock::now();
                    metrics_->on_request_start();

                    if (llama_model_ && tokenizer_) {
                        stream_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, req_id, requested_model, sink, metrics_.get());
                    } else if (transformer_model_ && tokenizer_) {
                        stream_transformer(*transformer_model_, *tokenizer_, prompt, max_new, sampler, req_id, requested_model, sink, metrics_.get());
                    } else {
                        stream_fallback("Serius API is running without a loaded model. Load a model to get live completions.", req_id, requested_model, sink);
                    }

                    auto t_end = std::chrono::steady_clock::now();
                    metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());
                    return false;
                });
            return;
        }

        // Non-streaming
        std::lock_guard<std::mutex> lock(inference_mutex_);
        auto t_start = std::chrono::steady_clock::now();
        metrics_->on_request_start();

        GenerationResult gen_res;
        if (llama_model_ && tokenizer_) {
            gen_res = generate_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, metrics_.get());
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

        infer::Sampler sampler(temp, top_k, top_p, 42u);

        if (stream) {
            res.status = 200;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, max_new, sampler, req_id, requested_model](size_t offset, httplib::DataSink &sink) -> bool {
                    if (offset > 0) return false;

                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    auto t_start = std::chrono::steady_clock::now();
                    metrics_->on_request_start();

                    if (llama_model_ && tokenizer_) {
                        stream_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, req_id, requested_model, sink, metrics_.get());
                    } else if (transformer_model_ && tokenizer_) {
                        stream_transformer(*transformer_model_, *tokenizer_, prompt, max_new, sampler, req_id, requested_model, sink, metrics_.get());
                    } else {
                        stream_fallback("Serius API is running without a loaded model. Load a model to get live completions.", req_id, requested_model, sink);
                    }

                    auto t_end = std::chrono::steady_clock::now();
                    metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());
                    return false;
                });
            return;
        }

        std::lock_guard<std::mutex> lock(inference_mutex_);
        auto t_start = std::chrono::steady_clock::now();
        metrics_->on_request_start();

        GenerationResult gen_res;
        if (llama_model_ && tokenizer_) {
            gen_res = generate_llama(*llama_model_, *tokenizer_, prompt, max_new, sampler, metrics_.get());
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
        out["model"] = body.value("model", model_name());

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
