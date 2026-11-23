#include "server.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>

#include "../hub/downloader.h"
#include "../infer/llama_inference.h"
#include "../infer/sampler.h"
#include "../io/config_loader.h"
#include "../io/gguf_loader.h"
#include "../io/hf_llama_loader.h"
#include "../tokenizer/bpe.h"
#include "../transformer/llama_model.h"
#include "../transformer/model.h"
#include "../transformer/qwen_model.h"
#include "metrics.h"
#include "server_internal.h"
#include "static_files.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <psapi.h>
#include <windows.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__linux__)
#include <cstdlib>
#include <fstream>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

#ifdef SERIUS_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace serius {
namespace server {

namespace {

/// Resident memory of THIS server process (bytes); 0 when unsupported.
size_t process_ram_bytes()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc), sizeof(pmc))) {
        return static_cast<size_t>(pmc.WorkingSetSize);
    }
#elif defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            return std::strtoull(line.c_str() + 6, nullptr, 10) * 1024;
        }
    }
#elif defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size);
    }
#endif
    return 0;
}

/// Device memory currently allocated in THIS process (bytes); 0 without CUDA.
size_t process_vram_bytes()
{
#ifdef SERIUS_USE_CUDA
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        return total_bytes > free_bytes ? total_bytes - free_bytes : 0;
    }
#endif
    return 0;
}

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

/// Hard cap on client-requested completion length. Without it a single
/// request (e.g. max_tokens = -1, which wraps to SIZE_MAX) pins
/// inference_mutex_ — and the GPU — indefinitely.
constexpr size_t kMaxNewTokens = 32768;
/// Cap on the number of inputs accepted by /v1/embeddings per request.
constexpr size_t kMaxEmbeddingInputs = 256;

/// Bearer token required on model-management routes (POST /v1/models/load,
/// DELETE /v1/models/{id}) when the SERIUS_ADMIN_TOKEN env var is set.
/// Empty env = open mode, kept for local development.
std::string admin_token()
{
    const char *t = std::getenv("SERIUS_ADMIN_TOKEN");
    return t ? std::string(t) : std::string();
}

bool token_matches(const std::string &presented, const std::string &expected)
{
    if (presented.size() != expected.size()) {
        return false;
    }
    uint8_t diff = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        diff |= static_cast<uint8_t>(presented[i] ^ expected[i]);
    }
    return diff == 0;
}

/// Returns true and fills `res` with 401 when the request lacks a valid
/// admin token. No-op (returns true) when SERIUS_ADMIN_TOKEN is unset.
bool admin_authorized(const httplib::Request &req, httplib::Response &res)
{
    const std::string token = admin_token();
    if (token.empty()) {
        return true;
    }
    const std::string auth = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    const bool ok = auth.rfind(prefix, 0) == 0 && token_matches(auth.substr(prefix.size()), token);
    if (!ok) {
        res.status = 401;
        res.set_content(
            nlohmann::json{{"error", {{"message", "admin token required"}, {"type", "authentication_error"}}}}.dump(),
            "application/json");
    }
    return ok;
}

/// When SERIUS_MODEL_ROOTS is set (';'-separated directories), a model path
/// must canonicalize inside one of them. Unset = any existing path allowed.
bool model_path_allowed(const std::string &path)
{
    const char *roots = std::getenv("SERIUS_MODEL_ROOTS");
    if (!roots || !*roots) {
        return true;
    }
    std::error_code ec;
    const auto canon = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        return false;
    }
    std::string list(roots);
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t sep = list.find(';', pos);
        std::string root = list.substr(pos, sep == std::string::npos ? sep : sep - pos);
        pos = sep == std::string::npos ? list.size() + 1 : sep + 1;
        if (root.empty()) {
            continue;
        }
        std::error_code ec2;
        const auto root_canon = std::filesystem::weakly_canonical(root, ec2);
        if (ec2) {
            continue;
        }
        const auto &c = canon.string();
        const auto &r = root_canon.string();
        if (c == r || (c.size() > r.size() && c.compare(0, r.size(), r) == 0 &&
                       (r.back() == '/' || r.back() == '\\' || c[r.size()] == '/' || c[r.size()] == '\\'))) {
            return true;
        }
    }
    return false;
}

}  // namespace

Server::Server(uint16_t port) : Server(ServerConfig{port}) {}

LoadedModel::~LoadedModel() = default;

Server::Server(const ServerConfig &config)
    : port_(config.port),
      host_(config.host),
      config_(config),
      model_name_(config.model_name),
      model_type_(config.model_type),
      metrics_(std::make_unique<InferenceMetrics>())
{
    if (!config_.model_path.empty()) {
        load_model(config_.model_path, config_.model_type, config_.model_name);
    }
}

Server::~Server() = default;

LoadedModel Server::load_llama_gguf(const std::string &gguf_path)
{
    LoadedModel entry;
    entry.type = "llama";

    auto tokenizer = std::make_unique<BPETokenizer>();

    // Prefer a JSON tokenizer next to the GGUF; fall back to the
    // tokenizer metadata embedded in the GGUF itself.
    const std::string dir = std::filesystem::path(gguf_path).parent_path().string();
    std::string vocab_path = dir + "/tokenizer.json";
    if (!std::filesystem::exists(vocab_path)) {
        vocab_path = dir + "/vocab.json";
    }
    if (std::filesystem::exists(vocab_path)) {
        tokenizer->load(vocab_path);
    }
    if (tokenizer->vocab_size() <= 260) {
        auto tok_data = io::load_gguf_tokenizer(gguf_path);
        if (tok_data.valid) {
            tokenizer->load_from_gguf(tok_data);
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
        if (tokenizer->vocab_size() > 0) {
            config.vocab_size = tokenizer->vocab_size();
        }
    }
    if (config.vocab_size == 0) {
        config.vocab_size = tokenizer->vocab_size() > 0 ? tokenizer->vocab_size() : 32000;
    }

    entry.llama = std::make_unique<model::LLaMAModel>(config);
    io::load_llama_weights_from_gguf_file(gguf_path, entry.llama.get());
    entry.tokenizer = std::move(tokenizer);
    entry.kv_cache = std::make_unique<infer::KVCache>();

    maybe_offload_to_gpu(*entry.llama);
    return entry;
}

void Server::maybe_offload_to_gpu(model::LLaMAModel &model)
{
    if (config_.gpu_mode == ServerConfig::GpuMode::kCpu) {
        return;
    }
    // --gpu-layers N: partial offload (GPU prefix + CPU suffix). Returns
    // false for N >= layer count, which falls through to the full offload.
    if (config_.gpu_layers > 0 && infer::offload_llama_model_partial(model, static_cast<size_t>(config_.gpu_layers))) {
        spdlog::info("LLaMA model partial offload: {} layers on GPU, rest CPU", config_.gpu_layers);
        return;
    }
    const auto mode =
        (config_.gpu_mode == ServerConfig::GpuMode::kCuda) ? infer::LlamaGpuMode::kCuda : infer::LlamaGpuMode::kAuto;
    const auto dec = infer::resolve_llama_gpu(model, mode);
    if (!dec.use_gpu) {
        if (dec.fit == infer::LlamaGpuFit::kInsufficientVram) {
            spdlog::warn(
                "Model ~{:.1f}B params needs ~{:.1f} GiB VRAM but only "
                "{:.1f} GiB free — staying on CPU ({})",
                dec.est_params / 1e9, dec.need_bytes / 1073741824.0, dec.free_bytes / 1073741824.0, dec.reason);
        } else {
            spdlog::info("LLaMA model stays on CPU ({})", dec.reason);
        }
        return;
    }
    if (dec.fit == infer::LlamaGpuFit::kInsufficientVram) {
        spdlog::warn(
            "GPU forced but estimate needs ~{:.1f} GiB vs {:.1f} GiB free — "
            "attempting offload anyway",
            dec.need_bytes / 1073741824.0, dec.free_bytes / 1073741824.0);
    }
    try {
        if (infer::offload_llama_model_to_gpu(model)) {
            spdlog::info(
                "LLaMA model ~{:.1f}B params offloaded to GPU "
                "(~{:.1f} GiB VRAM, {})",
                dec.est_params / 1e9, dec.need_bytes / 1073741824.0, dec.reason);
        }
    }
    catch (const std::exception &e) {
        spdlog::error("GPU offload failed ({}): {} — staying on CPU", dec.reason, e.what());
    }
}

namespace {

/// Derive a registry name from a model path: directory name for dirs,
/// file stem for .gguf, parent dir name for config files. Empty if the
/// path gives no usable name.
std::string derive_model_name(const std::string &path)
{
    namespace fs = std::filesystem;
    if (path.empty()) {
        return "";
    }
    const fs::path p(path);
    std::error_code ec;
    if (fs::is_regular_file(p, ec)) {
        if (p.extension() == ".gguf") {
            return p.stem().string();
        }
        return p.parent_path().filename().string();
    }
    return p.filename().string();
}

}  // namespace

std::string Server::pick_model_name(const std::string &explicit_name, const std::string &derived)
{
    std::string base;
    if (!explicit_name.empty() && explicit_name != "default") {
        base = explicit_name;
    } else if (!derived.empty()) {
        base = derived;
    } else if (model_name_ != "default") {
        base = model_name_;
    } else {
        base = "model";
    }

    std::string candidate = base;
    for (int i = 2; models_.count(candidate) != 0; ++i) {
        candidate = base + "-" + std::to_string(i);
    }
    return candidate;
}

LoadedModel Server::load_llama_dir(const std::string &model_dir_or_config)
{
    LoadedModel entry;
    entry.type = "llama";

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
        for (const auto &dir_entry : std::filesystem::directory_iterator(model_dir)) {
            std::string p = dir_entry.path().string();
            if (p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0) {
                dir_gguf = p;
                break;
            }
        }
    }

    model::LLamaConfig config;
    bool vocab_known = false;
    const bool has_config_json = actual_config_path.size() >= 5 &&
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

    entry.tokenizer = std::make_unique<BPETokenizer>();
    std::string vocab_path = model_dir + "/vocab.json";
    if (!std::filesystem::exists(vocab_path)) {
        vocab_path = model_dir + "/tokenizer.json";
    }
    if (std::filesystem::exists(vocab_path)) {
        entry.tokenizer->load(vocab_path);
    }
    if (entry.tokenizer->vocab_size() <= 260 && !dir_gguf.empty()) {
        auto tok_data = io::load_gguf_tokenizer(dir_gguf);
        if (tok_data.valid) {
            entry.tokenizer->load_from_gguf(tok_data);
        }
    }
    if (!vocab_known && entry.tokenizer->vocab_size() > 0) {
        config.vocab_size = entry.tokenizer->vocab_size();
    }

    entry.llama = std::make_unique<model::LLaMAModel>(config);
    bool weights_loaded = false;
    if (!model_dir.empty() && std::filesystem::is_directory(model_dir)) {
        weights_loaded = io::load_llama_weights_from_dir(model_dir, entry.llama.get());
    }
    if (!weights_loaded && !dir_gguf.empty()) {
        weights_loaded = io::load_llama_weights_from_gguf_file(dir_gguf, entry.llama.get());
    }
    (void)weights_loaded;
    entry.kv_cache = std::make_unique<infer::KVCache>();

    maybe_offload_to_gpu(*entry.llama);
    return entry;
}

LoadedModel Server::load_transformer_entry(const std::string &model_dir_or_config)
{
    LoadedModel entry;
    entry.type = "transformer";

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
    entry.transformer = std::make_unique<model::TransformerModel>(config);
    entry.tokenizer = std::make_unique<BPETokenizer>();
    return entry;
}

bool Server::load_model(const std::string &model_dir_or_config, const std::string &type, const std::string &name)
{
    // Serialize loads against each other, but NOT against inference: the
    // slow weight loading below runs without inference_mutex_ so other
    // models keep serving requests. Only the registry insert is locked.
    std::lock_guard<std::mutex> load_lock(load_mutex_);

    LoadedModel entry;
    if (type == "llama") {
        const bool is_gguf = !model_dir_or_config.empty() && model_dir_or_config.size() >= 5 &&
                             model_dir_or_config.compare(model_dir_or_config.size() - 5, 5, ".gguf") == 0 &&
                             std::filesystem::is_regular_file(model_dir_or_config);
        entry = is_gguf ? load_llama_gguf(model_dir_or_config) : load_llama_dir(model_dir_or_config);
        if (!entry.llama) {
            return false;
        }
    } else if (type == "transformer") {
        entry = load_transformer_entry(model_dir_or_config);
        if (!entry.transformer) {
            return false;
        }
    } else {
        return false;
    }

    // Registry insert: brief critical section. pick_model_name reads
    // models_, so it must run under inference_mutex_.
    std::string entry_name;
    {
        std::lock_guard<std::mutex> lock(inference_mutex_);
        model_type_ = type;
        entry.name = pick_model_name(name, derive_model_name(model_dir_or_config));
        entry.created = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
        entry_name = entry.name;
        models_[entry_name] = std::move(entry);
        last_loaded_name_ = entry_name;

        if (model_name_ == "default" && models_.size() == 1) {
            model_name_ = entry_name;
        }
    }
    spdlog::info("Model '{}' loaded (type {})", entry_name, type);
    return true;
}

bool Server::unload_model(const std::string &name)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    return models_.erase(name) > 0;
}

LoadedModel *Server::find_model(const std::string &name)
{
    if (models_.empty()) {
        return nullptr;
    }
    if (!name.empty() && name != model_name_) {
        auto it = models_.find(name);
        return it == models_.end() ? nullptr : &it->second;
    }
    auto it = models_.find(model_name_);
    if (it != models_.end()) {
        return &it->second;
    }
    return &models_.begin()->second;
}

void Server::set_llama_model(std::unique_ptr<model::LLaMAModel> model, std::unique_ptr<BPETokenizer> tokenizer)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    LoadedModel entry;
    entry.name = model_name_;
    entry.type = "llama";
    entry.created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    entry.llama = std::move(model);
    auto it = models_.find(entry.name);
    if (tokenizer) {
        entry.tokenizer = std::move(tokenizer);
    } else if (it != models_.end() && it->second.tokenizer) {
        entry.tokenizer = std::move(it->second.tokenizer);
    } else {
        entry.tokenizer = std::make_unique<BPETokenizer>();
    }
    entry.kv_cache = std::make_unique<infer::KVCache>();
    models_[entry.name] = std::move(entry);
    model_type_ = "llama";
}

void Server::set_transformer_model(std::unique_ptr<model::TransformerModel> model,
                                   std::unique_ptr<BPETokenizer> tokenizer)
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    LoadedModel entry;
    entry.name = model_name_;
    entry.type = "transformer";
    entry.created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    entry.transformer = std::move(model);
    auto it = models_.find(entry.name);
    if (tokenizer) {
        entry.tokenizer = std::move(tokenizer);
    } else if (it != models_.end() && it->second.tokenizer) {
        entry.tokenizer = std::move(it->second.tokenizer);
    } else {
        entry.tokenizer = std::make_unique<BPETokenizer>();
    }
    models_[entry.name] = std::move(entry);
    model_type_ = "transformer";
}

bool Server::has_model() const
{
    std::lock_guard<std::mutex> lock(inference_mutex_);
    return !models_.empty();
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
    server_.set_exception_handler([](const httplib::Request &, httplib::Response &res, std::exception_ptr ep) {
        try {
            std::rethrow_exception(ep);
        }
        catch (const std::exception &e) {
            spdlog::error("Request handler exception: {}", e.what());
            res.status = 500;
            res.set_content(nlohmann::json{{"error", {{"message", e.what()}, {"type", "internal_error"}}}}.dump(),
                            "application/json");
        }
        catch (...) {
            spdlog::error("Request handler exception: unknown");
            res.status = 500;
            res.set_content("{\"error\":{\"message\":\"internal error\",\"type\":\"internal_error\"}}",
                            "application/json");
        }
    });

    // Setup static file serving for web UI
    setup_static_routes(server_, config_.ui_dir);

    server_.Get("/health", [this](const httplib::Request &req, httplib::Response &res) { handle_health(req, res); });

    server_.Get("/metrics", [this](const httplib::Request &req, httplib::Response &res) { handle_metrics(req, res); });

    server_.Get("/v1/models", [this](const httplib::Request &, httplib::Response &res) {
        nlohmann::json out;
        out["object"] = "list";
        out["data"] = nlohmann::json::array();
        const auto created_now = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(g_start_time.time_since_epoch()).count());

        std::lock_guard<std::mutex> lock(inference_mutex_);
        if (models_.empty()) {
            // No model loaded yet: still advertise the configured default
            // name so clients have something to select (backward compat).
            out["data"].push_back(
                {{"id", model_name_}, {"object", "model"}, {"created", created_now}, {"owned_by", "serius"}});
        } else {
            for (const auto &[name, m] : models_) {
                out["data"].push_back({{"id", name},
                                       {"object", "model"},
                                       {"created", m.created != 0 ? m.created : created_now},
                                       {"owned_by", "serius"},
                                       {"type", m.type}});
            }
        }
        res.set_content(out.dump(), "application/json");
    });

    server_.Get(R"(/v1/models/([^/]+))", [this](const httplib::Request &req, httplib::Response &res) {
        std::string requested = req.matches[1];

        std::lock_guard<std::mutex> lock(inference_mutex_);
        if (!models_.empty() && models_.find(requested) == models_.end()) {
            res.status = 404;
            res.set_content(
                nlohmann::json{
                    {"error", {{"message", "model not found: " + requested}, {"type", "invalid_request_error"}}}}
                    .dump(),
                "application/json");
            return;
        }
        nlohmann::json out;
        out["id"] = requested;
        out["object"] = "model";
        out["created"] = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(g_start_time.time_since_epoch()).count());
        out["owned_by"] = "serius";
        res.set_content(out.dump(), "application/json");
    });

    // Load a model at runtime: {"name"?, "path", "type"?}
    // path: directory, config.json or .gguf file. name defaults to the
    // derived path name; an explicit name that already exists -> 409.
    server_.Post("/v1/models/load", [this](const httplib::Request &req, httplib::Response &res) {
        if (!admin_authorized(req, res)) {
            return;
        }
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        }
        catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        std::string path = body.value("path", body.value("model", ""));
        std::string type = body.value("type", "llama");
        const std::string name = body.value("name", "");
        const std::string quant = body.value("quant", "");

        if (path.empty()) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"'path' is required\"}}", "application/json");
            return;
        }

        // hf://repo_id → resolve (and download if needed) the best GGUF on
        // HuggingFace Hub. Runs outside any lock: network + disk only.
        if (path.rfind("hf://", 0) == 0) {
            const std::string repo_id = path.substr(5);
            if (!hub::ModelDownloader::is_valid_repo_id(repo_id)) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error",
                                                {{"message", "invalid HuggingFace repo id (expected owner/name)"},
                                                 {"type", "invalid_request_error"}}}}
                                    .dump(),
                                "application/json");
                return;
            }
            hub::ModelDownloader dl;
            std::string local_path, chosen_name;
            size_t last_logged_pct = 0;
            const hub::ProgressFn progress = [&](size_t downloaded, size_t total) {
                if (total == 0) {
                    return;
                }
                const size_t pct = downloaded * 100 / total;
                if (pct >= last_logged_pct + 10 || downloaded >= total) {
                    last_logged_pct = pct;
                    spdlog::info("hf://{} download: {}%", repo_id, pct);
                }
            };
            if (!dl.resolve_hf_gguf(repo_id, quant, local_path, chosen_name, "", progress)) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error",
                                                {{"message", "failed to resolve hf://" + repo_id},
                                                 {"type", "invalid_request_error"}}}}
                                    .dump(),
                                "application/json");
                return;
            }
            spdlog::info("hf://{} resolved to {}", repo_id, local_path);
            path = local_path;
            type = "llama";  // GGUF files only load through the llama path
        }

        // Qwen2-family models run through the LLaMA path (arch details come
        // from config/metadata, not the flag).
        if (type == "qwen") {
            type = "llama";
        }
        if (type != "llama" && type != "transformer") {
            res.status = 400;
            res.set_content(
                nlohmann::json{
                    {"error", {{"message", "unsupported model type: " + type}, {"type", "invalid_request_error"}}}}
                    .dump(),
                "application/json");
            return;
        }
        std::error_code ec;
        if (!std::filesystem::exists(std::filesystem::path(path), ec)) {
            res.status = 400;
            res.set_content(
                nlohmann::json{
                    {"error", {{"message", "path does not exist: " + path}, {"type", "invalid_request_error"}}}}
                    .dump(),
                "application/json");
            return;
        }
        if (!model_path_allowed(path)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error",
                                            {{"message", "model path is outside the allowed roots"},
                                             {"type", "invalid_request_error"}}}}
                                .dump(),
                            "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(inference_mutex_);
            if (!name.empty() && name != "default" && models_.count(name) != 0) {
                res.status = 409;
                res.set_content(
                    nlohmann::json{
                        {"error", {{"message", "model already loaded: " + name}, {"type", "invalid_request_error"}}}}
                        .dump(),
                    "application/json");
                return;
            }
        }

        if (!load_model(path, type, name)) {
            res.status = 500;
            res.set_content(
                nlohmann::json{{"error", {{"message", "failed to load model from: " + path}, {"type", "server_error"}}}}
                    .dump(),
                "application/json");
            return;
        }

        // load_model picked the registry name (explicit, derived or deduped).
        std::string loaded_name;
        {
            std::lock_guard<std::mutex> lock(inference_mutex_);
            loaded_name = last_loaded_name_;
        }

        nlohmann::json out;
        out["id"] = loaded_name;
        out["object"] = "model";
        out["owned_by"] = "serius";
        out["type"] = type;
        res.set_content(out.dump(), "application/json");
    });

    // Unload a model: DELETE /v1/models/{id}
    server_.Delete(R"(/v1/models/([^/]+))", [this](const httplib::Request &req, httplib::Response &res) {
        if (!admin_authorized(req, res)) {
            return;
        }
        const std::string name = req.matches[1];
        if (!unload_model(name)) {
            res.status = 404;
            res.set_content(
                nlohmann::json{{"error", {{"message", "model not found: " + name}, {"type", "invalid_request_error"}}}}
                    .dump(),
                "application/json");
            return;
        }
        nlohmann::json out;
        out["id"] = name;
        out["object"] = "model";
        out["deleted"] = true;
        res.set_content(out.dump(), "application/json");
    });

    server_.Post("/v1/chat/completions", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        }
        catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        // Require at least one input: a non-empty "prompt" or a "messages"
        // array with at least one usable entry. An empty object would
        // otherwise fall back to the default prompt and burn max_tokens of
        // generation for a meaningless request.
        const auto messages = body.value("messages", nlohmann::json::array());
        const bool has_messages = messages.is_array() && !messages.empty();
        const std::string prompt_field = body.value("prompt", "");
        if (!has_messages && prompt_field.empty()) {
            res.status = 400;
            res.set_content(
                nlohmann::json{
                    {"error", {{"message", "messages or prompt is required"}, {"type", "invalid_request_error"}}}}
                    .dump(),
                "application/json");
            return;
        }

        std::string requested_model = body.value("model", model_name());
        if (requested_model.empty()) {
            requested_model = model_name();
        }

        // Resolve the model up front: its tokenizer decides the prompt
        // format (ChatML vs plain transcript); unknown names get a 404.
        // An empty registry with the default name keeps the old
        // fallback-response behavior.
        bool model_available = false;
        bool is_chatml = false;
        {
            std::lock_guard<std::mutex> lock(inference_mutex_);
            if (LoadedModel *lm = find_model(requested_model)) {
                model_available = true;
                if (lm->tokenizer) {
                    is_chatml = lm->tokenizer->get_id("<|im_start|>") != BPETokenizer::UNK_ID;
                }
            } else if (models_.empty() && requested_model == model_name_) {
                model_available = true;
            }
        }
        if (!model_available) {
            res.status = 404;
            res.set_content(
                nlohmann::json{
                    {"error", {{"message", "model not found: " + requested_model}, {"type", "invalid_request_error"}}}}
                    .dump(),
                "application/json");
            return;
        }

        const std::string prompt = build_chat_prompt(body, is_chatml);
        const float temp = body.value("temperature", config_.default_temperature);
        const int top_k = body.value("top_k", config_.default_top_k);
        const float top_p = body.value("top_p", config_.default_top_p);
        if (!std::isfinite(temp) || !std::isfinite(top_p)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"temperature/top_p must be finite\"}}", "application/json");
            return;
        }
        size_t max_new = body.value("max_tokens", body.value("max_new_tokens", config_.max_new_tokens));
        if (max_new > kMaxNewTokens) {
            max_new = kMaxNewTokens;
        }

        // Reject prompts that exceed the model's context window: without this
        // guard a huge prompt grows the KV cache (and prefill time) without
        // bound. The KV cache itself grows on demand, so this is a UX/DoS
        // guard, not a memory-safety one.
        {
            std::lock_guard<std::mutex> lock(inference_mutex_);
            if (LoadedModel *lm = find_model(requested_model)) {
                if (lm->llama && lm->tokenizer && prompt_exceeds_context(*lm->llama, *lm->tokenizer, prompt, max_new)) {
                    res.status = 400;
                    res.set_content(
                        nlohmann::json{{"error",
                                        {{"message", "prompt + max_tokens exceeds the model context window"},
                                         {"type", "invalid_request_error"}}}}
                            .dump(),
                        "application/json");
                    return;
                }
            }
        }

        const bool stream = body.value("stream", false);
        const std::string req_id =
            "chatcmpl-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        infer::Sampler sampler(temp, top_k, top_p, resolve_seed(body));

        if (stream) {
            res.status = 200;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, max_new, sampler, req_id, requested_model](size_t offset,
                                                                          httplib::DataSink &sink) -> bool {
                    if (offset > 0) {
                        sink.done();
                        return true;
                    }

                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    auto t_start = std::chrono::steady_clock::now();
                    metrics_->on_request_start();

                    try {
                        LoadedModel *lm = find_model(requested_model);
                        if (lm && lm->llama && lm->tokenizer && lm->kv_cache) {
                            stream_llama(*lm->llama, *lm->tokenizer, prompt, max_new, sampler, *lm->kv_cache, req_id,
                                         requested_model, sink, metrics_.get(), &lm->cached_token_ids);
                        } else if (lm && lm->transformer && lm->tokenizer) {
                            stream_transformer(*lm->transformer, *lm->tokenizer, prompt, max_new, sampler, req_id,
                                               requested_model, sink, metrics_.get());
                        } else {
                            stream_fallback(
                                "Serius API is running without a loaded model. Load a model to get live completions.",
                                req_id, requested_model, sink);
                        }
                    }
                    catch (const std::exception &e) {
                        const std::string err =
                            "data: " + nlohmann::json{{"error", e.what()}}.dump() + "\n\ndata: [DONE]\n\n";
                        sink.write(err.data(), err.size());
                    }
                    catch (...) {
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
        LoadedModel *lm = find_model(requested_model);
        if (lm && lm->llama && lm->tokenizer && lm->kv_cache) {
            gen_res = generate_llama(*lm->llama, *lm->tokenizer, prompt, max_new, sampler, *lm->kv_cache,
                                     metrics_.get(), &lm->cached_token_ids);
        } else if (lm && lm->transformer && lm->tokenizer) {
            gen_res = generate_transformer(*lm->transformer, *lm->tokenizer, prompt, max_new, sampler, metrics_.get());
        } else {
            // Echo only the last user text, not the rendered transcript.
            gen_res = fallback_generation(extract_user_prompt(body));
        }

        auto t_end = std::chrono::steady_clock::now();
        metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());

        const auto created = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());

        nlohmann::json out;
        out["id"] = req_id;
        out["object"] = "chat.completion";
        out["created"] = created;
        out["model"] = requested_model;
        out["choices"] = nlohmann::json::array();
        out["choices"].push_back(
            {{"index", 0}, {"message", {{"role", "assistant"}, {"content", gen_res.text}}}, {"finish_reason", "stop"}});
        out["usage"] = {{"prompt_tokens", gen_res.prompt_tokens},
                        {"completion_tokens", gen_res.completion_tokens},
                        {"total_tokens", gen_res.prompt_tokens + gen_res.completion_tokens}};
        res.set_content(out.dump(), "application/json");
    });

    server_.Post("/v1/completions", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        }
        catch (const std::exception &) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"Invalid JSON\"}}", "application/json");
            return;
        }

        const std::string prompt = body.value("prompt", "");
        const float temp = body.value("temperature", config_.default_temperature);
        const int top_k = body.value("top_k", config_.default_top_k);
        const float top_p = body.value("top_p", config_.default_top_p);
        if (!std::isfinite(temp) || !std::isfinite(top_p)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"temperature/top_p must be finite\"}}", "application/json");
            return;
        }
        size_t max_new = body.value("max_tokens", body.value("max_new_tokens", config_.max_new_tokens));
        if (max_new > kMaxNewTokens) {
            max_new = kMaxNewTokens;
        }
        const bool stream = body.value("stream", false);
        std::string requested_model = body.value("model", model_name());
        if (requested_model.empty()) {
            requested_model = model_name();
        }
        {
            std::lock_guard<std::mutex> lock(inference_mutex_);
            LoadedModel *lm = find_model(requested_model);
            if (!lm && !(models_.empty() && requested_model == model_name_)) {
                res.status = 404;
                res.set_content(nlohmann::json{{"error",
                                                {{"message", "model not found: " + requested_model},
                                                 {"type", "invalid_request_error"}}}}
                                    .dump(),
                                "application/json");
                return;
            }
            if (lm && lm->llama && lm->tokenizer &&
                prompt_exceeds_context(*lm->llama, *lm->tokenizer, prompt, max_new)) {
                res.status = 400;
                res.set_content(nlohmann::json{{"error",
                                                {{"message", "prompt + max_tokens exceeds the model context window"},
                                                 {"type", "invalid_request_error"}}}}
                                    .dump(),
                                "application/json");
                return;
            }
        }
        const std::string req_id =
            "cmpl-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

        infer::Sampler sampler(temp, top_k, top_p, resolve_seed(body));

        if (stream) {
            res.status = 200;
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt, max_new, sampler, req_id, requested_model](size_t offset,
                                                                          httplib::DataSink &sink) -> bool {
                    if (offset > 0) {
                        sink.done();
                        return true;
                    }

                    std::lock_guard<std::mutex> lock(inference_mutex_);
                    auto t_start = std::chrono::steady_clock::now();
                    metrics_->on_request_start();

                    LoadedModel *lm = find_model(requested_model);
                    if (lm && lm->llama && lm->tokenizer && lm->kv_cache) {
                        stream_llama(*lm->llama, *lm->tokenizer, prompt, max_new, sampler, *lm->kv_cache, req_id,
                                     requested_model, sink, metrics_.get(), &lm->cached_token_ids);
                    } else if (lm && lm->transformer && lm->tokenizer) {
                        stream_transformer(*lm->transformer, *lm->tokenizer, prompt, max_new, sampler, req_id,
                                           requested_model, sink, metrics_.get());
                    } else {
                        stream_fallback(
                            "Serius API is running without a loaded model. Load a model to get live completions.",
                            req_id, requested_model, sink);
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
        LoadedModel *lm = find_model(requested_model);
        if (lm && lm->llama && lm->tokenizer && lm->kv_cache) {
            gen_res = generate_llama(*lm->llama, *lm->tokenizer, prompt, max_new, sampler, *lm->kv_cache,
                                     metrics_.get(), &lm->cached_token_ids);
        } else if (lm && lm->transformer && lm->tokenizer) {
            gen_res = generate_transformer(*lm->transformer, *lm->tokenizer, prompt, max_new, sampler, metrics_.get());
        } else {
            gen_res = fallback_generation(prompt);
        }

        auto t_end = std::chrono::steady_clock::now();
        metrics_->on_request_end(std::chrono::duration<float>(t_end - t_start).count());

        const auto created = static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());

        nlohmann::json out;
        out["id"] = req_id;
        out["object"] = "text_completion";
        out["created"] = created;
        out["model"] = requested_model;
        out["choices"] = nlohmann::json::array();
        out["choices"].push_back({{"text", gen_res.text}, {"index", 0}, {"finish_reason", "stop"}});
        out["usage"] = {{"prompt_tokens", gen_res.prompt_tokens},
                        {"completion_tokens", gen_res.completion_tokens},
                        {"total_tokens", gen_res.prompt_tokens + gen_res.completion_tokens}};
        res.set_content(out.dump(), "application/json");
    });

    server_.Post("/v1/embeddings", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        }
        catch (const std::exception &) {
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
        if (inputs.size() > kMaxEmbeddingInputs) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"too many inputs\"}}", "application/json");
            return;
        }

        const std::string emb_model_name = body.value("model", model_name_);
        std::lock_guard<std::mutex> lock(inference_mutex_);
        metrics_->on_request_start();

        LoadedModel *lm = find_model(emb_model_name);
        const bool emb_ok = lm && lm->llama && lm->tokenizer;

        nlohmann::json out;
        out["object"] = "list";
        out["data"] = nlohmann::json::array();
        out["model"] = emb_ok ? lm->name : emb_model_name;

        size_t total_prompt_tokens = 0;

        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            const std::string &inp = inputs[idx];
            std::vector<float> emb_vector;

            if (emb_ok) {
                std::vector<TokenId> ids = lm->tokenizer->encode(inp);
                if (ids.empty()) ids.push_back(lm->tokenizer->bos_id());
                total_prompt_tokens += ids.size();

                size_t hs = lm->llama->config().hidden_size;
                emb_vector.assign(hs, 0.0f);

                core::autograd::Node *embed_node = lm->llama->embed_tokens();
                if (embed_node && embed_node->data.size() > 0 &&
                    embed_node->data.dtype() == hal::DType::Float32) {
                    const auto &weights = embed_node->data;
                    const auto &wshape = weights.shape();
                    // Tensor::at() does no bounds checking: clamp every index
                    // to the real tensor dimensions so a config/tensor size
                    // mismatch can't read out of bounds.
                    const size_t w_rows = wshape.size() >= 1 ? wshape[0] : 0;
                    const size_t w_cols = wshape.size() >= 2 ? wshape[1] : 0;
                    const size_t vocab = lm->llama->config().vocab_size;
                    const size_t cols = std::min(hs, w_cols);
                    for (auto id : ids) {
                        if (w_rows == 0) break;
                        size_t tid = static_cast<size_t>(id) % vocab;
                        tid %= w_rows;
                        for (size_t d = 0; d < cols; ++d) {
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

            out["data"].push_back({{"object", "embedding"}, {"embedding", emb_vector}, {"index", idx}});
        }

        out["usage"] = {{"prompt_tokens", total_prompt_tokens}, {"total_tokens", total_prompt_tokens}};
        metrics_->on_request_end(0.0f);
        res.set_content(out.dump(), "application/json");
    });
}

void Server::handle_health(const httplib::Request &, httplib::Response &res)
{
    nlohmann::json j;
    j["status"] = "ok";
    j["model_loaded"] = has_model();
    j["model_name"] = model_name();
    j["ram_bytes"] = process_ram_bytes();
    j["vram_bytes"] = process_vram_bytes();
    res.set_content(j.dump(), "application/json");
}

void Server::handle_metrics(const httplib::Request &req, httplib::Response &res)
{
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
}

int Server::run()
{
    setup_routes();

    // Dedicated ops listener: /health and /metrics are lock-free and cheap,
    // but on the main port each connection occupies a httplib pool slot for
    // the whole request — under load they can starve behind generations.
    // health_port gives monitoring its own listener that never queues.
    if (config_.health_port != 0 && config_.health_port != config_.port) {
        ops_server_ = std::make_unique<httplib::Server>();
        ops_server_->Get("/health",
                         [this](const httplib::Request &req, httplib::Response &res) { handle_health(req, res); });
        ops_server_->Get("/metrics",
                         [this](const httplib::Request &req, httplib::Response &res) { handle_metrics(req, res); });
        ops_thread_ = std::thread([this] {
            // Same resilience as the main listener: a fatal accept() error
            // closes the socket, so re-bind and retry until stop().
            while (!stop_requested_.load()) {
                ops_server_->listen(host_.c_str(), static_cast<int>(config_.health_port));
                if (stop_requested_.load()) {
                    break;
                }
                spdlog::warn("Ops listener failed; re-binding in 1s");
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
        ops_thread_.detach();
        spdlog::info("Ops endpoints (/health, /metrics) also listening on {}:{}", host_, config_.health_port);
    }

    spdlog::info("Serius server listening on {}:{}", host_, port_);

    // httplib treats any accept() error other than EMFILE/EINTR/EAGAIN as
    // fatal: it closes the listen socket and listen() returns false. Under
    // connection churn (stress ramp, client aborts) that used to kill the
    // whole process silently. Re-bind and retry until stop() is requested;
    // a genuine bind failure at startup still fails fast on the first try.
    if (!server_.bind_to_port(host_.c_str(), static_cast<int>(port_))) {
        spdlog::error("Failed to bind {}:{}", host_, port_);
        return 1;
    }
    while (!stop_requested_.load()) {
        if (server_.listen_after_bind()) {
            return 0;
        }
        if (stop_requested_.load()) {
            return 0;
        }
        spdlog::warn("Listener socket error on {}:{}; re-binding in 1s", host_, port_);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!server_.bind_to_port(host_.c_str(), static_cast<int>(port_))) {
            spdlog::warn("Re-bind of {}:{} failed; retrying", host_, port_);
        }
    }
    return 0;
}

void Server::stop()
{
    stop_requested_.store(true);
    server_.stop();
    if (ops_server_) {
        ops_server_->stop();
    }
    if (ops_thread_.joinable()) {
        ops_thread_.join();
    }
}

std::string extract_last_user_message(const std::string &json_body)
{
    try {
        auto body = nlohmann::json::parse(json_body.empty() ? "{}" : json_body);
        return extract_user_prompt(body);
    }
    catch (...) {
        return "";
    }
}

std::string build_completion_text(const std::string &prompt)
{
    return prompt.empty() ? "Serius API is running." : prompt;
}

}  // namespace server
}  // namespace serius
