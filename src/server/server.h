#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <httplib.h>

#include "../common/defaults.h"
#include "../tokenizer/bpe.h"

namespace serius {
namespace model {
class LLaMAModel;
class QwenModel;
class TransformerModel;
}  // namespace model
namespace infer {
struct KVCache;
}
class BPETokenizer;

namespace server {

class InferenceMetrics;

struct ServerConfig
{
    uint16_t port = defaults::kHttpPort;
    std::string host = "0.0.0.0";
    std::string model_path = "";
    std::string model_type = "llama";  // "llama", "qwen", "transformer"
    std::string model_name = "default";
    size_t max_new_tokens = 256;
    float default_temperature = 0.7f;
    int default_top_k = 40;
    float default_top_p = 0.9f;
    std::string ui_dir = "ui";  // Directory for web UI static files
    /// GPU backend for the loaded LLaMA model: auto = plan from model size vs
    /// free VRAM, cpu = never offload, cuda = require GPU (warn if it can't fit).
    enum class GpuMode { kAuto, kCpu, kCuda } gpu_mode = GpuMode::kAuto;
};

/// A model loaded into the server's registry, keyed by name.
/// Holds the weights, tokenizer and (for LLaMA) a dedicated KV cache —
/// the CUDA decode graph is bound to the cache instance, so each model
/// needs its own.
struct LoadedModel
{
    std::string name;
    std::string type;  // "llama" | "transformer" | "qwen"
    int64_t created = 0;
    std::unique_ptr<model::LLaMAModel> llama;
    std::unique_ptr<model::QwenModel> qwen;
    std::unique_ptr<model::TransformerModel> transformer;
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<infer::KVCache> kv_cache;
    /// Token ids currently materialized in kv_cache (prompt + generated),
    /// used for prefix reuse across chat turns. Empty = cache cold.
    std::vector<TokenId> cached_token_ids;
    LoadedModel() = default;
    LoadedModel(LoadedModel &&) = default;
    LoadedModel &operator=(LoadedModel &&) = default;
    LoadedModel(const LoadedModel &) = delete;
    LoadedModel &operator=(const LoadedModel &) = delete;
    ~LoadedModel();
};

/// HTTP server for inference API (OpenAI-compatible endpoints).
/// Uses cpp-httplib; run() blocks until shutdown.
class Server
{
public:
    explicit Server(uint16_t port = defaults::kHttpPort);
    explicit Server(const ServerConfig &config);
    ~Server();

    /// Load a model and tokenizer from directory or config file.
    /// `name` overrides the registry key; when empty the name is derived
    /// from the path (file stem or directory name). Re-loading an existing
    /// name fails; use a different name or unload first.
    bool load_model(const std::string &model_dir_or_config, const std::string &type = "llama",
                    const std::string &name = "");

    /// Unload a model by registry name. Returns false if not found.
    /// Blocks until any in-flight inference on that model finishes.
    bool unload_model(const std::string &name);

    /// Direct setters for pre-constructed models (useful for testing or programmatic setup)
    void set_llama_model(std::unique_ptr<model::LLaMAModel> model, std::unique_ptr<BPETokenizer> tokenizer = nullptr);
    void set_transformer_model(std::unique_ptr<model::TransformerModel> model, std::unique_ptr<BPETokenizer> tokenizer = nullptr);

    /// Check if a model is currently loaded
    bool has_model() const;

    /// Get or set model name reported by /v1/models
    void set_model_name(const std::string &name);
    std::string model_name() const;

    /// Start listening; blocks until stop or error.
    /// Returns 0 on normal stop, non-zero on error.
    int run();

    /// Stop the server (thread-safe signal to exit).
    void stop();

    uint16_t port() const { return port_; }
    InferenceMetrics &metrics();

private:
    void setup_routes();

    /// Resolve a requested model name to a registry entry.
    /// Empty/"default-ish" names fall back to model_name_ then the first
    /// loaded model. Caller must hold inference_mutex_.
    LoadedModel *find_model(const std::string &name);

    /// Load a GGUF file into a LoadedModel (no registry insert, no lock).
    LoadedModel load_llama_gguf(const std::string &gguf_path);
    /// Load a LLaMA model from a directory or config.json path.
    LoadedModel load_llama_dir(const std::string &model_dir_or_config);
    /// Load a TransformerModel from a directory or config.json path.
    LoadedModel load_transformer_entry(const std::string &model_dir_or_config);
    /// Pick a unique registry name: explicit > derived > model_name_.
    std::string pick_model_name(const std::string &explicit_name, const std::string &derived);
    /// Offload a LLaMA model to GPU when it fits; logs and stays on CPU
    /// otherwise. No-op for non-LLaMA or when gpu_mode is kCpu.
    void maybe_offload_to_gpu(model::LLaMAModel &model);

    uint16_t port_;
    std::string host_;
    ServerConfig config_;
    httplib::Server server_;

    std::string model_name_ = "default";
    std::string model_type_ = "llama";
    /// Registry of loaded models, keyed by name. model_name_ is the
    /// default/primary name used when a request doesn't specify one.
    std::map<std::string, LoadedModel> models_;
    /// Registry name of the most recent successful load_model() call.
    std::string last_loaded_name_;
    std::unique_ptr<InferenceMetrics> metrics_;
    /// Serializes load_model() calls among themselves; the slow weight
    /// loading runs WITHOUT inference_mutex_ so inference keeps serving
    /// other models while a GGUF loads. Only the registry insert takes
    /// inference_mutex_.
    std::mutex load_mutex_;
    mutable std::mutex inference_mutex_;
};

/// Build a small OpenAI-compatible completion payload from the last user prompt.
std::string build_completion_text(const std::string &prompt);

/// Extract the last user message from an OpenAI-like chat JSON payload.
std::string extract_last_user_message(const std::string &json_body);

}  // namespace server
}  // namespace serius

