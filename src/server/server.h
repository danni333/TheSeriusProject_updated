#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <httplib.h>

#include "../common/defaults.h"

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

/// HTTP server for inference API (OpenAI-compatible endpoints).
/// Uses cpp-httplib; run() blocks until shutdown.
class Server
{
public:
    explicit Server(uint16_t port = defaults::kHttpPort);
    explicit Server(const ServerConfig &config);
    ~Server();

    /// Load a model and tokenizer from directory or config file
    bool load_model(const std::string &model_dir_or_config, const std::string &type = "llama");

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

    /// Load a LLaMA model directly from a .gguf file (config, tokenizer and
    /// weights all come from the GGUF container).
    bool load_llama_from_gguf(const std::string &gguf_path);

    /// Offload the loaded LLaMA model to GPU when it fits in free VRAM;
    /// logs a warning and stays on CPU otherwise. No-op without a model.
    void maybe_offload_to_gpu();

    uint16_t port_;
    std::string host_;
    ServerConfig config_;
    httplib::Server server_;

    std::string model_name_ = "default";
    std::string model_type_ = "llama";
    std::unique_ptr<model::LLaMAModel> llama_model_;
    std::unique_ptr<model::QwenModel> qwen_model_;
    std::unique_ptr<model::TransformerModel> transformer_model_;
    std::unique_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<InferenceMetrics> metrics_;
    /// Persistent KV cache for LLaMA decode: keeping the same instance across
    /// requests preserves the GPU arena and the captured CUDA decode graph
    /// (keyed by cache instance_id) instead of recapturing per request.
    std::unique_ptr<infer::KVCache> llama_cache_;
    mutable std::mutex inference_mutex_;
};

/// Build a small OpenAI-compatible completion payload from the last user prompt.
std::string build_completion_text(const std::string &prompt);

/// Extract the last user message from an OpenAI-like chat JSON payload.
std::string extract_last_user_message(const std::string &json_body);

}  // namespace server
}  // namespace serius

