#pragma once

#include <functional>
#include <string>

namespace serius {
namespace hub {

/// Progress callback for download operations.
using ProgressFn = std::function<void(size_t downloaded, size_t total)>;

/// Model downloader for HuggingFace Hub and direct URLs.
class ModelDownloader
{
   public:
    ModelDownloader();
    ~ModelDownloader() = default;

    /// Download model from HuggingFace Hub.
    /// @param repo_id Repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
    /// @param output_dir Directory to save the model
    /// @param token Optional HuggingFace authentication token
    /// @param progress Optional progress callback
    /// @return true on success
    bool download_hf(const std::string &repo_id,
                     const std::string &output_dir,
                     const std::string &token = "",
                     ProgressFn progress = nullptr);

    /// Download from direct URL.
    /// @param url URL to download from
    /// @param output_path Path to save the file
    /// @param progress Optional progress callback
    /// @return true on success
    bool download_url(const std::string &url,
                      const std::string &output_path,
                      ProgressFn progress = nullptr);

    /// Set timeout for HTTP requests (in seconds).
    void set_timeout(int seconds) { timeout_seconds_ = seconds; }

   private:
    int timeout_seconds_ = 300;

    /// Download a single file.
    bool download_file(const std::string &url, const std::string &path, ProgressFn progress = nullptr);

    /// Fetch model info from HuggingFace API.
    std::string fetch_model_info(const std::string &repo_id, const std::string &token = "");

    /// Construct HuggingFace file URL.
    std::string build_hf_url(const std::string &repo_id, const std::string &filename);
};

}  // namespace hub
}  // namespace serius
