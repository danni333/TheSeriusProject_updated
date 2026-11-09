#pragma once

#include <functional>
#include <string>
#include <vector>

namespace serius {
namespace hub {

/// Progress callback for download operations.
using ProgressFn = std::function<void(size_t downloaded, size_t total)>;

/// A file entry in a HuggingFace repo (from the tree API).
struct HubFile {
    std::string path;   // path relative to repo root
    size_t size = 0;    // bytes, 0 if unknown
};

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

    /// List files in a HF repo via the tree API (recursive), with sizes.
    /// Returns empty vector on failure.
    std::vector<HubFile> list_hf_files(const std::string &repo_id,
                                       const std::string &token = "");

    /// Download a single file from a HF repo to output_path.
    bool download_hf_file(const std::string &repo_id, const std::string &filename,
                          const std::string &output_path, const std::string &token = "",
                          ProgressFn progress = nullptr);

   private:
    int timeout_seconds_ = 300;

    /// Download a single file.
    bool download_file(const std::string &url, const std::string &path, ProgressFn progress = nullptr,
                       const std::string &token = "");

    /// Fetch model info from HuggingFace API.
    std::string fetch_model_info(const std::string &repo_id, const std::string &token = "");

    /// Construct HuggingFace file URL.
    std::string build_hf_url(const std::string &repo_id, const std::string &filename);
};

}  // namespace hub
}  // namespace serius
