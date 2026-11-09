#include "downloader.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define popen _popen
#define pclose _pclose
#endif

namespace serius {
namespace hub {

ModelDownloader::ModelDownloader()
{
}

std::string ModelDownloader::build_hf_url(const std::string &repo_id, const std::string &filename)
{
    // Convert repo_id to URL format: replace "/" with "/" in URL
    // HuggingFace URL pattern: https://huggingface.co/{repo_id}/resolve/main/{filename}
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + filename;
}

std::string ModelDownloader::fetch_model_info(const std::string &repo_id, const std::string &token)
{
    std::string url = "https://huggingface.co/api/models/" + repo_id;

    std::ostringstream cmd;
    cmd << "curl -s -X GET \"" << url << "\"";

    if (!token.empty()) {
        cmd << " -H \"Authorization: Bearer " << token << "\"";
    }

    FILE *pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        return "";
    }

    char buffer[4096];
    std::string response;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        response += buffer;
    }
    pclose(pipe);

    return response;
}

bool ModelDownloader::download_file(const std::string &url, const std::string &path, ProgressFn progress,
                                    const std::string &token)
{
    // Create output directory if needed
    std::filesystem::path output_path(path);
    std::filesystem::create_directories(output_path.parent_path());

    // Use curl for download: -L follows redirects (HF resolve → CDN),
    // --retry covers transient failures, -C - resumes partial downloads.
    std::ostringstream cmd;
    cmd << "curl -sSL --retry 3 -C - -o \"" << path << "\" \"" << url << "\"";
    if (!token.empty()) {
        cmd << " -H \"Authorization: Bearer " << token << "\"";
    }

    if (progress) {
        // Without libcurl, we can't get real-time progress easily
        // For now, just call the callback with 0/0
        progress(0, 0);
    }

    int result = std::system(cmd.str().c_str());

    if (progress) {
        // Call completion callback
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file) {
            size_t size = file.tellg();
            progress(size, size);
        }
    }

    return (result == 0);
}

bool ModelDownloader::download_hf(const std::string &repo_id,
                                 const std::string &output_dir,
                                 const std::string &token,
                                 ProgressFn progress)
{
    // Fetch model info to get file list
    std::string info = fetch_model_info(repo_id, token);
    if (info.empty()) {
        return false;
    }

    // For now, we'll try to download common files
    // In a full implementation, we'd parse the JSON response to get the file list
    std::vector<std::string> common_files = {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "vocab.json"
    };

    bool all_success = true;
    for (const auto &filename : common_files) {
        std::string url = build_hf_url(repo_id, filename);
        std::string output_path = output_dir + "/" + filename;

        if (!download_file(url, output_path, progress)) {
            // Some files might not exist, that's okay
            // But if essential files fail, we should report it
        }
    }

    return all_success;
}

std::vector<HubFile> ModelDownloader::list_hf_files(const std::string &repo_id,
                                                    const std::string &token)
{
    // GET /api/models/{repo}/tree/main?recursive=true → [{type,path,size,...}]
    std::string url = "https://huggingface.co/api/models/" + repo_id + "/tree/main?recursive=true";

    std::ostringstream cmd;
    cmd << "curl -s -X GET \"" << url << "\"";
    if (!token.empty()) {
        cmd << " -H \"Authorization: Bearer " << token << "\"";
    }

    FILE *pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) return {};

    std::string response;
    char buffer[8192];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        response += buffer;
    }
    pclose(pipe);
    if (response.empty()) return {};

    std::vector<HubFile> files;
    try {
        auto j = nlohmann::json::parse(response);
        if (!j.is_array()) return {};
        for (const auto &entry : j) {
            if (!entry.is_object()) continue;
            if (entry.value("type", "") != "file") continue;
            HubFile f;
            f.path = entry.value("path", "");
            f.size = entry.value("size", (size_t)0);
            if (!f.path.empty()) files.push_back(std::move(f));
        }
    } catch (...) {
        return {};
    }
    return files;
}

bool ModelDownloader::download_hf_file(const std::string &repo_id, const std::string &filename,
                                       const std::string &output_path, const std::string &token,
                                       ProgressFn progress)
{
    std::string url = build_hf_url(repo_id, filename);
    return download_file(url, output_path, progress, token);
}

bool ModelDownloader::download_url(const std::string &url,
                                  const std::string &output_path,
                                  ProgressFn progress)
{
    return download_file(url, output_path, progress);
}

}  // namespace hub
}  // namespace serius
