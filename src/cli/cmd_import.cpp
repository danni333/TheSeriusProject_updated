#include "cli/cmd_import.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "../hub/downloader.h"
#include "cli/args_parser.h"
#include "cli/download_progress.h"

namespace fs = std::filesystem;

namespace serius {
namespace cli {

namespace {

/// Copy a local file or directory into the import output directory.
/// Returns 0 on success, 1 on failure.
int import_local_path(const std::string &source, const std::string &output)
{
    std::error_code ec;
    const fs::path src(source);
    if (!fs::exists(src, ec)) {
        std::cerr << "Error: local path does not exist: " << source << "\n";
        return 1;
    }

    const fs::path dst = fs::path(output) / src.filename();

    // Already imported in place (e.g. source lives inside --output).
    if (fs::exists(dst, ec) && fs::equivalent(src, dst, ec)) {
        std::cout << "Already present at " << dst.string() << "\n";
        return 0;
    }

    // Refuse to copy a directory into its own subtree.
    if (fs::is_directory(src, ec)) {
        const fs::path src_abs = fs::absolute(src, ec);
        const fs::path dst_abs = fs::absolute(dst, ec);
        const auto src_str = src_abs.generic_string();
        const auto dst_str = dst_abs.generic_string();
        if (dst_str.rfind(src_str + "/", 0) == 0) {
            std::cerr << "Error: output directory is inside the source: " << dst.string() << "\n";
            return 1;
        }
    }

    fs::create_directories(output, ec);
    if (ec) {
        std::cerr << "Error: cannot create output directory: " << output << "\n";
        return 1;
    }

    if (fs::is_directory(src, ec)) {
        fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    } else {
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    }
    if (ec) {
        std::cerr << "Error: copy failed: " << ec.message() << "\n";
        return 1;
    }
    std::cout << "Imported " << source << " -> " << dst.string() << "\n";
    return 0;
}

}  // namespace

int cmd_import(int argc, char **argv)
{
    ArgsParser parser("serius import", "Import model from HuggingFace Hub or URL");

    parser.add_flag({"source", "s", "Source (hf://repo_id or URL)", true, ""});
    parser.add_flag({"output", "o", "Output directory", true, "./models"});
    parser.add_flag({"token", "t", "HuggingFace token (optional)", true, ""});
    parser.add_flag({"format", "f", "Output format (gguf, safetensors)", true, "gguf"});

    if (!parser.parse(argc, argv)) {
        std::cerr << "Error: " << parser.error() << "\n";
        parser.print_help();
        return 1;
    }

    if (parser.has("help")) {
        parser.print_help();
        return 0;
    }

    std::string source = parser.get("source");
    std::string output = parser.get("output");
    std::string token = parser.get("token");
    std::string format = parser.get("format");

    if (source.empty()) {
        std::cerr << "Error: --source is required\n";
        return 1;
    }

    hub::ModelDownloader downloader;

    if (source.find("hf://") == 0) {
        // HuggingFace Hub download
        std::string repo_id = source.substr(5);  // Remove "hf://" prefix
        std::cout << "Downloading from HuggingFace: " << repo_id << "\n";
        std::cout << "Output directory: " << output << "\n";

        bool success = downloader.download_hf(repo_id, output, token, make_download_progress());
        finish_download_progress();
        if (success) {
            std::cout << "Download completed successfully\n";
            return 0;
        } else {
            std::cerr << "Download failed\n";
            return 1;
        }
    } else if (source.rfind("http://", 0) == 0 || source.rfind("https://", 0) == 0) {
        // Direct URL download
        std::cout << "Downloading from URL: " << source << "\n";
        std::cout << "Output path: " << output << "\n";

        bool success = downloader.download_url(source, output, make_download_progress());
        finish_download_progress();
        if (success) {
            std::cout << "Download completed successfully\n";
            return 0;
        } else {
            std::cerr << "Download failed\n";
            return 1;
        }
    } else {
        // Local path: directory or file -> copy into the output directory.
        return import_local_path(source, output);
    }
}

}  // namespace cli
}  // namespace serius
