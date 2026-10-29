#include "cli/cmd_import.h"
#include "cli/args_parser.h"
#include "../hub/downloader.h"

#include <iostream>

namespace serius {
namespace cli {

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

        bool success = downloader.download_hf(repo_id, output, token);
        if (success) {
            std::cout << "Download completed successfully\n";
            return 0;
        } else {
            std::cerr << "Download failed\n";
            return 1;
        }
    } else {
        // Direct URL download
        std::cout << "Downloading from URL: " << source << "\n";
        std::cout << "Output path: " << output << "\n";

        bool success = downloader.download_url(source, output);
        if (success) {
            std::cout << "Download completed successfully\n";
            return 0;
        } else {
            std::cerr << "Download failed\n";
            return 1;
        }
    }
}

}  // namespace cli
}  // namespace serius
