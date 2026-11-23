#include "../common/defaults.h"
#include "metadata_manager.h"
#include "model_hub_server.h"
// #include "auth_manager.h"
#include <signal.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "plugin_manager.h"
#include "search_engine.h"

namespace serius::hub {

// Non-static: main() lives at global scope and reaches it qualified.
std::unique_ptr<ModelHubServer> g_server;
static std::unique_ptr<MetadataManager> g_metadata_manager;
// static std::unique_ptr<AuthManager> g_auth_manager;
static std::unique_ptr<PluginManager> g_plugin_manager;
static std::unique_ptr<SearchEngine> g_search_engine;

void signal_handler(int signal)
{
    std::cout << "\nReceived signal " << signal << ", shutting down...\n";
    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

bool initialize_components(int port)
{
    // Initialize metadata manager
    g_metadata_manager = std::make_unique<MetadataManager>("hub_metadata.db");
    if (!g_metadata_manager->initialize()) {
        std::cerr << "Failed to initialize metadata manager\n";
        return false;
    }

    // Initialize auth manager - DISABLED
    /*
    g_auth_manager = std::make_unique<AuthManager>("hub_auth.db", "your-jwt-secret-key");
    if (!g_auth_manager->initialize()) {
        std::cerr << "Failed to initialize auth manager\n";
        return false;
    }
    */

    // Initialize plugin manager
    g_plugin_manager = std::make_unique<PluginManager>();

    // Initialize search engine
    g_search_engine = std::make_unique<SearchEngine>("hub_search.db");
    if (!g_search_engine->initialize()) {
        std::cerr << "Failed to initialize search engine\n";
        return false;
    }

    // Initialize hub server
    g_server = std::make_unique<ModelHubServer>(port);

    return true;
}

void print_usage()
{
    std::cout << "Serius Model Hub Server\n";
    std::cout << "Usage: serius_hub_server [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --port <port>     Server port (default: " << serius::defaults::kHubPort << ")\n";
    std::cout << "  --help            Show this help message\n";
}

}  // namespace serius::hub

int main(int argc, char *argv[])
{
    int port = serius::defaults::kHubPort;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            serius::hub::print_usage();
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            serius::hub::print_usage();
            return 1;
        }
    }

    // Setup signal handlers
    signal(SIGINT, serius::hub::signal_handler);
    signal(SIGTERM, serius::hub::signal_handler);

    // Initialize components
    if (!serius::hub::initialize_components(port)) {
        std::cerr << "Failed to initialize hub components\n";
        return 1;
    }

    std::cout << "Serius Model Hub Server starting on port " << port << "\n";
    std::cout << "Available endpoints:\n";
    std::cout << "  GET  /health\n";
    std::cout << "  GET  /metrics\n";
    std::cout << "  POST /v1/auth/register\n";
    std::cout << "  GET  /v1/auth/me\n";
    std::cout << "  GET  /v1/models\n";
    std::cout << "  GET  /v1/models/search?q=<query>\n";
    std::cout << "  GET  /v1/search/suggest?q=<query>\n";
    std::cout << "  GET  /v1/models/<model_id>\n";
    std::cout << "  POST /v1/models/upload   (multipart: 'model' + 'metadata')\n";
    std::cout << "  GET  /v1/models/<model_id>/download\n";
    std::cout << "  DELETE /v1/models/<model_id>   (owner or 'delete' permission)\n";
    std::cout << "\nPress Ctrl+C to stop the server\n";

    // Start server
    try {
        serius::hub::g_server->start();
    }
    catch (const std::exception &e) {
        std::cerr << "Server error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
