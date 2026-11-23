#include "static_files.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace serius {
namespace server {

namespace {
std::string read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return "";
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

std::string get_mime_type(const std::string &path)
{
    std::string ext = std::filesystem::path(path).extension().string();

    if (ext == ".html") return "text/html";
    if (ext == ".css") return "text/css";
    if (ext == ".js") return "application/javascript";
    if (ext == ".json") return "application/json";
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".svg") return "image/svg+xml";

    return "application/octet-stream";
}
}  // namespace

void setup_static_routes(httplib::Server &server, const std::string &ui_dir)
{
    namespace fs = std::filesystem;

    // Serve index.html at root
    server.Get("/", [ui_dir](const httplib::Request &, httplib::Response &res) {
        std::string index_path = ui_dir + "/index.html";
        std::string content = read_file(index_path);

        if (content.empty()) {
            res.status = 404;
            res.set_content("UI not found. Make sure ui/ directory exists.", "text/plain");
            return;
        }

        res.set_content(content, "text/html");
    });

    // Serve static files from ui/assets/
    server.Get(R"(/assets/(.*))", [ui_dir](const httplib::Request &req, httplib::Response &res) {
        std::string filename = req.matches[1];

        // Reject traversal attempts, then verify the resolved path stays
        // inside the assets root (defense in depth: httplib decodes %2e%2e
        // and does not normalize dot segments).
        if (filename.empty() || filename.find("..") != std::string::npos) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }

        std::error_code ec;
        const fs::path root = fs::weakly_canonical(fs::path(ui_dir) / "assets", ec);
        if (ec) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }
        const fs::path target = fs::weakly_canonical(root / fs::path(filename), ec);
        const std::string root_str = root.string() + static_cast<char>(fs::path::preferred_separator);
        if (ec || target.string().compare(0, root_str.size(), root_str) != 0 || !fs::is_regular_file(target)) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }

        std::string content = read_file(target.string());
        if (content.empty()) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }

        res.set_content(content, get_mime_type(target.string()));
    });
}

}  // namespace server
}  // namespace serius
