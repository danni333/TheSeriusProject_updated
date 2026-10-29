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
        std::string file_path = ui_dir + "/assets/" + filename;

        if (!fs::exists(file_path)) {
            res.status = 404;
            res.set_content("File not found: " + filename, "text/plain");
            return;
        }

        std::string content = read_file(file_path);
        std::string mime_type = get_mime_type(file_path);

        res.set_content(content, mime_type);
    });
}

}  // namespace server
}  // namespace serius


