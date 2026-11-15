#include "model_hub_server.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace serius::hub {

namespace {

// ---------------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4) for model checksums.
// ---------------------------------------------------------------------------
struct Sha256
{
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t total = 0;
    uint8_t buf[64]{};
    size_t buflen = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t *p)
    {
        static const uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i*4]) << 24) | (uint32_t(p[i*4+1]) << 16) |
                   (uint32_t(p[i*4+2]) << 8) | uint32_t(p[i*4+3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    void update(const uint8_t *data, size_t len)
    {
        total += len;
        while (len > 0) {
            const size_t take = std::min(len, 64 - buflen);
            std::memcpy(buf + buflen, data, take);
            buflen += take; data += take; len -= take;
            if (buflen == 64) { block(buf); buflen = 0; }
        }
    }

    std::string final()
    {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (buflen != 56) update(&zero, 1);
        uint8_t lenbuf[8];
        for (int i = 0; i < 8; ++i) lenbuf[i] = uint8_t(bits >> (56 - 8*i));
        update(lenbuf, 8);
        std::ostringstream ss;
        ss << std::hex << std::setfill('0');
        for (uint32_t v : h) ss << std::setw(8) << v;
        return ss.str();
    }
};

std::string sha256_hex(const std::string &data)
{
    Sha256 s;
    s.update(reinterpret_cast<const uint8_t *>(data.data()), data.size());
    return s.final();
}

std::string random_hex(size_t bytes)
{
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes; ++i) {
        ss << std::setw(2) << (rng() & 0xFF);
    }
    return ss.str();
}

std::string sanitize_filename(std::string name)
{
    for (char &c : name) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_')) {
            c = '_';
        }
    }
    if (name.empty() || name == "." || name == "..") { name = "model.bin"; }
    return name;
}

int to_int_param(const httplib::Request &req, const char *name, int fallback)
{
    if (!req.has_param(name)) { return fallback; }
    try {
        return std::stoi(req.get_param_value(name));
    } catch (...) {
        return fallback;
    }
}

nlohmann::json search_result_to_json(const SearchResult &r)
{
    return nlohmann::json{{"model_id", r.model_id},
                          {"name", r.name},
                          {"description", r.description},
                          {"author", r.author},
                          {"score", r.score},
                          {"matched_tags", r.matched_tags},
                          {"highlighted_text", r.highlighted_text}};
}

}  // namespace

ModelHubServer::ModelHubServer(int port)
    : port_(port),
      // Absolute paths: handlers may run after the process cwd changes.
      models_dir_(std::filesystem::absolute("models").string()),
      metadata_db_path_(std::filesystem::absolute("models_metadata.json").string()),
      users_path_(std::filesystem::absolute("hub_users.json").string()),
      metadata_(metadata_db_path_),
      search_(metadata_db_path_)
{
    server_ = std::make_unique<httplib::Server>();
    std::filesystem::create_directories(models_dir_);
    metadata_.initialize();
    search_.initialize();
}

ModelHubServer::~ModelHubServer()
{
    stop();
}

void ModelHubServer::start()
{
    setup_routes();
    spdlog::info("Model Hub Server starting on port {}", port_);
    server_->listen("0.0.0.0", port_);
}

void ModelHubServer::stop()
{
    if (server_) {
        server_->stop();
    }
}

void ModelHubServer::setup_routes()
{
    // Register /v1/models/search before the generic /v1/models/(.*) pattern
    // so httplib's first-match routing doesn't shadow it.
    setup_auth_routes();
    setup_search_routes();
    setup_model_routes();
}

// ---------------------------------------------------------------------------
// Auth: file-based API keys. Empty users store => open mode (default_user).
// ---------------------------------------------------------------------------

nlohmann::json ModelHubServer::load_users() const
{
    std::ifstream in(users_path_);
    if (!in) { return nlohmann::json::object(); }
    try {
        nlohmann::json j;
        in >> j;
        return j.is_object() ? j : nlohmann::json::object();
    } catch (...) {
        return nlohmann::json::object();
    }
}

bool ModelHubServer::save_users(const nlohmann::json &users)
{
    std::ofstream out(users_path_, std::ios::trunc);
    if (!out) { return false; }
    out << users.dump(2);
    return static_cast<bool>(out);
}

std::string ModelHubServer::resolve_user_id(const std::string &api_key) const
{
    const auto users = load_users();
    for (auto it = users.begin(); it != users.end(); ++it) {
        if (it.key() == api_key && it.value().is_object()) {
            return it.value().value("user_id", "");
        }
    }
    return {};
}

std::vector<std::string> ModelHubServer::user_permissions(const std::string &user_id) const
{
    const auto users = load_users();
    for (auto it = users.begin(); it != users.end(); ++it) {
        if (!it.value().is_object()) { continue; }
        if (it.value().value("user_id", "") == user_id &&
            it.value().contains("permissions") && it.value()["permissions"].is_array()) {
            std::vector<std::string> perms;
            for (const auto &p : it.value()["permissions"]) {
                if (p.is_string()) { perms.push_back(p.get<std::string>()); }
            }
            return perms;
        }
    }
    return {};
}

std::string ModelHubServer::register_user(const std::string &username)
{
    if (username.empty()) { return {}; }
    auto users = load_users();
    // reuse existing account for same username
    for (auto it = users.begin(); it != users.end(); ++it) {
        if (it.value().is_object() && it.value().value("username", "") == username) {
            return it.key();
        }
    }
    const std::string api_key = "sk-" + random_hex(16);
    const std::string user_id = "user_" + random_hex(8);
    users[api_key] = {{"user_id", user_id},
                      {"username", username},
                      {"permissions", {"read", "upload", "delete"}},
                      {"created_at", now_iso8601()}};
    if (!save_users(users)) { return {}; }
    spdlog::info("Hub: registered user '{}' ({})", username, user_id);
    return api_key;
}

bool ModelHubServer::authenticate_user(const std::string &token, std::string &user_id)
{
    // Strip "Bearer " prefix if present.
    std::string key = token;
    const std::string prefix = "Bearer ";
    if (key.rfind(prefix, 0) == 0) { key = key.substr(prefix.size()); }
    if (key.empty()) { return false; }

    const auto users = load_users();
    if (users.empty()) {
        // Open mode: no users registered yet.
        user_id = "default_user";
        return true;
    }
    user_id = resolve_user_id(key);
    return !user_id.empty();
}

bool ModelHubServer::check_permissions(const std::string &user_id, const std::string &action,
                                       const std::string &resource)
{
    const auto users = load_users();
    if (users.empty()) { return true; }  // open mode
    if (user_id == "default_user") { return true; }

    const auto perms = user_permissions(user_id);
    const std::string exact = resource.empty() ? action : action + ":" + resource;
    const std::string wildcard = action + ":*";
    for (const auto &p : perms) {
        if (p == "*" || p == action || p == exact || p == wildcard) { return true; }
    }
    return false;
}

void ModelHubServer::setup_auth_routes()
{
    server_->Post("/v1/auth/register", [this](const httplib::Request &req, httplib::Response &res) {
        nlohmann::json body;
        try { body = nlohmann::json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "invalid JSON"}}.dump(), "application/json");
            return;
        }
        const std::string username = body.value("username", "");
        const std::string api_key = register_user(username);
        if (api_key.empty()) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "username required"}}.dump(), "application/json");
            return;
        }
        res.status = 201;
        res.set_content(nlohmann::json{{"api_key", api_key},
                                       {"user_id", resolve_user_id(api_key)}}.dump(),
                        "application/json");
    });

    server_->Get("/v1/auth/me", [this](const httplib::Request &req, httplib::Response &res) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id)) {
            res.status = 401;
            res.set_content(nlohmann::json{{"error", "unauthorized"}}.dump(), "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"user_id", user_id},
                                       {"permissions", user_permissions(user_id)}}.dump(),
                        "application/json");
    });
}

// ---------------------------------------------------------------------------
// Model routes
// ---------------------------------------------------------------------------

void ModelHubServer::setup_model_routes()
{
    server_->Get("/health", [this](const httplib::Request &, httplib::Response &res) {
        int count = 0;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            count = metadata_.get_model_count();
        }
        res.set_content(nlohmann::json{{"status", "ok"}, {"models", count}}.dump(),
                        "application/json");
    });

    server_->Get("/metrics", [this](const httplib::Request &, httplib::Response &res) {
        int models = 0;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            models = metadata_.get_model_count();
        }
        const auto users = load_users();
        res.set_content(nlohmann::json{{"models", models},
                                       {"users", users.size()},
                                       {"uptime_seconds", 0}}.dump(),
                        "application/json");
    });

    server_->Get("/v1/models", [this](const httplib::Request &req, httplib::Response &res) {
        const int limit = to_int_param(req, "limit", 50);
        const int offset = to_int_param(req, "offset", 0);
        const std::string author = req.has_param("author") ? req.get_param_value("author") : "";
        std::vector<ModelMetadata> items;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            items = metadata_.search_models("", {}, author, limit, offset);
        }
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &m : items) { arr.push_back(model_to_json(m)); }
        res.set_content(nlohmann::json{{"data", arr}, {"total", arr.size()}}.dump(),
                        "application/json");
    });

    // Upload model (multipart: "model" file + "metadata" JSON file)
    server_->Post("/v1/models/upload", [this](const httplib::Request &req, httplib::Response &res) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id) ||
            !check_permissions(user_id, "upload")) {
            res.status = 401;
            res.set_content(nlohmann::json{{"error", "unauthorized"}}.dump(), "application/json");
            return;
        }

        if (!req.has_file("model") || !req.has_file("metadata")) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "expected multipart fields 'model' and 'metadata'"}}.dump(),
                            "application/json");
            return;
        }

        const auto &model_file = req.get_file_value("model");
        const auto &metadata_file = req.get_file_value("metadata");

        try {
            ModelMetadata metadata = json_to_model(nlohmann::json::parse(metadata_file.content));
            metadata.id = generate_model_id(metadata.name, metadata.author);
            {
                std::lock_guard<std::mutex> lk(db_mutex_);
                // ensure id uniqueness
                int suffix = 0;
                std::string base = metadata.id;
                while (!metadata_.get_model_metadata(metadata.id).id.empty()) {
                    metadata.id = base + "-" + std::to_string(++suffix);
                }
                metadata.file_size = model_file.content.size();
                metadata.checksum_sha256 = calculate_checksum_bytes(model_file.content);
                metadata.created_at = now_iso8601();
                metadata.updated_at = metadata.created_at;
                metadata.config["file_name"] = sanitize_filename(model_file.filename);
                metadata.config["uploaded_by"] = user_id;

                const std::string model_path =
                    models_dir_ + "/" + metadata.id + "_" +
                    sanitize_filename(model_file.filename);
                std::ofstream out(model_path, std::ios::binary);
                out.write(model_file.content.data(),
                          static_cast<std::streamsize>(model_file.content.size()));
                out.close();
                if (!out) { throw std::runtime_error("failed to write model file"); }

                if (!metadata_.save_model_metadata(metadata)) {
                    throw std::runtime_error("failed to save metadata");
                }
            }
            res.status = 201;
            res.set_content(model_to_json(metadata).dump(), "application/json");
        } catch (const std::exception &e) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    server_->Get("/v1/models/(.*)/download", [this](const httplib::Request &req, httplib::Response &res) {
        const std::string model_id = req.matches[1];

        ModelMetadata metadata;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            metadata = metadata_.get_model_metadata(model_id);
        }
        if (metadata.id.empty()) {
            res.status = 404;
            return;
        }

        std::string file_name;
        if (metadata.config.is_object() && metadata.config.contains("file_name")) {
            file_name = metadata.config["file_name"].get<std::string>();
        }
        std::string model_path = models_dir_ + "/" + model_id + "_" + file_name;
        if (file_name.empty() || !fs::exists(model_path)) {
            // fallback: scan for any file with the model id prefix
            model_path.clear();
            for (const auto &entry : fs::directory_iterator(models_dir_)) {
                const std::string fname = entry.path().filename().string();
                if (fname.rfind(model_id + "_", 0) == 0) {
                    model_path = entry.path().string();
                    break;
                }
            }
        }
        if (model_path.empty() || !fs::exists(model_path)) {
            res.status = 404;
            return;
        }

        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            metadata_.increment_download_count(model_id);
        }

        const uint64_t fsize = fs::file_size(model_path);
        res.set_content_provider(fsize, "application/octet-stream",
                                 [model_path](size_t offset, size_t length, httplib::DataSink &sink) {
                                     std::ifstream file(model_path, std::ios::binary);
                                     if (!file) { return false; }
                                     file.seekg(static_cast<std::streamoff>(offset));
                                     std::vector<char> buffer(length);
                                     file.read(buffer.data(), static_cast<std::streamsize>(length));
                                     sink.write(buffer.data(), static_cast<size_t>(file.gcount()));
                                     return true;
                                 });
    });

    server_->Get("/v1/models/(.*)", [this](const httplib::Request &req, httplib::Response &res) {
        const std::string model_id = req.matches[1];
        ModelMetadata metadata;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            metadata = metadata_.get_model_metadata(model_id);
        }
        if (metadata.id.empty()) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", "model not found"}}.dump(), "application/json");
            return;
        }
        res.set_content(model_to_json(metadata).dump(), "application/json");
    });

    server_->Delete("/v1/models/(.*)", [this](const httplib::Request &req, httplib::Response &res) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id)) {
            res.status = 401;
            res.set_content(nlohmann::json{{"error", "unauthorized"}}.dump(), "application/json");
            return;
        }
        const std::string model_id = req.matches[1];

        ModelMetadata metadata;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            metadata = metadata_.get_model_metadata(model_id);
        }
        if (metadata.id.empty()) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", "model not found"}}.dump(), "application/json");
            return;
        }

        const std::string uploader =
            metadata.config.is_object() && metadata.config.contains("uploaded_by")
                ? metadata.config["uploaded_by"].get<std::string>()
                : "";
        const bool is_owner = (!uploader.empty() && uploader == user_id);
        if (!is_owner && !check_permissions(user_id, "delete", model_id)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "forbidden"}}.dump(), "application/json");
            return;
        }

        std::error_code ec;
        for (const auto &entry : fs::directory_iterator(models_dir_, ec)) {
            const std::string fname = entry.path().filename().string();
            if (fname.rfind(model_id + "_", 0) == 0) {
                fs::remove(entry.path(), ec);
            }
        }
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            metadata_.delete_model_metadata(model_id);
        }
        res.status = 200;
        res.set_content(nlohmann::json{{"deleted", model_id}}.dump(), "application/json");
    });
}

void ModelHubServer::setup_search_routes()
{
    server_->Get("/v1/models/search", [this](const httplib::Request &req, httplib::Response &res) {
        SearchFilter filter;
        filter.query = req.has_param("q") ? req.get_param_value("q") : "";
        filter.author = req.has_param("author") ? req.get_param_value("author") : "";
        filter.model_type = req.has_param("type") ? req.get_param_value("type") : "";
        if (req.has_param("tags")) {
            std::istringstream ss(req.get_param_value("tags"));
            std::string t;
            while (std::getline(ss, t, ',')) {
                if (!t.empty()) { filter.tags.push_back(t); }
            }
        }
        const int limit = to_int_param(req, "limit", 50);
        const int offset = to_int_param(req, "offset", 0);

        std::vector<SearchResult> results;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            results = search_.search_models(filter, limit, offset);
        }
        if (!filter.query.empty()) {
            search_.log_search(filter.query, "anonymous", static_cast<int>(results.size()));
        }

        nlohmann::json arr = nlohmann::json::array();
        for (const auto &r : results) { arr.push_back(search_result_to_json(r)); }
        res.set_content(nlohmann::json{{"data", arr}, {"total", arr.size()}}.dump(),
                        "application/json");
    });

    server_->Get("/v1/search/suggest", [this](const httplib::Request &req, httplib::Response &res) {
        const std::string q = req.has_param("q") ? req.get_param_value("q") : "";
        const int limit = to_int_param(req, "limit", 10);
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &s : search_.get_suggestions(q, limit)) { arr.push_back(s); }
        res.set_content(nlohmann::json{{"suggestions", arr}}.dump(), "application/json");
    });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<ModelMetadata> ModelHubServer::search_models(const std::string &query,
                                                         const std::vector<std::string> &tags,
                                                         int limit, int offset)
{
    std::lock_guard<std::mutex> lk(db_mutex_);
    return metadata_.search_models(query, tags, "", limit, offset);
}

ModelMetadata ModelHubServer::get_model_metadata(const std::string &model_id)
{
    std::lock_guard<std::mutex> lk(db_mutex_);
    return metadata_.get_model_metadata(model_id);
}

bool ModelHubServer::upload_model(const std::string &model_path, const ModelMetadata &metadata)
{
    std::error_code ec;
    if (!fs::exists(model_path, ec)) { return false; }
    const std::string dest = models_dir_ + "/" + metadata.id + "_" +
                             sanitize_filename(fs::path(model_path).filename().string());
    fs::copy_file(model_path, dest, fs::copy_options::overwrite_existing, ec);
    if (ec) { return false; }
    std::lock_guard<std::mutex> lk(db_mutex_);
    return metadata_.save_model_metadata(metadata);
}

bool ModelHubServer::download_model(const std::string &model_id, const std::string &output_path)
{
    ModelMetadata metadata;
    {
        std::lock_guard<std::mutex> lk(db_mutex_);
        metadata = metadata_.get_model_metadata(model_id);
    }
    if (metadata.id.empty()) { return false; }

    std::string file_name =
        metadata.config.is_object() && metadata.config.contains("file_name")
            ? metadata.config["file_name"].get<std::string>()
            : "";
    const std::string src = models_dir_ + "/" + model_id + "_" + file_name;
    std::error_code ec;
    if (file_name.empty() || !fs::exists(src, ec)) {
        for (const auto &entry : fs::directory_iterator(models_dir_, ec)) {
            if (entry.path().filename().string().rfind(model_id + "_", 0) == 0) {
                fs::copy_file(entry.path(), output_path,
                              fs::copy_options::overwrite_existing, ec);
                return !ec;
            }
        }
        return false;
    }
    fs::copy_file(src, output_path, fs::copy_options::overwrite_existing, ec);
    return !ec;
}

bool ModelHubServer::delete_model(const std::string &model_id)
{
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(models_dir_, ec)) {
        if (entry.path().filename().string().rfind(model_id + "_", 0) == 0) {
            fs::remove(entry.path(), ec);
        }
    }
    std::lock_guard<std::mutex> lk(db_mutex_);
    return metadata_.delete_model_metadata(model_id);
}

std::string ModelHubServer::generate_model_id(const std::string &name, const std::string &author)
{
    std::string id;
    for (char c : name + "-" + author) {
        id.push_back(std::tolower(static_cast<unsigned char>(c)));
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) {
            id.back() = '-';
        }
    }
    // collapse repeated dashes and trim
    id.erase(std::unique(id.begin(), id.end(), [](char a, char b) { return a == '-' && b == '-'; }), id.end());
    while (!id.empty() && id.front() == '-') { id.erase(id.begin()); }
    while (!id.empty() && id.back() == '-') { id.pop_back(); }
    if (id.empty()) { id = "model"; }
    return id + "-" + random_hex(4);
}

std::string ModelHubServer::calculate_checksum(const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file) { return {}; }
    std::ostringstream ss;
    ss << file.rdbuf();
    return calculate_checksum_bytes(ss.str());
}

std::string ModelHubServer::calculate_checksum_bytes(const std::string &content)
{
    return sha256_hex(content);
}

nlohmann::json ModelHubServer::model_to_json(const ModelMetadata &metadata)
{
    return serius::hub::model_to_json(metadata);
}

ModelMetadata ModelHubServer::json_to_model(const nlohmann::json &json)
{
    return serius::hub::json_to_model(json);
}

bool ModelHubServer::validate_model_file(const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file) { return false; }
    char magic[4] = {};
    file.read(magic, 4);
    // GGUF magic "GGUF" or safetensors header sanity (>0 header size)
    return file.gcount() == 4;
}

void ModelHubServer::log_access(const std::string &user_id, const std::string &action,
                                const std::string &resource)
{
    spdlog::info("hub access: user={} action={} resource={}", user_id, action, resource);
}

}  // namespace serius::hub
