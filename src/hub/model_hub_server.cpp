// rand_s is opt-in in the UCRT: it is only declared when _CRT_RAND_S is set
// before the first <stdlib.h> include in this TU.
#if defined(_WIN32) && !defined(_CRT_RAND_S)
#define _CRT_RAND_S
#endif

#include "model_hub_server.h"

#include <spdlog/spdlog.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
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
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t total = 0;
    uint8_t buf[64]{};
    size_t buflen = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t *p)
    {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) | (uint32_t(p[i * 4 + 2]) << 8) |
                   uint32_t(p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    void update(const uint8_t *data, size_t len)
    {
        total += len;
        while (len > 0) {
            const size_t take = std::min(len, 64 - buflen);
            std::memcpy(buf + buflen, data, take);
            buflen += take;
            data += take;
            len -= take;
            if (buflen == 64) {
                block(buf);
                buflen = 0;
            }
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
        for (int i = 0; i < 8; ++i) lenbuf[i] = uint8_t(bits >> (56 - 8 * i));
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

// Cryptographically secure random bytes for API keys / user ids.
void fill_random(uint8_t *buf, size_t len)
{
#ifdef _WIN32
    for (size_t i = 0; i < len; ++i) {
        unsigned int r = 0;
        rand_s(&r);
        buf[i] = static_cast<uint8_t>(r & 0xFF);
    }
#else
    if (FILE *f = std::fopen("/dev/urandom", "rb")) {
        if (std::fread(buf, 1, len, f) == len) {
            std::fclose(f);
            return;
        }
        std::fclose(f);
    }
    // Last-resort fallback (should never happen on a sane system).
    std::random_device rd;
    for (size_t i = 0; i < len; ++i) {
        buf[i] = static_cast<uint8_t>(rd() & 0xFF);
    }
#endif
}

std::string random_hex(size_t bytes)
{
    std::string out;
    out.reserve(bytes * 2);
    static const char *kHex = "0123456789abcdef";
    std::vector<uint8_t> raw(bytes);
    fill_random(raw.data(), bytes);
    for (uint8_t b : raw) {
        out += kHex[b >> 4];
        out += kHex[b & 0xF];
    }
    return out;
}

// Open mode (no registered users) only grants full access to loopback
// clients; remote callers get 401 on mutating routes.
bool is_loopback_addr(const std::string &addr)
{
    if (addr.empty()) {
        return true;
    }  // non-network callers (CLI, tests)
    // IPv4-mapped IPv6 loopback (::ffff:127.x.x.x) counts as loopback too.
    std::string a = addr;
    const std::string v4mapped = "::ffff:";
    if (a.rfind(v4mapped, 0) == 0) {
        a = a.substr(v4mapped.size());
    }
    return a == "::1" || a.rfind("127.", 0) == 0 || a == "localhost";
}

std::string sanitize_filename(std::string name)
{
    for (char &c : name) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_')) {
            c = '_';
        }
    }
    if (name.empty() || name == "." || name == "..") {
        name = "model.bin";
    }
    return name;
}

int to_int_param(const httplib::Request &req, const char *name, int fallback)
{
    if (!req.has_param(name)) {
        return fallback;
    }
    try {
        return std::stoi(req.get_param_value(name));
    }
    catch (...) {
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
    if (server_->bind_to_port("0.0.0.0", port_) && server_->listen_after_bind()) {
        return;
    }
    spdlog::error("Model Hub Server failed to listen on port {}", port_);
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

nlohmann::json ModelHubServer::load_users_locked() const
{
    std::ifstream in(users_path_);
    if (!in) {
        return nlohmann::json::object();
    }
    try {
        nlohmann::json j;
        in >> j;
        return j.is_object() ? j : nlohmann::json::object();
    }
    catch (...) {
        return nlohmann::json::object();
    }
}

bool ModelHubServer::save_users_locked(const nlohmann::json &users)
{
    const fs::path target(users_path_);
    const fs::path tmp = users_path_ + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
        if (!out) {
            return false;
        }
        out << users.dump(2);
        out.flush();
        if (!out) {
            std::error_code rm_ec;
            fs::remove(tmp, rm_ec);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        // Windows rename fails if the target exists: remove then retry.
        fs::remove(target, ec);
        ec.clear();
        fs::rename(tmp, target, ec);
        if (ec) {
            std::error_code rm_ec;
            fs::remove(tmp, rm_ec);
            return false;
        }
    }
    return true;
}

nlohmann::json ModelHubServer::load_users() const
{
    // mtime-keyed cache: auth resolves a user on every request, so avoid
    // re-reading and re-parsing hub_users.json when it hasn't changed.
    std::error_code ec;
    const auto mtime = fs::last_write_time(users_path_, ec);
    {
        std::lock_guard<std::mutex> clk(users_cache_mutex_);
        if (users_cache_valid_ && !ec && mtime == users_cache_mtime_) {
            return users_cache_;
        }
    }
    std::lock_guard<std::mutex> lk(users_mutex_);
    // Re-check under the write lock: another thread may have refreshed it.
    {
        std::lock_guard<std::mutex> clk(users_cache_mutex_);
        if (users_cache_valid_ && !ec && mtime == users_cache_mtime_) {
            return users_cache_;
        }
    }
    auto fresh = load_users_locked();
    {
        std::lock_guard<std::mutex> clk(users_cache_mutex_);
        users_cache_ = fresh;
        users_cache_mtime_ = mtime;
        users_cache_valid_ = !ec;
    }
    return fresh;
}

bool ModelHubServer::save_users(const nlohmann::json &users)
{
    std::lock_guard<std::mutex> lk(users_mutex_);
    return save_users_locked(users);
}

std::string ModelHubServer::resolve_user_id(const std::string &api_key) const
{
    const auto users = load_users();
    // New accounts are stored under sha256(api_key); entries written by older
    // versions may still be keyed by the raw key, so check both.
    const std::string key_hash = sha256_hex(api_key);
    for (auto it = users.begin(); it != users.end(); ++it) {
        if ((it.key() == key_hash || it.key() == api_key) && it.value().is_object()) {
            return it.value().value("user_id", "");
        }
    }
    return {};
}

std::vector<std::string> ModelHubServer::user_permissions(const std::string &user_id) const
{
    const auto users = load_users();
    for (auto it = users.begin(); it != users.end(); ++it) {
        if (!it.value().is_object()) {
            continue;
        }
        if (it.value().value("user_id", "") == user_id && it.value().contains("permissions") &&
            it.value()["permissions"].is_array()) {
            std::vector<std::string> perms;
            for (const auto &p : it.value()["permissions"]) {
                if (p.is_string()) {
                    perms.push_back(p.get<std::string>());
                }
            }
            return perms;
        }
    }
    return {};
}

int ModelHubServer::register_user(const std::string &username, std::string &api_key)
{
    if (username.empty()) {
        return 1;
    }
    // Serialize the whole check-then-write cycle: without the lock two
    // concurrent registrations of the same username could both pass the
    // conflict check and the loser's api_key would never be persisted.
    std::lock_guard<std::mutex> lk(users_mutex_);
    auto users = load_users_locked();
    // An existing username is a conflict: never return the stored key, that
    // would allow account takeover by anyone who can reach /v1/auth/register.
    for (auto it = users.begin(); it != users.end(); ++it) {
        if (it.value().is_object() && it.value().value("username", "") == username) {
            return 2;
        }
    }
    api_key = "sk-" + random_hex(16);
    const std::string key_hash = sha256_hex(api_key);
    const std::string user_id = "user_" + random_hex(8);
    users[key_hash] = {{"user_id", user_id},
                       {"username", username},
                       {"permissions", {"read", "upload"}},
                       {"created_at", now_iso8601()}};
    if (!save_users_locked(users)) {
        api_key.clear();
        return 1;
    }
    spdlog::info("Hub: registered user '{}' ({})", username, user_id);
    return 0;
}

bool ModelHubServer::authenticate_user(const std::string &token, std::string &user_id, const std::string &remote_addr)
{
    // Strip "Bearer " prefix if present.
    std::string key = token;
    const std::string prefix = "Bearer ";
    if (key.rfind(prefix, 0) == 0) {
        key = key.substr(prefix.size());
    }
    if (key.empty()) {
        return false;
    }

    const auto users = load_users();
    if (users.empty()) {
        // Open mode: no users registered yet. Full access is limited to
        // loopback clients so an exposed hub port can't be claimed remotely.
        if (!is_loopback_addr(remote_addr)) {
            return false;
        }
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
    if (users.empty()) {
        return true;
    }  // open mode
    if (user_id == "default_user") {
        return true;
    }

    const auto perms = user_permissions(user_id);
    const std::string exact = resource.empty() ? action : action + ":" + resource;
    const std::string wildcard = action + ":*";
    for (const auto &p : perms) {
        if (p == "*" || p == action || p == exact || p == wildcard) {
            return true;
        }
    }
    return false;
}

bool ModelHubServer::rate_limit_ok(const std::string &bucket, const std::string &remote_addr, int max_per_minute)
{
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const int64_t window = now / 60;
    const std::string key = bucket + "|" + remote_addr;
    std::lock_guard<std::mutex> lk(rate_mutex_);
    auto &slot = rate_windows_[key];
    if (slot.first != window) {
        slot = {window, 0};
    }
    if (++slot.second > max_per_minute) {
        return false;
    }
    // Opportunistic cleanup so the map can't grow unboundedly.
    if (rate_windows_.size() > 4096) {
        for (auto it = rate_windows_.begin(); it != rate_windows_.end();) {
            it = (it->second.first < window - 1) ? rate_windows_.erase(it) : std::next(it);
        }
    }
    return true;
}

void ModelHubServer::setup_auth_routes()
{
    server_->Post("/v1/auth/register", [this](const httplib::Request &req, httplib::Response &res) {
        // Registration is unauthenticated by design; cap the rate so a remote
        // client can't fill hub_users.json (each write rewrites the whole
        // file) or mass-create accounts.
        if (!rate_limit_ok("register", req.remote_addr, 20)) {
            res.status = 429;
            res.set_content(nlohmann::json{{"error", "rate limit exceeded"}}.dump(), "application/json");
            return;
        }
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        }
        catch (...) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "invalid JSON"}}.dump(), "application/json");
            return;
        }
        const std::string username = body.value("username", "");
        std::string api_key;
        const int rc = register_user(username, api_key);
        if (rc == 2) {
            res.status = 409;
            res.set_content(nlohmann::json{{"error", "username already registered"}}.dump(), "application/json");
            return;
        }
        if (rc != 0) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "username required"}}.dump(), "application/json");
            return;
        }
        res.status = 201;
        res.set_content(nlohmann::json{{"api_key", api_key}, {"user_id", resolve_user_id(api_key)}}.dump(),
                        "application/json");
    });

    server_->Get("/v1/auth/me", [this](const httplib::Request &req, httplib::Response &res) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id, req.remote_addr)) {
            res.status = 401;
            res.set_content(nlohmann::json{{"error", "unauthorized"}}.dump(), "application/json");
            return;
        }
        res.set_content(nlohmann::json{{"user_id", user_id}, {"permissions", user_permissions(user_id)}}.dump(),
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
        res.set_content(nlohmann::json{{"status", "ok"}, {"models", count}}.dump(), "application/json");
    });

    server_->Get("/metrics", [this](const httplib::Request &, httplib::Response &res) {
        int models = 0;
        {
            std::lock_guard<std::mutex> lk(db_mutex_);
            models = metadata_.get_model_count();
        }
        const auto users = load_users();
        res.set_content(nlohmann::json{{"models", models}, {"users", users.size()}, {"uptime_seconds", 0}}.dump(),
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
        for (const auto &m : items) {
            arr.push_back(model_to_json(m));
        }
        res.set_content(nlohmann::json{{"data", arr}, {"total", arr.size()}}.dump(), "application/json");
    });

    // Upload model (streaming multipart: "model" file + "metadata" JSON file).
    // Streams the model payload straight to a temporary file while computing
    // SHA-256 incrementally so large uploads never buffer the entire file in RAM.
    server_->Post("/v1/models/upload", [this](const httplib::Request &req, httplib::Response &res,
                                              const httplib::ContentReader &content_reader) {
        std::string user_id;
        if (!authenticate_user(req.get_header_value("Authorization"), user_id, req.remote_addr) ||
            !check_permissions(user_id, "upload")) {
            res.status = 401;
            res.set_content(nlohmann::json{{"error", "unauthorized"}}.dump(), "application/json");
            return;
        }
        if (!req.is_multipart_form_data()) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "expected multipart/form-data"}}.dump(), "application/json");
            return;
        }

        constexpr size_t kMaxMetadataBytes = 1024 * 1024;  // 1 MiB
        const std::string tmp_path = models_dir_ + "/.upload_" + random_hex(8) + ".tmp";
        struct TmpFileGuard {
            std::string path;
            bool keep = false;
            ~TmpFileGuard()
            {
                if (!keep && !path.empty()) {
                    std::error_code ec;
                    fs::remove(path, ec);
                }
            }
        } tmp_guard{tmp_path, false};

        std::ofstream model_out;
        Sha256 sha;
        size_t model_bytes = 0;
        bool has_model = false;
        bool has_metadata = false;
        bool duplicate_field = false;
        bool metadata_too_large = false;
        bool write_failed = false;
        std::string model_filename;
        std::string metadata_json;
        std::string current_field;

        const bool read_ok = content_reader(
            [&](const httplib::FormData &file) {
                current_field = file.name;
                if (file.name == "model") {
                    if (has_model) {
                        duplicate_field = true;
                        return false;
                    }
                    has_model = true;
                    model_filename = sanitize_filename(file.filename);
                    model_out.open(tmp_path, std::ios::binary | std::ios::trunc);
                    if (!model_out) {
                        write_failed = true;
                        return false;
                    }
                } else if (file.name == "metadata") {
                    if (has_metadata) {
                        duplicate_field = true;
                        return false;
                    }
                    has_metadata = true;
                    metadata_json.clear();
                }
                return true;
            },
            [&](const char *data, size_t data_length) {
                if (current_field == "model") {
                    if (!model_out.is_open()) {
                        write_failed = true;
                        return false;
                    }
                    if (data_length > 0) {
                        model_out.write(data, static_cast<std::streamsize>(data_length));
                        if (!model_out) {
                            write_failed = true;
                            return false;
                        }
                        sha.update(reinterpret_cast<const uint8_t *>(data), data_length);
                        model_bytes += data_length;
                    }
                } else if (current_field == "metadata") {
                    if (metadata_json.size() + data_length > kMaxMetadataBytes) {
                        metadata_too_large = true;
                        return false;
                    }
                    metadata_json.append(data, data_length);
                }
                return true;
            });

        if (model_out.is_open()) {
            model_out.flush();
            model_out.close();
            if (!model_out) {
                write_failed = true;
            }
        }
        if (metadata_too_large) {
            res.status = 413;
            res.set_content(nlohmann::json{{"error", "metadata field too large"}}.dump(), "application/json");
            return;
        }
        if (write_failed) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", "failed to stream model file"}}.dump(), "application/json");
            return;
        }
        if (duplicate_field || !read_ok || !has_model || !has_metadata) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", "expected multipart fields 'model' and 'metadata'"}}.dump(),
                            "application/json");
            return;
        }

        try {
            ModelMetadata metadata = json_to_model(nlohmann::json::parse(metadata_json));
            metadata.id = generate_model_id(metadata.name, metadata.author);
            {
                std::lock_guard<std::mutex> lk(db_mutex_);
                // ensure id uniqueness
                int suffix = 0;
                std::string base = metadata.id;
                while (!metadata_.get_model_metadata(metadata.id).id.empty()) {
                    metadata.id = base + "-" + std::to_string(++suffix);
                }
                metadata.file_size = model_bytes;
                metadata.checksum_sha256 = sha.final();
                metadata.created_at = now_iso8601();
                metadata.updated_at = metadata.created_at;
                metadata.config["file_name"] = model_filename;
                metadata.config["uploaded_by"] = user_id;
                const std::string model_path = models_dir_ + "/" + metadata.id + "_" + model_filename;
                std::error_code ec;
                fs::rename(tmp_path, model_path, ec);
                if (ec) {
                    throw std::runtime_error("failed to finalize model file: " + ec.message());
                }
                tmp_guard.keep = true;
                if (!metadata_.save_model_metadata(metadata)) {
                    fs::remove(model_path, ec);
                    throw std::runtime_error("failed to save metadata");
                }
            }
            res.status = 201;
            res.set_content(model_to_json(metadata).dump(), "application/json");
        }
        catch (const nlohmann::json::exception &e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", std::string("invalid metadata JSON: ") + e.what()}}.dump(),
                            "application/json");
        }
        catch (const std::exception &e) {
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
                                 [model_path, fsize](size_t offset, size_t length, httplib::DataSink &sink) {
                                     if (offset >= fsize) {
                                         return false;
                                     }
                                     // Clamp to the real file size: a client
                                     // asking for an oversized range must not
                                     // trigger a huge allocation.
                                     const size_t avail = static_cast<size_t>(fsize - offset);
                                     const size_t to_read = std::min(length, avail);
                                     std::ifstream file(model_path, std::ios::binary);
                                     if (!file) {
                                         return false;
                                     }
                                     file.seekg(static_cast<std::streamoff>(offset));
                                     std::vector<char> buffer(to_read);
                                     file.read(buffer.data(), static_cast<std::streamsize>(to_read));
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
        if (!authenticate_user(req.get_header_value("Authorization"), user_id, req.remote_addr)) {
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

        const std::string uploader = metadata.config.is_object() && metadata.config.contains("uploaded_by")
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
                if (!t.empty()) {
                    filter.tags.push_back(t);
                }
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
        for (const auto &r : results) {
            arr.push_back(search_result_to_json(r));
        }
        res.set_content(nlohmann::json{{"data", arr}, {"total", arr.size()}}.dump(), "application/json");
    });

    server_->Get("/v1/search/suggest", [this](const httplib::Request &req, httplib::Response &res) {
        const std::string q = req.has_param("q") ? req.get_param_value("q") : "";
        const int limit = to_int_param(req, "limit", 10);
        nlohmann::json arr = nlohmann::json::array();
        for (const auto &s : search_.get_suggestions(q, limit)) {
            arr.push_back(s);
        }
        res.set_content(nlohmann::json{{"suggestions", arr}}.dump(), "application/json");
    });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<ModelMetadata> ModelHubServer::search_models(const std::string &query, const std::vector<std::string> &tags,
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
    if (!fs::exists(model_path, ec)) {
        return false;
    }
    const std::string dest =
        models_dir_ + "/" + metadata.id + "_" + sanitize_filename(fs::path(model_path).filename().string());
    fs::copy_file(model_path, dest, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        return false;
    }
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
    if (metadata.id.empty()) {
        return false;
    }

    std::string file_name = metadata.config.is_object() && metadata.config.contains("file_name")
                                ? metadata.config["file_name"].get<std::string>()
                                : "";
    const std::string src = models_dir_ + "/" + model_id + "_" + file_name;
    std::error_code ec;
    if (file_name.empty() || !fs::exists(src, ec)) {
        for (const auto &entry : fs::directory_iterator(models_dir_, ec)) {
            if (entry.path().filename().string().rfind(model_id + "_", 0) == 0) {
                fs::copy_file(entry.path(), output_path, fs::copy_options::overwrite_existing, ec);
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
    while (!id.empty() && id.front() == '-') {
        id.erase(id.begin());
    }
    while (!id.empty() && id.back() == '-') {
        id.pop_back();
    }
    if (id.empty()) {
        id = "model";
    }
    return id + "-" + random_hex(4);
}

std::string ModelHubServer::calculate_checksum(const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        return {};
    }
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
    if (!file) {
        return false;
    }
    char magic[8] = {};
    file.read(magic, 8);
    if (file.gcount() < 4) {
        return false;
    }
    // GGUF magic is "GGUF" at offset 0.
    if (std::memcmp(magic, "GGUF", 4) == 0) {
        return true;
    }
    if (file.gcount() != 8) {
        return false;
    }
    // Safetensors: little-endian uint64 header length, then a '{' JSON
    // header. Bound the claimed length to something sane.
    uint64_t header_len = 0;
    std::memcpy(&header_len, magic, sizeof(header_len));
    if (header_len == 0 || header_len >= (1ull << 32)) {
        return false;
    }
    char first_json = 0;
    file.read(&first_json, 1);
    return file.gcount() == 1 && first_json == '{';
}

void ModelHubServer::log_access(const std::string &user_id, const std::string &action, const std::string &resource)
{
    spdlog::info("hub access: user={} action={} resource={}", user_id, action, resource);
}

}  // namespace serius::hub
