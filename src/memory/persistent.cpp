#include "persistent.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <nlohmann/json.hpp>

namespace serius {
namespace memory {

namespace fs = std::filesystem;
using json = nlohmann::json;

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

std::string PersistentMemory::make_timestamp()
{
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
#if defined(_MSC_VER)
    struct tm buf;
    gmtime_s(&buf, &time);
    oss << std::put_time(&buf, "%Y%m%dT%H%M%SZ");
#else
    oss << std::put_time(std::gmtime(&time), "%Y%m%dT%H%M%SZ");
#endif
    return oss.str();
}

std::string PersistentMemory::make_session_id()
{
    // Use a combination of timestamp and a short random suffix to avoid
    // collisions when two sessions are created in the same second.
    auto ts = make_timestamp();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    return "session_" + ts + "_" + std::to_string(ms % 100000);
}

std::string PersistentMemory::user_dir(const std::string &user_id, bool create) const
{
    std::string dir = root_dir_ + "/" + user_id;
    if (create) {
        std::error_code ec;
        fs::create_directories(dir, ec);
    }
    return dir;
}

std::string PersistentMemory::index_path(const std::string &user_id) const
{
    return user_dir(user_id) + "/index.json";
}

// ──────────────────────────────────────────────────────────────────────────
// Index I/O
// ──────────────────────────────────────────────────────────────────────────

std::vector<SessionMeta> PersistentMemory::read_index(const std::string &user_id) const
{
    std::vector<SessionMeta> sessions;
    std::string path = index_path(user_id);
    std::ifstream in(path);
    if (!in.is_open()) return sessions;

    try {
        json j;
        in >> j;
        for (const auto &entry : j["sessions"]) {
            SessionMeta m;
            m.session_id       = entry.value("session_id", "");
            m.user_id          = entry.value("user_id", user_id);
            m.timestamp        = entry.value("timestamp", "");
            m.num_vectors      = entry.value("num_vectors", static_cast<size_t>(0));
            m.d_model          = entry.value("d_model", static_cast<size_t>(0));
            m.information_ratio = entry.value("information_ratio", 0.0f);
            m.file_path        = entry.value("file_path", "");
            sessions.push_back(std::move(m));
        }
    } catch (...) {
        // Corrupt index — treat as empty.
    }
    return sessions;
}

bool PersistentMemory::write_index(const std::string &user_id,
                                   const std::vector<SessionMeta> &sessions) const
{
    std::string dir = user_dir(user_id);
    std::error_code ec;
    fs::create_directories(dir, ec);

    json j;
    j["user_id"] = user_id;
    json arr = json::array();
    for (const auto &m : sessions) {
        json entry;
        entry["session_id"]        = m.session_id;
        entry["user_id"]           = m.user_id;
        entry["timestamp"]         = m.timestamp;
        entry["num_vectors"]       = m.num_vectors;
        entry["d_model"]           = m.d_model;
        entry["information_ratio"] = m.information_ratio;
        entry["file_path"]         = m.file_path;
        arr.push_back(std::move(entry));
    }
    j["sessions"] = std::move(arr);

    std::string path = index_path(user_id);
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << j.dump(2);
    return true;
}

// ──────────────────────────────────────────────────────────────────────────
// Tensor binary I/O (rows, cols, raw floats)
// ──────────────────────────────────────────────────────────────────────────

bool PersistentMemory::save_tensor(const std::string &path, const core::Tensor &t)
{
    if (t.ndim() != 2) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    uint32_t rows = static_cast<uint32_t>(t.shape()[0]);
    uint32_t cols = static_cast<uint32_t>(t.shape()[1]);
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char *>(t.data()),
              static_cast<std::streamsize>(rows) * cols * sizeof(float));
    return out.good();
}

core::Tensor PersistentMemory::load_tensor(const std::string &path, size_t expected_d)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return core::Tensor({0, expected_d});
    }

    uint32_t rows = 0, cols = 0;
    in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    if (!in || rows == 0 || cols == 0) {
        return core::Tensor({0, expected_d});
    }

    core::Tensor t({static_cast<size_t>(rows), static_cast<size_t>(cols)});
    in.read(reinterpret_cast<char *>(t.data()),
            static_cast<std::streamsize>(rows) * cols * sizeof(float));
    return t;
}

// ──────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────

PersistentMemory::PersistentMemory(const std::string &root_dir, size_t d_model)
    : root_dir_(root_dir), d_model_(d_model)
{
    std::error_code ec;
    fs::create_directories(root_dir_, ec);
}

SessionMeta PersistentMemory::save_session(const std::string &user_id,
                                           const core::Tensor &state,
                                           float info_ratio)
{
    SessionMeta meta;
    if (state.ndim() != 2 || state.shape()[1] != d_model_) {
        return meta;  // empty session_id signals failure
    }

    meta.session_id       = make_session_id();
    meta.user_id          = user_id;
    meta.timestamp        = make_timestamp();
    meta.num_vectors      = state.shape()[0];
    meta.d_model          = d_model_;
    meta.information_ratio = info_ratio;

    std::string dir = user_dir(user_id, /*create=*/true);
    meta.file_path  = dir + "/" + meta.session_id + ".bin";

    if (!save_tensor(meta.file_path, state)) {
        meta.session_id.clear();
        return meta;
    }

    // Append to index.
    auto sessions = read_index(user_id);
    sessions.push_back(meta);
    write_index(user_id, sessions);

    return meta;
}

core::Tensor PersistentMemory::load_session(const std::string &user_id,
                                            const std::string &session_id) const
{
    auto sessions = read_index(user_id);
    for (const auto &m : sessions) {
        if (m.session_id == session_id) {
            return load_tensor(m.file_path, d_model_);
        }
    }
    return core::Tensor({0, d_model_});
}

std::vector<SessionMeta> PersistentMemory::list_sessions(const std::string &user_id) const
{
    return read_index(user_id);
}

bool PersistentMemory::delete_session(const std::string &user_id,
                                      const std::string &session_id)
{
    auto sessions = read_index(user_id);
    auto it = std::find_if(sessions.begin(), sessions.end(),
                           [&](const SessionMeta &m) { return m.session_id == session_id; });
    if (it == sessions.end()) return false;

    // Remove binary file.
    std::error_code ec;
    fs::remove(it->file_path, ec);

    sessions.erase(it);
    write_index(user_id, sessions);
    return true;
}

core::Tensor PersistentMemory::build_user_context(const std::string &user_id,
                                                  size_t max_sessions) const
{
    auto sessions = read_index(user_id);
    if (sessions.empty()) {
        return core::Tensor({0, d_model_});
    }

    // Sort by timestamp (lexicographic is fine for our format).
    std::sort(sessions.begin(), sessions.end(),
              [](const SessionMeta &a, const SessionMeta &b) {
                  return a.timestamp < b.timestamp;
              });

    if (max_sessions > 0 && sessions.size() > max_sessions) {
        sessions.erase(sessions.begin(),
                       sessions.begin() + static_cast<std::ptrdiff_t>(sessions.size() - max_sessions));
    }

    // Collect tensor paths.
    std::vector<core::Tensor> parts;
    size_t total_rows = 0;
    for (const auto &m : sessions) {
        auto t = load_tensor(m.file_path, d_model_);
        if (t.size() > 0) {
            total_rows += t.shape()[0];
            parts.push_back(std::move(t));
        }
    }

    if (parts.empty()) {
        return core::Tensor({0, d_model_});
    }

    // Concatenate along dim-0.
    core::Tensor context({total_rows, d_model_});
    size_t offset = 0;
    for (const auto &t : parts) {
        std::memcpy(context.data() + offset * d_model_,
                    t.data(),
                    t.shape()[0] * d_model_ * sizeof(float));
        offset += t.shape()[0];
    }
    return context;
}

size_t PersistentMemory::session_count(const std::string &user_id) const
{
    return read_index(user_id).size();
}

}  // namespace memory
}  // namespace serius
