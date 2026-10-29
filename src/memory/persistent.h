#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace memory {

/// Metadata for a single stored memory session.
struct SessionMeta
{
    std::string session_id;   // Unique session identifier
    std::string user_id;      // Owner of this session
    std::string timestamp;    // ISO-8601 creation timestamp
    size_t num_vectors;       // Number of summary vectors in this session
    size_t d_model;           // Dimension of each vector
    float information_ratio;  // Compression ratio from the original conversation
    std::string file_path;    // Path to the binary tensor data on disk
};

/// Persistent memory manager: stores and retrieves compressed memory sessions
/// per user on disk.  Each user gets a subdirectory under the configured
/// root, and a JSON manifest (`index.json`) tracks the individual session
/// files and their metadata.
///
/// Directory layout:
///   <root>/<user_id>/index.json
///   <root>/<user_id>/session_<timestamp>.bin
class PersistentMemory
{
   public:
    /// @param root_dir  Directory that will contain per-user subdirectories.
    /// @param d_model   Expected vector dimension (used for validation).
    explicit PersistentMemory(const std::string &root_dir, size_t d_model);

    /// Save a compressed memory tensor for a user.
    /// @param user_id  Owner identifier.
    /// @param state    (num_vectors, d_model) tensor to persist.
    /// @param info_ratio  Original compression ratio (for bookkeeping).
    /// @return The SessionMeta that was written, or a meta with empty
    ///         session_id on failure.
    SessionMeta save_session(const std::string &user_id,
                             const core::Tensor &state,
                             float info_ratio = 0.0f);

    /// Load a specific session tensor by id.
    core::Tensor load_session(const std::string &user_id,
                              const std::string &session_id) const;

    /// List all sessions for a user.
    std::vector<SessionMeta> list_sessions(const std::string &user_id) const;

    /// Delete a specific session.
    bool delete_session(const std::string &user_id,
                        const std::string &session_id);

    /// Build a combined context tensor from all (or the N most recent)
    /// sessions for a user.
    /// @param max_sessions  0 = all sessions; otherwise limit to the N newest.
    core::Tensor build_user_context(const std::string &user_id,
                                    size_t max_sessions = 0) const;

    /// Total number of stored sessions for a given user.
    size_t session_count(const std::string &user_id) const;

    const std::string &root_dir() const { return root_dir_; }
    size_t d_model() const { return d_model_; }

   private:
    std::string root_dir_;
    size_t d_model_;

    /// Return the path to a user's directory (creates it if needed).
    std::string user_dir(const std::string &user_id, bool create = false) const;

    /// Return the path to the index manifest for a user.
    std::string index_path(const std::string &user_id) const;

    /// Read the JSON index for a user.
    std::vector<SessionMeta> read_index(const std::string &user_id) const;

    /// Write the JSON index for a user.
    bool write_index(const std::string &user_id,
                     const std::vector<SessionMeta> &sessions) const;

    /// Generate a timestamp string.
    static std::string make_timestamp();

    /// Generate a session id (timestamp-based).
    static std::string make_session_id();

    /// Save a tensor in the simple binary format (rows, cols, raw floats).
    static bool save_tensor(const std::string &path, const core::Tensor &t);

    /// Load a tensor from the simple binary format.
    static core::Tensor load_tensor(const std::string &path, size_t expected_d);
};

}  // namespace memory
}  // namespace serius
