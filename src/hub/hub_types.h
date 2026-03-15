#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace serius::hub {

struct ModelMetadata
{
    std::string id;
    std::string name;
    std::string description;
    std::string author;
    std::string version;
    std::string license;
    std::vector<std::string> tags;
    std::string model_type;  // "gguf", "safetensors", etc.
    size_t file_size;
    std::string checksum_sha256;
    nlohmann::json config;
    std::string created_at;
    std::string updated_at;
    // SearchResult row_to_search_result(sqlite3_stmt *stmt, float score = 0.0f);
    int download_count;
    float rating;
    bool is_public;
};

inline nlohmann::json model_to_json(const ModelMetadata &m)
{
    return nlohmann::json{{"id", m.id},
                          {"name", m.name},
                          {"description", m.description},
                          {"author", m.author},
                          {"version", m.version},
                          {"license", m.license},
                          {"tags", m.tags},
                          {"model_type", m.model_type},
                          {"file_size", m.file_size},
                          {"checksum_sha256", m.checksum_sha256},
                          {"config", m.config},
                          {"created_at", m.created_at},
                          {"updated_at", m.updated_at},
                          {"download_count", m.download_count},
                          {"rating", m.rating},
                          {"is_public", m.is_public}};
}

inline  // ModelMetadata row_to_model(sqlite3_stmt *stmt);
    ModelMetadata json_to_model(const nlohmann::json &j)
{
    ModelMetadata m;
    m.id = j.value("id", "");
    m.name = j.value("name", "");
    m.description = j.value("description", "");
    m.author = j.value("author", "");
    m.version = j.value("version", "");
    m.license = j.value("license", "");
    m.tags = j.value("tags", std::vector<std::string>{});
    m.model_type = j.value("model_type", "");
    m.file_size = j.value("file_size", (size_t)0);
    m.checksum_sha256 = j.value("checksum_sha256", "");
    m.config = j.value("config", nlohmann::json::object());
    m.created_at = j.value("created_at", "");
    m.updated_at = j.value("updated_at", "");
    m.download_count = j.value("download_count", 0);
    m.rating = j.value("rating", 0.0f);
    m.is_public = j.value("is_public", true);
    return m;
}

}  // namespace serius::hub
