#include "metadata_manager.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>


namespace serius::hub {

MetadataManager::MetadataManager(const std::string &db_path) : db_path_(db_path) {}

MetadataManager::~MetadataManager()
{
    // if (db_) {
    //     sqlite3_close(db_);
    // }
}

bool MetadataManager::initialize()
{
    // Simple file-based storage for now
    // In production, this would use SQLite
    return true;
}

bool MetadataManager::save_model_metadata(const ModelMetadata &metadata)
{
    // Simple JSON file storage for now
    auto stored = metadata;
    if (stored.created_at.empty()) stored.created_at = now_iso8601();
    stored.updated_at = now_iso8601();

    auto db = load_db();
    db[stored.id] = model_to_json(stored);
    return save_db(db);
}

std::vector<ModelMetadata> MetadataManager::search_models(const std::string &query,
                                                          const std::vector<std::string> &tags,
                                                          const std::string &author, int limit, int offset)
{
    std::vector<ModelMetadata> results;

    std::ifstream file(db_path_);
    if (!file.is_open()) return results;

    nlohmann::json db;
    try {
        file >> db;
    }
    catch (...) {
        return results;
    }

    for (auto &[id, model_json] : db.items()) {
        ModelMetadata metadata = json_to_model(model_json);

        // Apply filters
        if (!query.empty()) {
            std::string search_text = metadata.name + " " + metadata.description;
            std::transform(search_text.begin(), search_text.end(), search_text.begin(), ::tolower);
            std::string query_lower = query;
            std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), ::tolower);

            if (search_text.find(query_lower) == std::string::npos) {
                continue;
            }
        }

        if (!author.empty() && metadata.author != author) {
            continue;
        }

        if (!tags.empty()) {
            bool has_tag = false;
            for (const auto &tag : tags) {
                if (std::find(metadata.tags.begin(), metadata.tags.end(), tag) != metadata.tags.end()) {
                    has_tag = true;
                    break;
                }
            }
            if (!has_tag) continue;
        }

        results.push_back(metadata);
    }

    // Sort by download count
    std::sort(results.begin(), results.end(),
              [](const ModelMetadata &a, const ModelMetadata &b) { return a.download_count > b.download_count; });

    // Apply pagination
    if (offset >= results.size()) return {};
    const size_t begin = static_cast<size_t>(offset);
    const size_t end = std::min(results.size(), begin + static_cast<size_t>(limit));
    results.erase(results.begin() + end, results.end());
    results.erase(results.begin(), results.begin() + begin);

    return results;
}

ModelMetadata MetadataManager::get_model_metadata(const std::string &model_id)
{
    const auto db = load_db();
    if (db.is_object() && db.contains(model_id)) {
        return json_to_model(db[model_id]);
    }
    return ModelMetadata{};
}

bool MetadataManager::update_model_metadata(const std::string &model_id, const ModelMetadata &metadata)
{
    auto db = load_db();
    if (!db.is_object()) db = nlohmann::json::object();
    if (!db.contains(model_id)) return false;
    auto updated = metadata;
    updated.id = model_id;
    updated.updated_at = now_iso8601();
    db[model_id] = model_to_json(updated);
    return save_db(db);
}

bool MetadataManager::delete_model_metadata(const std::string &model_id)
{
    auto db = load_db();
    if (!db.is_object() || !db.contains(model_id)) return false;
    db.erase(model_id);
    return save_db(db);
}

int MetadataManager::get_model_count()
{
    const auto db = load_db();
    return db.is_object() ? static_cast<int>(db.size()) : 0;
}

int MetadataManager::get_download_count(const std::string &model_id)
{
    return get_model_metadata(model_id).download_count;
}

bool MetadataManager::increment_download_count(const std::string &model_id)
{
    auto db = load_db();
    if (!db.is_object() || !db.contains(model_id)) return false;
    auto m = json_to_model(db[model_id]);
    m.download_count += 1;
    db[model_id] = model_to_json(m);
    return save_db(db);
}

bool MetadataManager::update_rating(const std::string &model_id, float rating)
{
    auto db = load_db();
    if (!db.is_object() || !db.contains(model_id)) return false;
    auto m = json_to_model(db[model_id]);
    m.rating = rating;
    db[model_id] = model_to_json(m);
    return save_db(db);
}

nlohmann::json MetadataManager::load_db() const
{
    std::ifstream file(db_path_);
    if (!file.is_open()) return nlohmann::json::object();
    try {
        nlohmann::json j;
        file >> j;
        return j;
    } catch (const std::exception &e) {
        std::cerr << "[Hub] Corrupt metadata DB " << db_path_ << ": " << e.what() << "\n";
        return nlohmann::json::object();
    }
}

bool MetadataManager::save_db(const nlohmann::json &db)
{
    std::ofstream out(db_path_);
    if (!out.is_open()) return false;
    out << db.dump(2);
    return true;
}

}  // namespace serius::hub
