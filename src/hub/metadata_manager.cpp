#include "metadata_manager.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
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
    std::ifstream file(db_path_);
    nlohmann::json db;
    if (file.is_open()) {
        try {
            file >> db;
        }
        catch (...) {
            db = nlohmann::json::object();
        }
        file.close();
    }

    db[metadata.id] = model_to_json(metadata);

    std::ofstream out(db_path_);
    out << db.dump(2);
    return true;
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
    if (offset + limit > results.size()) {
        results.erase(results.begin(), results.begin() + offset);
    } else {
        results.erase(results.begin(), results.begin() + offset);
        results.erase(results.begin() + limit, results.end());
    }

    return results;
}

}  // namespace serius::hub
