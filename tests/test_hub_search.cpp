#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "hub/hub_types.h"
#include "hub/metadata_manager.h"
#include "hub/search_engine.h"

namespace fs = std::filesystem;

namespace {

std::string make_temp_db(const std::string &name)
{
    const auto dir = fs::temp_directory_path() / "serius_hub_test";
    fs::create_directories(dir);
    return (dir / name).string();
}

serius::hub::ModelMetadata make_meta(const std::string &id, const std::string &name,
                                     const std::string &desc, const std::string &author,
                                     int downloads, float rating)
{
    serius::hub::ModelMetadata m;
    m.id = id;
    m.name = name;
    m.description = desc;
    m.author = author;
    m.license = "apache-2.0";
    m.model_type = "gguf";
    m.tags = {"llm", "instruct"};
    m.download_count = downloads;
    m.rating = rating;
    m.is_public = true;
    return m;
}

}  // namespace

TEST(HubSearch, MetadataManagerRoundtrip)
{
    const std::string db = make_temp_db("meta_roundtrip.json");
    fs::remove(db);

    serius::hub::MetadataManager mgr(db);
    ASSERT_TRUE(mgr.initialize());

    auto m = make_meta("org/model-a", "Model A", "A chat model", "alice", 10, 4.5f);
    ASSERT_TRUE(mgr.save_model_metadata(m));

    auto got = mgr.get_model_metadata("org/model-a");
    EXPECT_EQ(got.name, "Model A");
    EXPECT_EQ(got.author, "alice");
    EXPECT_FALSE(got.created_at.empty());
    EXPECT_FALSE(got.updated_at.empty());

    ASSERT_TRUE(mgr.increment_download_count("org/model-a"));
    EXPECT_EQ(mgr.get_download_count("org/model-a"), 11);

    ASSERT_TRUE(mgr.update_rating("org/model-a", 3.0f));
    EXPECT_FLOAT_EQ(mgr.get_model_metadata("org/model-a").rating, 3.0f);

    EXPECT_EQ(mgr.get_model_count(), 1);
    ASSERT_TRUE(mgr.delete_model_metadata("org/model-a"));
    EXPECT_EQ(mgr.get_model_count(), 0);
}

TEST(HubSearch, SearchModelsFilters)
{
    const std::string db = make_temp_db("meta_search.json");
    fs::remove(db);

    serius::hub::MetadataManager mgr(db);
    mgr.save_model_metadata(make_meta("a/one", "Qwen Chat", "Small instruct model", "alice", 100, 4.0f));
    mgr.save_model_metadata(make_meta("b/two", "Code Llama", "Code completion model", "bob", 50, 3.5f));

    serius::hub::SearchEngine engine(db);
    engine.initialize();

    // Keyword search
    auto res = engine.search_models({.query = "code"});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].model_id, "b/two");

    // Author filter
    res = engine.search_models({.author = "alice"});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].model_id, "a/one");

    // Min downloads filter
    res = engine.search_models({.min_downloads = 75});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].model_id, "a/one");

    // Pagination
    res = engine.search_models({}, 1, 1);
    ASSERT_EQ(res.size(), 1u);
}

TEST(HubSearch, SemanticSearchRanksRelevant)
{
    const std::string db = make_temp_db("meta_semantic.json");
    fs::remove(db);

    serius::hub::MetadataManager mgr(db);
    mgr.save_model_metadata(make_meta("a/chat", "Chat Assistant", "assistant for chat and dialogue with users", "alice", 10, 4.0f));
    mgr.save_model_metadata(make_meta("b/vision", "Vision Model", "image classification and captioning for photos", "bob", 10, 4.0f));

    serius::hub::SearchEngine engine(db);
    engine.initialize();

    auto res = engine.semantic_search("chat assistant for dialogue", {}, 10, 0);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res[0].model_id, "a/chat");
}

TEST(HubSearch, SuggestionsAndAnalytics)
{
    const std::string db = make_temp_db("meta_suggest.json");
    fs::remove(db);
    fs::remove(db + ".analytics.json");

    serius::hub::MetadataManager mgr(db);
    mgr.save_model_metadata(make_meta("a/qwen", "Qwen2.5", "desc", "alice", 1, 4.0f));
    mgr.save_model_metadata(make_meta("b/llama", "Llama3", "desc", "bob", 1, 4.0f));

    serius::hub::SearchEngine engine(db);
    engine.initialize();

    auto sug = engine.get_suggestions("qwe", 10);
    ASSERT_EQ(sug.size(), 1u);
    EXPECT_EQ(sug[0], "Qwen2.5");

    engine.log_search("qwen", "user1", 3);
    engine.log_search("qwen", "user2", 3);
    engine.log_search("llama", "user1", 1);

    auto popular = engine.get_popular_queries(10);
    ASSERT_EQ(popular.size(), 2u);
    EXPECT_EQ(popular[0], "qwen");
    EXPECT_EQ(popular[1], "llama");
}
