#include <gtest/gtest.h>

#include <cmath>
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

TEST(HubSearch, EmbeddingManagerRealModel)
{
    // Real embedding path: token_embd matrix from a GGUF + BPE tokenizer.
    const std::string gguf = "../models/smollm2-q4/SmolLM2-135M-Instruct-Q4_0.gguf";
    if (!fs::exists(gguf)) {
        GTEST_SKIP() << "GGUF model not found: " << gguf;
    }

    serius::hub::EmbeddingManager em;
    ASSERT_TRUE(em.load_model(gguf));
    ASSERT_TRUE(em.is_model_loaded());

    const auto e1 = em.compute_embedding("hello world");
    const auto e2 = em.compute_embedding("quantum chromodynamics");
    const auto e1b = em.compute_embedding("hello world");

    ASSERT_FALSE(e1.empty());
    EXPECT_EQ(e1.size(), e2.size());

    // Deterministic
    ASSERT_EQ(e1.size(), e1b.size());
    for (size_t i = 0; i < e1.size(); ++i) {
        EXPECT_FLOAT_EQ(e1[i], e1b[i]);
    }

    // L2-normalized (mean-pooled then normalized)
    double norm = 0.0;
    for (float v : e1) norm += static_cast<double>(v) * v;
    EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-4);

    // Different texts -> different embeddings
    double diff = 0.0;
    for (size_t i = 0; i < e1.size(); ++i) {
        const double d = static_cast<double>(e1[i]) - e2[i];
        diff += d * d;
    }
    EXPECT_GT(diff, 1e-6);
}

TEST(HubSearch, EmbeddingManagerSafetensorsDir)
{
    // Real embedding path via model dir: model.safetensors (BF16
    // model.embed_tokens.weight) + tokenizer.json.
    const std::string dir = "../models/smollm2-135m";
    if (!fs::exists(dir + "/model.safetensors") || !fs::exists(dir + "/tokenizer.json")) {
        GTEST_SKIP() << "safetensors model dir not found: " << dir;
    }

    serius::hub::EmbeddingManager em;
    ASSERT_TRUE(em.load_model(dir));
    ASSERT_TRUE(em.is_model_loaded());

    const auto e = em.compute_embedding("hello world");
    ASSERT_FALSE(e.empty());
    // smollm2-135m hidden_size = 576
    EXPECT_EQ(e.size(), 576u);

    double norm = 0.0;
    for (float v : e) norm += static_cast<double>(v) * v;
    EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-4);
}

TEST(HubSearch, EmbeddingManagerSemanticProbe)
{
    // Diagnostic only: token_embd mean-pool (LLM fallback path) has weak
    // semantics — anisotropic, dominated by frequent tokens. The real
    // embedding path is the nomic-bert encoder (EmbeddingManagerNomicBert).
    // We only assert the strongest signal; the rest is printed for info.
    const std::string gguf = "../models/smollm2-q4/SmolLM2-135M-Instruct-Q4_0.gguf";
    if (!fs::exists(gguf)) {
        GTEST_SKIP() << "GGUF model not found: " << gguf;
    }
    serius::hub::EmbeddingManager em;
    ASSERT_TRUE(em.load_model(gguf));

    auto cos = [&](const std::string &a, const std::string &b) {
        const auto ea = em.compute_embedding(a);
        const auto eb = em.compute_embedding(b);
        double dot = 0.0;
        for (size_t i = 0; i < ea.size(); ++i) dot += static_cast<double>(ea[i]) * eb[i];
        return dot;  // both L2-normalized -> dot == cosine
    };

    const double cat_kitten = cos("cat", "kitten");
    const double cat_quantum = cos("cat", "quantum");
    const double dog_cat = cos("dog", "cat");
    const double dog_feline = cos("dog", "feline");

    std::cout << "cos(cat,kitten)=" << cat_kitten
              << " cos(cat,quantum)=" << cat_quantum
              << " cos(dog,cat)=" << dog_cat
              << " cos(dog,feline)=" << dog_feline << "\n";

    EXPECT_GT(cat_kitten, cat_quantum);
}

TEST(HubSearch, EmbeddingManagerNomicBert)
{
    // Real embedding model: nomic-embed-text-v1.5 GGUF (nomic-bert arch).
    // Path via env var since it's outside the repo.
    const char *env = std::getenv("SERIUS_TEST_NOMIC_GGUF");
    if (env == nullptr || !fs::exists(env)) {
        GTEST_SKIP() << "SERIUS_TEST_NOMIC_GGUF not set or file missing";
    }
    const std::string gguf = env;

    serius::hub::EmbeddingManager em;
    ASSERT_TRUE(em.load_model(gguf));
    ASSERT_TRUE(em.is_model_loaded());

    const auto e1 = em.compute_embedding("a cat sits on the mat");
    const auto e2 = em.compute_embedding("quantum chromodynamics");
    const auto e1b = em.compute_embedding("a cat sits on the mat");

    ASSERT_EQ(e1.size(), 768u);
    ASSERT_EQ(e1.size(), e1b.size());
    for (size_t i = 0; i < e1.size(); ++i) EXPECT_FLOAT_EQ(e1[i], e1b[i]);

    double norm = 0.0;
    for (float v : e1) norm += static_cast<double>(v) * v;
    EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-3);

    auto cos = [&](const std::string &a, const std::string &b) {
        const auto ea = em.compute_embedding(a);
        const auto eb = em.compute_embedding(b);
        double d = 0.0;
        for (size_t i = 0; i < ea.size(); ++i) d += static_cast<double>(ea[i]) * eb[i];
        return d;
    };
    const double cat_kitten = cos("a cat sits on the mat", "a kitten plays nearby");
    const double cat_quantum = cos("a cat sits on the mat", "quantum chromodynamics");
    const double dog_cat = cos("a dog runs in the park", "a cat sits on the mat");
    const double dog_feline = cos("a dog runs in the park", "a feline sleeps");
    const double py_java = cos("python is a programming language", "java is a programming language");
    const double py_pizza = cos("python is a programming language", "pizza is a food");
    std::cout << "nomic dim=" << e1.size()
              << " cat_kitten=" << cat_kitten
              << " cat_quantum=" << cat_quantum
              << " dog_cat=" << dog_cat
              << " dog_feline=" << dog_feline
              << " py_java=" << py_java
              << " py_pizza=" << py_pizza << "\n";
    EXPECT_GT(cat_kitten, cat_quantum);
    EXPECT_GT(py_java, py_pizza);
}
