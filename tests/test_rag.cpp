#include <gtest/gtest.h>

#include "rag/embeddings.h"
#include "rag/vector_client.h"
#include "rag/retriever.h"

using namespace serius;
using namespace serius::rag;

TEST(Embeddings, NormalizeEmbedding)
{
    std::vector<float> embedding = {1.0f, 2.0f, 3.0f};
    normalize_embedding(embedding);

    float norm = l2_norm(embedding);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(Embeddings, L2Norm)
{
    std::vector<float> vec = {3.0f, 4.0f};
    float norm = l2_norm(vec);
    EXPECT_NEAR(norm, 5.0f, 1e-5f);
}

TEST(Embeddings, CosineSimilarity)
{
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    std::vector<float> c = {1.0f, 0.0f};

    float sim_ab = cosine_similarity(a, b);
    float sim_ac = cosine_similarity(a, c);

    EXPECT_NEAR(sim_ab, 0.0f, 1e-5f);
    EXPECT_NEAR(sim_ac, 1.0f, 1e-5f);
}

TEST(Embeddings, ComputeEmbedding)
{
    std::string text = "test text";
    std::vector<float> embedding = compute_embedding(text, 128);

    EXPECT_EQ(embedding.size(), 128);

    // Should be normalized
    float norm = l2_norm(embedding);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(Embeddings, SimilarTexts)
{
    std::string text1 = "hello world";
    std::string text2 = "hello world";
    std::string text3 = "completely different";

    auto emb1 = compute_embedding(text1, 128);
    auto emb2 = compute_embedding(text2, 128);
    auto emb3 = compute_embedding(text3, 128);

    float sim_12 = cosine_similarity(emb1, emb2);
    float sim_13 = cosine_similarity(emb1, emb3);

    // Same text should have high similarity
    EXPECT_GT(sim_12, 0.9f);

    // Different text should have lower similarity
    EXPECT_LT(sim_13, sim_12);
}

TEST(VectorDBClient, UpsertAndSearch)
{
    VectorDBClient::Config cfg;
    cfg.use_local = true;
    cfg.dimension = 128;
    VectorDBClient client(cfg);

    std::vector<float> emb1(128, 1.0f);
    std::vector<float> emb2(128, 2.0f);

    client.upsert("doc1", emb1, "Document 1 text");
    client.upsert("doc2", emb2, "Document 2 text");

    EXPECT_EQ(client.size(), 2);

    std::vector<float> query(128, 1.0f);
    auto results = client.search(query, 2, 0.5f);

    EXPECT_GT(results.size(), 0);
    EXPECT_EQ(results[0].id, "doc1");
}

TEST(VectorDBClient, GetDocument)
{
    VectorDBClient::Config cfg;
    cfg.use_local = true;
    cfg.dimension = 64;
    VectorDBClient client(cfg);

    std::vector<float> emb(64, 1.0f);
    client.upsert("doc1", emb, "Test document");

    auto doc = client.get("doc1");
    EXPECT_EQ(doc.id, "doc1");
    EXPECT_EQ(doc.text, "Test document");
}

TEST(VectorDBClient, RemoveDocument)
{
    VectorDBClient::Config cfg;
    cfg.use_local = true;
    cfg.dimension = 64;
    VectorDBClient client(cfg);

    std::vector<float> emb(64, 1.0f);
    client.upsert("doc1", emb, "Test document");

    EXPECT_EQ(client.size(), 1);

    client.remove("doc1");
    EXPECT_EQ(client.size(), 0);

    auto doc = client.get("doc1");
    EXPECT_EQ(doc.id, "");
}

TEST(VectorDBClient, Clear)
{
    VectorDBClient::Config cfg;
    cfg.use_local = true;
    cfg.dimension = 64;
    VectorDBClient client(cfg);

    std::vector<float> emb(64, 1.0f);
    client.upsert("doc1", emb, "Doc 1");
    client.upsert("doc2", emb, "Doc 2");

    EXPECT_EQ(client.size(), 2);

    client.clear();
    EXPECT_EQ(client.size(), 0);
}

TEST(RAGRetriever, AddDocument)
{
    RAGRetriever::Config cfg;
    cfg.embedding_dim = 128;
    RAGRetriever retriever(cfg);

    retriever.add_document("doc1", "This is a test document");

    EXPECT_EQ(retriever.get_vector_db().size(), 1);
}

TEST(RAGRetriever, RetrieveAndBuildPrompt)
{
    RAGRetriever::Config cfg;
    cfg.embedding_dim = 128;
    cfg.top_k = 2;
    cfg.similarity_threshold = 0.0f;  // Include all
    RAGRetriever retriever(cfg);

    retriever.add_document("doc1", "Paris is the capital of France");
    retriever.add_document("doc2", "London is the capital of UK");

    std::string query = "What is the capital of France?";
    std::string prompt = retriever.retrieve(query);

    EXPECT_TRUE(prompt.find("[SYSTEM]") != std::string::npos);
    EXPECT_TRUE(prompt.find("[DOCUMENTI]") != std::string::npos);
    EXPECT_TRUE(prompt.find("[DOMANDA]") != std::string::npos);
    EXPECT_TRUE(prompt.find(query) != std::string::npos);
}

TEST(RAGRetriever, Clear)
{
    RAGRetriever::Config cfg;
    cfg.embedding_dim = 128;
    RAGRetriever retriever(cfg);

    retriever.add_document("doc1", "Test document");
    EXPECT_EQ(retriever.get_vector_db().size(), 1);

    retriever.clear();
    EXPECT_EQ(retriever.get_vector_db().size(), 0);
}
