#include "server/server.h"

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

#include "server/metrics.h"
#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"
#include "transformer/model.h"

namespace {

int find_free_port()
{
    httplib::Server port_probe;
    int port = port_probe.bind_to_any_port("127.0.0.1");
    bool probe_result = false;
    std::thread port_probe_thread([&] { probe_result = port_probe.listen_after_bind(); });
    port_probe.wait_until_ready();
    port_probe.stop();
    port_probe_thread.join();
    if (!probe_result) return 0;
    return port;
}

void wait_for_server(httplib::Client &client, int timeout_seconds = 5)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        auto res = client.Get("/health");
        if (res && res->status == 200) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
}

TEST(ServerTest, StopsFromAnotherThread)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::server::Server server(static_cast<uint16_t>(port));
    int run_result = -1;
    std::thread server_thread([&] { run_result = server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    auto response = client.Get("/health");
    server.stop();
    server_thread.join();

    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 200);
    EXPECT_EQ(run_result, 0);
}

TEST(ServerTest, HealthAndPrometheusMetrics)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::server::Server server(static_cast<uint16_t>(port));
    std::thread server_thread([&] { server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    auto health_res = client.Get("/health");
    ASSERT_TRUE(health_res);
    EXPECT_EQ(health_res->status, 200);
    auto health_json = nlohmann::json::parse(health_res->body);
    EXPECT_EQ(health_json["status"], "ok");
    EXPECT_FALSE(health_json["model_loaded"].get<bool>());

    auto metrics_res = client.Get("/metrics");
    ASSERT_TRUE(metrics_res);
    EXPECT_EQ(metrics_res->status, 200);
    EXPECT_NE(metrics_res->body.find("llm_requests_total"), std::string::npos);

    server.stop();
    server_thread.join();
}

TEST(ServerTest, ModelsEndpoint)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::server::Server server(static_cast<uint16_t>(port));
    server.set_model_name("serius-test-model");
    std::thread server_thread([&] { server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    auto models_res = client.Get("/v1/models");
    ASSERT_TRUE(models_res);
    EXPECT_EQ(models_res->status, 200);
    auto models_json = nlohmann::json::parse(models_res->body);
    EXPECT_EQ(models_json["object"], "list");
    ASSERT_FALSE(models_json["data"].empty());
    EXPECT_EQ(models_json["data"][0]["id"], "serius-test-model");

    auto single_model_res = client.Get("/v1/models/serius-test-model");
    ASSERT_TRUE(single_model_res);
    EXPECT_EQ(single_model_res->status, 200);
    auto single_json = nlohmann::json::parse(single_model_res->body);
    EXPECT_EQ(single_json["id"], "serius-test-model");

    server.stop();
    server_thread.join();
}

TEST(ServerTest, LiveLLaMAChatCompletion)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::model::LLamaConfig config;
    config.vocab_size = 100;
    config.hidden_size = 64;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;
    config.intermediate_size = 128;
    config.num_hidden_layers = 1;
    config.max_position_embeddings = 64;

    auto model = std::make_unique<serius::model::LLaMAModel>(config);
    auto tokenizer = std::make_unique<serius::BPETokenizer>();

    serius::server::Server server(static_cast<uint16_t>(port));
    server.set_model_name("llama-mini");
    server.set_llama_model(std::move(model), std::move(tokenizer));
    EXPECT_TRUE(server.has_model());

    std::thread server_thread([&] { server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    nlohmann::json req_body = {
        {"model", "llama-mini"},
        {"messages", {{{"role", "user"}, {"content", "Hello"}}}} ,
        {"max_tokens", 8},
        {"temperature", 0.0f}
    };

    auto res = client.Post("/v1/chat/completions", req_body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto res_json = nlohmann::json::parse(res->body);
    EXPECT_EQ(res_json["object"], "chat.completion");
    EXPECT_EQ(res_json["model"], "llama-mini");
    ASSERT_FALSE(res_json["choices"].empty());
    EXPECT_EQ(res_json["choices"][0]["message"]["role"], "assistant");
    EXPECT_GT(res_json["usage"]["total_tokens"].get<int>(), 0);

    server.stop();
    server_thread.join();
}

TEST(ServerTest, LiveLLaMAStreaming)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::model::LLamaConfig config;
    config.vocab_size = 100;
    config.hidden_size = 64;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;
    config.intermediate_size = 128;
    config.num_hidden_layers = 1;
    config.max_position_embeddings = 64;

    auto model = std::make_unique<serius::model::LLaMAModel>(config);
    auto tokenizer = std::make_unique<serius::BPETokenizer>();

    serius::server::Server server(static_cast<uint16_t>(port));
    server.set_model_name("llama-stream");
    server.set_llama_model(std::move(model), std::move(tokenizer));

    std::thread server_thread([&] { server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    nlohmann::json req_body = {
        {"model", "llama-stream"},
        {"messages", {{{"role", "user"}, {"content", "Count to 5"}}}} ,
        {"max_tokens", 5},
        {"stream", true}
    };

    std::string stream_content;
    auto res = client.Post("/v1/chat/completions", req_body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_NE(res->body.find("data:"), std::string::npos);
    EXPECT_NE(res->body.find("[DONE]"), std::string::npos);

    server.stop();
    server_thread.join();
}

TEST(ServerTest, LiveLLaMAEmbeddings)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::model::LLamaConfig config;
    config.vocab_size = 100;
    config.hidden_size = 64;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;
    config.intermediate_size = 128;
    config.num_hidden_layers = 1;
    config.max_position_embeddings = 64;

    auto model = std::make_unique<serius::model::LLaMAModel>(config);
    auto tokenizer = std::make_unique<serius::BPETokenizer>();

    serius::server::Server server(static_cast<uint16_t>(port));
    server.set_model_name("llama-embed");
    server.set_llama_model(std::move(model), std::move(tokenizer));

    std::thread server_thread([&] { server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    nlohmann::json req_body = {
        {"model", "llama-embed"},
        {"input", "Test embeddings input"}
    };

    auto res = client.Post("/v1/embeddings", req_body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    auto res_json = nlohmann::json::parse(res->body);
    EXPECT_EQ(res_json["object"], "list");
    ASSERT_FALSE(res_json["data"].empty());
    auto embedding = res_json["data"][0]["embedding"];
    EXPECT_EQ(embedding.size(), 64u);

    server.stop();
    server_thread.join();
}

TEST(ServerTest, MultiModelRegistry)
{
    int port = find_free_port();
    ASSERT_GT(port, 0);

    auto make_tiny_llama = [] {
        serius::model::LLamaConfig config;
        config.vocab_size = 100;
        config.hidden_size = 64;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.intermediate_size = 128;
        config.num_hidden_layers = 1;
        config.max_position_embeddings = 64;
        return std::make_unique<serius::model::LLaMAModel>(config);
    };

    serius::server::Server server(static_cast<uint16_t>(port));
    server.set_model_name("alpha");
    server.set_llama_model(make_tiny_llama(), std::make_unique<serius::BPETokenizer>());
    server.set_model_name("beta");
    server.set_llama_model(make_tiny_llama(), std::make_unique<serius::BPETokenizer>());

    std::thread server_thread([&] { server.run(); });

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client);

    // /v1/models lists both registered models.
    auto models_res = client.Get("/v1/models");
    ASSERT_TRUE(models_res);
    ASSERT_EQ(models_res->status, 200);
    auto models_json = nlohmann::json::parse(models_res->body);
    std::vector<std::string> ids;
    for (const auto &m : models_json["data"]) {
        ids.push_back(m["id"].get<std::string>());
    }
    EXPECT_NE(std::find(ids.begin(), ids.end(), "alpha"), ids.end());
    EXPECT_NE(std::find(ids.begin(), ids.end(), "beta"), ids.end());

    auto chat = [&](const std::string &model) {
        nlohmann::json body = {
            {"model", model},
            {"messages", {{{"role", "user"}, {"content", "Hi"}}}} ,
            {"max_tokens", 4},
            {"temperature", 0.0f}
        };
        return client.Post("/v1/chat/completions", body.dump(), "application/json");
    };

    // Both models answer; unknown name -> 404.
    auto res_a = chat("alpha");
    ASSERT_TRUE(res_a);
    EXPECT_EQ(res_a->status, 200);
    EXPECT_EQ(nlohmann::json::parse(res_a->body)["model"], "alpha");

    auto res_b = chat("beta");
    ASSERT_TRUE(res_b);
    EXPECT_EQ(res_b->status, 200);

    auto res_bad = chat("nope");
    ASSERT_TRUE(res_bad);
    EXPECT_EQ(res_bad->status, 404);

    // Unload alpha: chat on it now 404s, beta still works.
    auto del = client.Delete("/v1/models/alpha");
    ASSERT_TRUE(del);
    EXPECT_EQ(del->status, 200);

    auto res_a2 = chat("alpha");
    ASSERT_TRUE(res_a2);
    EXPECT_EQ(res_a2->status, 404);

    auto res_b2 = chat("beta");
    ASSERT_TRUE(res_b2);
    EXPECT_EQ(res_b2->status, 200);

    // Unloading a missing model -> 404.
    auto del2 = client.Delete("/v1/models/alpha");
    ASSERT_TRUE(del2);
    EXPECT_EQ(del2->status, 404);

    server.stop();
    server_thread.join();
}

}  // namespace
