#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "hub/model_hub_server.h"

using namespace serius::hub;

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
    return probe_result ? port : 0;
}

struct ServerGuard
{
    // Constructed inside the ctor body, AFTER the chdir into workdir:
    // ModelHubServer captures users_path_ = absolute("hub_users.json") at
    // construction time, so building it before the chdir would leak state
    // into (and read stale state from) the test's working directory.
    std::unique_ptr<ModelHubServer> server;
    std::thread thread;
    std::filesystem::path old_cwd;
    std::filesystem::path workdir;
    int port;

    explicit ServerGuard() : port(0)
    {
        workdir = std::filesystem::temp_directory_path() / ("serius_hub_server_test_" + std::to_string(find_free_port()));
        std::filesystem::remove_all(workdir);
        std::filesystem::create_directories(workdir);
        old_cwd = std::filesystem::current_path();
        std::filesystem::current_path(workdir);
        server = std::make_unique<ModelHubServer>(find_free_port());
        port = server->get_port();
        thread = std::thread([this] { server->start(); });
    }

    ~ServerGuard()
    {
        server->stop();
        if (thread.joinable()) { thread.join(); }
        // Destroy the server BEFORE removing workdir: its MetadataManager holds
        // an open SQLite handle on models_metadata.json inside workdir, and
        // members are only destroyed after the dtor body — remove_all() would
        // throw filesystem_error on the locked file, and a throwing dtor calls
        // std::terminate.
        server.reset();
        std::filesystem::current_path(old_cwd);
        std::filesystem::remove_all(workdir);
    }

    int get_port() const { return port; }
};

httplib::Client make_client(int port)
{
    return httplib::Client("127.0.0.1", port);
}

}  // namespace

TEST(ModelHubServer, HealthAndEmptyList)
{
    ServerGuard g;
    httplib::Client cli = make_client(g.get_port());

    // wait for server to come up
    httplib::Result res;
    for (int i = 0; i < 100; ++i) {
        res = cli.Get("/health");
        if (res && res->status == 200) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["status"], "ok");
    EXPECT_EQ(j["models"], 0);

    res = cli.Get("/v1/models");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(nlohmann::json::parse(res->body)["total"], 0);
}

TEST(ModelHubServer, RegisterUploadSearchDownloadDelete)
{
    ServerGuard g;
    httplib::Client cli = make_client(g.get_port());

    httplib::Result res;
    for (int i = 0; i < 100; ++i) {
        res = cli.Get("/health");
        if (res && res->status == 200) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(res);

    // register user
    res = cli.Post("/v1/auth/register", nlohmann::json{{"username", "alice"}}.dump(),
                   "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 201);
    const auto reg = nlohmann::json::parse(res->body);
    const std::string api_key = reg["api_key"].get<std::string>();
    ASSERT_FALSE(api_key.empty());
    const httplib::Headers auth = {{"Authorization", "Bearer " + api_key}};

    // upload without auth -> 401
    httplib::UploadFormDataItems noauth_items = {
        {"model", "GGUFfakebytes", "fake.gguf", "application/octet-stream"},
        {"metadata", "{\"name\":\"m\",\"author\":\"a\"}", "meta.json", "application/json"},
    };
    res = cli.Post("/v1/models/upload", noauth_items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 401);

    // upload with auth
    const std::string meta_json =
        nlohmann::json{{"name", "test-model"},
                       {"author", "alice"},
                       {"description", "a tiny test model"},
                       {"tags", {"llm", "tiny"}},
                       {"model_type", "gguf"}}.dump();
    httplib::UploadFormDataItems items = {
        {"model", "GGUFfakebytes0123456789", "fake.gguf", "application/octet-stream"},
        {"metadata", meta_json, "meta.json", "application/json"},
    };
    res = cli.Post("/v1/models/upload", auth, items);
    ASSERT_TRUE(res);
    if (res->status != 201) {
        ADD_FAILURE() << "upload failed: " << res->status << " " << res->body;
    }
    EXPECT_EQ(res->status, 201);
    const auto up = nlohmann::json::parse(res->body);
    const std::string model_id = up["id"].get<std::string>();
    ASSERT_FALSE(model_id.empty());
    EXPECT_EQ(up["checksum_sha256"].get<std::string>().size(), 64u);
    EXPECT_EQ(up["file_size"], 23);

    // list + get
    res = cli.Get("/v1/models");
    ASSERT_TRUE(res);
    EXPECT_EQ(nlohmann::json::parse(res->body)["total"], 1);

    res = cli.Get("/v1/models/" + model_id);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(nlohmann::json::parse(res->body)["name"], "test-model");

    res = cli.Get("/v1/models/nonexistent");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);

    // search
    res = cli.Get("/v1/models/search?q=tiny");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto search_j = nlohmann::json::parse(res->body);
    EXPECT_GE(search_j["total"], 1);
    EXPECT_EQ(search_j["data"][0]["model_id"], model_id);

    res = cli.Get("/v1/models/search?tags=llm");
    ASSERT_TRUE(res);
    EXPECT_GE(nlohmann::json::parse(res->body)["total"], 1);

    // download
    res = cli.Get("/v1/models/" + model_id + "/download");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->body, "GGUFfakebytes0123456789");

    // download count incremented
    res = cli.Get("/v1/models/" + model_id);
    ASSERT_TRUE(res);
    EXPECT_GE(nlohmann::json::parse(res->body)["download_count"].get<int>(), 1);

    // delete without auth -> 401
    res = cli.Delete("/v1/models/" + model_id);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 401);

    // delete as owner (uploaded_by == user)
    res = cli.Delete("/v1/models/" + model_id, auth);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    res = cli.Get("/v1/models/" + model_id);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
}
