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

constexpr int kPort = 18099;
constexpr const char *kBase = "http://127.0.0.1:18099";

struct ServerGuard
{
    ModelHubServer server{kPort};
    std::thread thread;
    std::filesystem::path old_cwd;
    std::filesystem::path workdir;

    explicit ServerGuard()
    {
        workdir = std::filesystem::temp_directory_path() / "serius_hub_server_test";
        std::filesystem::remove_all(workdir);
        std::filesystem::create_directories(workdir);
        old_cwd = std::filesystem::current_path();
        std::filesystem::current_path(workdir);
        thread = std::thread([this] { server.start(); });
    }

    ~ServerGuard()
    {
        server.stop();
        if (thread.joinable()) { thread.join(); }
        std::filesystem::current_path(old_cwd);
        std::filesystem::remove_all(workdir);
    }
};

httplib::Result get(httplib::Client &cli, const std::string &path,
                    const httplib::Headers &headers = {})
{
    return cli.Get(path, headers);
}

}  // namespace

TEST(ModelHubServer, HealthAndEmptyList)
{
    ServerGuard g;
    httplib::Client cli(kBase);

    // wait for server to come up
    httplib::Result res;
    for (int i = 0; i < 100; ++i) {
        res = get(cli, "/health");
        if (res && res->status == 200) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto j = nlohmann::json::parse(res->body);
    EXPECT_EQ(j["status"], "ok");
    EXPECT_EQ(j["models"], 0);

    res = get(cli, "/v1/models");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(nlohmann::json::parse(res->body)["total"], 0);
}

TEST(ModelHubServer, RegisterUploadSearchDownloadDelete)
{
    ServerGuard g;
    httplib::Client cli(kBase);

    httplib::Result res;
    for (int i = 0; i < 100; ++i) {
        res = get(cli, "/health");
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
    httplib::MultipartFormDataItems noauth_items = {
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
    httplib::MultipartFormDataItems items = {
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
    res = get(cli, "/v1/models");
    ASSERT_TRUE(res);
    EXPECT_EQ(nlohmann::json::parse(res->body)["total"], 1);

    res = get(cli, "/v1/models/" + model_id);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(nlohmann::json::parse(res->body)["name"], "test-model");

    res = get(cli, "/v1/models/nonexistent");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);

    // search
    res = get(cli, "/v1/models/search?q=tiny");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto search_j = nlohmann::json::parse(res->body);
    EXPECT_GE(search_j["total"], 1);
    EXPECT_EQ(search_j["data"][0]["model_id"], model_id);

    res = get(cli, "/v1/models/search?tags=llm");
    ASSERT_TRUE(res);
    EXPECT_GE(nlohmann::json::parse(res->body)["total"], 1);

    // download
    res = get(cli, "/v1/models/" + model_id + "/download");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(res->body, "GGUFfakebytes0123456789");

    // download count incremented
    res = get(cli, "/v1/models/" + model_id);
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

    res = get(cli, "/v1/models/" + model_id);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
}
