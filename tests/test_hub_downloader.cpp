#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "hub/downloader.h"

namespace {

using serius::hub::ModelDownloader;

TEST(HubDownloaderTest, AcceptsWellFormedRepoIds)
{
    EXPECT_TRUE(ModelDownloader::is_valid_repo_id("Qwen/Qwen2.5-0.5B-Instruct"));
    EXPECT_TRUE(ModelDownloader::is_valid_repo_id("bartowski/Llama-3.2-1B-Instruct-GGUF"));
    EXPECT_TRUE(ModelDownloader::is_valid_repo_id("a/b"));
    EXPECT_TRUE(ModelDownloader::is_valid_repo_id("user_1/model.v2"));
}

TEST(HubDownloaderTest, RejectsMalformedRepoIds)
{
    const std::vector<std::string> bad = {
        "",      "noslash", "a/b/c",    "/b",     "a/",
        "../b",  "a/..",    "a/b..c",   "-a/b",   "a/.b",
        "a b/c", "a/b?x=1", "a/b#frag", "a/b\nc", std::string(97, 'a') + "/b",
    };
    for (const auto &id : bad) {
        EXPECT_FALSE(ModelDownloader::is_valid_repo_id(id)) << id;
    }
}

TEST(HubDownloaderTest, RejectsUnsafeRepoPaths)
{
    EXPECT_TRUE(ModelDownloader::is_safe_repo_path("model-Q4_K_M.gguf"));
    EXPECT_TRUE(ModelDownloader::is_safe_repo_path("sub/dir/model.gguf"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path(""));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("/etc/passwd"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("../escape.gguf"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("a/../../escape.gguf"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("a//b.gguf"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("C:/x.gguf"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("a\\b.gguf"));
    EXPECT_FALSE(ModelDownloader::is_safe_repo_path("dir/"));
}

TEST(HubDownloaderTest, ShellMetacharactersInRepoIdAreRejectedAndNotExecuted)
{
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path marker = std::filesystem::temp_directory_path() / ("serius_inj_" + stamp);
    const std::string m = marker.string();
    std::filesystem::remove(marker);

    const std::vector<std::string> payloads = {
        "x\"; touch " + m + "; echo \"",       "x/$(touch " + m + ")", "x/`touch " + m + "`",
        "x/y\" && touch " + m + " && echo \"", "x/y; touch " + m,      "x/y | touch " + m,
    };

    ModelDownloader dl;
    for (const auto &repo : payloads) {
        std::string local_path, chosen;
        EXPECT_FALSE(ModelDownloader::is_valid_repo_id(repo)) << repo;
        EXPECT_FALSE(dl.resolve_hf_gguf(repo, "", local_path, chosen)) << repo;
        EXPECT_TRUE(dl.list_hf_files(repo).empty()) << repo;
        EXPECT_FALSE(dl.download_hf_file(repo, "model.gguf", m + ".out")) << repo;
        EXPECT_FALSE(dl.download_hf(repo, m + ".dir")) << repo;
    }
    EXPECT_FALSE(std::filesystem::exists(marker));
    EXPECT_FALSE(std::filesystem::exists(m + ".out"));
    EXPECT_FALSE(std::filesystem::exists(m + ".dir"));
}

TEST(HubDownloaderTest, DownloadUrlRejectsNonHttpSchemes)
{
    ModelDownloader dl;
    EXPECT_FALSE(dl.download_url("file:///etc/passwd", "out.bin"));
    EXPECT_FALSE(dl.download_url("-K/etc/passwd", "out.bin"));
}

}  // namespace
