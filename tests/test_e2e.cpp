// End-to-end integration tests (Fase 56): real GGUF model through the real
// Server — tokenizer -> model -> sampler -> HTTP — plus a CLI lifecycle test
// that spawns the actual `serius serve` binary.
#include "server/server.h"

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <process.h>
#include <windows.h>
#endif

namespace {

/// Locate models/smollm2-q4 regardless of the directory ctest runs from.
std::filesystem::path find_model_dir(const std::string &rel)
{
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::path(rel),
        std::filesystem::path("..") / rel,
        std::filesystem::path("../..") / rel,
        std::filesystem::path("../../..") / rel,
        std::filesystem::absolute(std::filesystem::path(SERIOUS_REPO_ROOT) / rel),
    };
    for (const auto &c : candidates) {
        if (std::filesystem::exists(c)) {
            return c;
        }
    }
    return {};
}

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

void wait_for_server(httplib::Client &client, int timeout_seconds = 60)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        auto res = client.Get("/health");
        if (res && res->status == 200) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    FAIL() << "Server did not become ready within " << timeout_seconds << " seconds";
}

#ifdef _WIN32
/// Run a command line via CreateProcess, capturing stdout. Returns exit code;
/// stdout is appended to `out`.
DWORD run_capture(const std::string &cmd, std::string &out)
{
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_pipe = nullptr, write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        return STILL_ACTIVE;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::string mutable_cmd = cmd;
    if (!CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        CloseHandle(read_pipe);
        CloseHandle(write_pipe);
        return STILL_ACTIVE;
    }
    CloseHandle(write_pipe);

    // Wait first so a hung child cannot block the pipe read forever. The
    // budget must cover the slowest pipeline step (GGUF->safetensors convert
    // + CPU generate) on a loaded CI machine — 180s was hit under -j4.
    if (WaitForSingleObject(pi.hProcess, 600000) == WAIT_TIMEOUT) {
        TerminateProcess(pi.hProcess, 1);
    }

    char buf[4096];
    DWORD n = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf), &n, nullptr) && n > 0) {
        out.append(buf, n);
    }
    CloseHandle(read_pipe);

    DWORD code = STILL_ACTIVE;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}
#endif

/// Shared fixture: one Server with the real smollm2-q4 GGUF loaded.
class E2EServer : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        model_dir_ = find_model_dir("models/smollm2-q4");
        if (model_dir_.empty()) {
            GTEST_SKIP() << "models/smollm2-q4 not found";
        }
    }

    void SetUp() override
    {
        port_ = find_free_port();
        ASSERT_GT(port_, 0);

        serius::server::ServerConfig config;
        config.port = static_cast<uint16_t>(port_);
        config.host = "127.0.0.1";
        config.gpu_mode = serius::server::ServerConfig::GpuMode::kCpu;  // deterministic

        server_ = std::make_unique<serius::server::Server>(config);
        ASSERT_TRUE(server_->load_model(model_dir_.string(), "llama"));
        ASSERT_TRUE(server_->has_model());

        server_thread_ = std::thread([this] { server_->run(); });
        client_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
        wait_for_server(*client_, 60);
    }

    void TearDown() override
    {
        if (server_) {
            server_->stop();
        }
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

    nlohmann::json chat(const nlohmann::json &messages, int max_tokens = 16,
                        bool stream = false)
    {
        nlohmann::json body = {
            {"model", "smollm2-q4"},
            {"messages", messages},
            {"max_tokens", max_tokens},
            {"temperature", 0.0},
        };
        if (stream) {
            body["stream"] = true;
        }
        auto res = client_->Post("/v1/chat/completions", body.dump(), "application/json");
        EXPECT_TRUE(static_cast<bool>(res));
        if (!res) {
            return {};
        }
        EXPECT_EQ(res->status, 200);
        return nlohmann::json::parse(res->body);
    }

    static std::filesystem::path model_dir_;
    int port_ = 0;
    std::unique_ptr<serius::server::Server> server_;
    std::unique_ptr<httplib::Client> client_;
    std::thread server_thread_;
};

std::filesystem::path E2EServer::model_dir_;

// 1.1 Full inference pipeline: prompt -> tokens -> detokenized text.
TEST_F(E2EServer, FullPipelineChatCompletion)
{
    auto res = chat({{{"role", "user"}, {"content", "Say hello."}}}, 16);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res["object"], "chat.completion");
    ASSERT_FALSE(res["choices"].empty());
    const auto &choice = res["choices"][0];
    EXPECT_EQ(choice["message"]["role"], "assistant");
    EXPECT_TRUE(choice["message"]["content"].is_string());
    EXPECT_TRUE(choice.contains("finish_reason"));
    EXPECT_GT(res["usage"]["completion_tokens"].get<int>(), 0);
    EXPECT_GT(res["usage"]["prompt_tokens"].get<int>(), 0);
}

// 1.2 Multi-turn: conversation history is accepted and produces valid output.
TEST_F(E2EServer, MultiTurnChat)
{
    nlohmann::json messages = nlohmann::json::array({
        {{"role", "user"}, {"content", "My name is Marco."}},
        {{"role", "assistant"}, {"content", "Nice to meet you, Marco!"}},
        {{"role", "user"}, {"content", "What is my name?"}},
    });
    auto res = chat(messages, 16);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res["object"], "chat.completion");
    ASSERT_FALSE(res["choices"].empty());
    EXPECT_TRUE(res["choices"][0]["message"]["content"].is_string());
}

// 1.3 Server endpoints all live with a real model loaded.
TEST_F(E2EServer, AllEndpoints)
{
    auto health = client_->Get("/health");
    ASSERT_TRUE(health);
    EXPECT_EQ(health->status, 200);
    auto health_json = nlohmann::json::parse(health->body);
    EXPECT_EQ(health_json["status"], "ok");
    EXPECT_TRUE(health_json["model_loaded"].get<bool>());

    auto models = client_->Get("/v1/models");
    ASSERT_TRUE(models);
    EXPECT_EQ(models->status, 200);
    auto models_json = nlohmann::json::parse(models->body);
    ASSERT_FALSE(models_json["data"].empty());
    EXPECT_EQ(models_json["data"][0]["id"], "smollm2-q4");

    auto one = client_->Get("/v1/models/smollm2-q4");
    ASSERT_TRUE(one);
    EXPECT_EQ(one->status, 200);

    nlohmann::json comp_body = {
        {"model", "smollm2-q4"},
        {"prompt", "The capital of France is"},
        {"max_tokens", 8},
        {"temperature", 0.0},
    };
    auto comp = client_->Post("/v1/completions", comp_body.dump(), "application/json");
    ASSERT_TRUE(comp);
    EXPECT_EQ(comp->status, 200);
    auto comp_json = nlohmann::json::parse(comp->body);
    EXPECT_EQ(comp_json["object"], "text_completion");
    EXPECT_FALSE(comp_json["choices"][0]["text"].get<std::string>().empty());

    nlohmann::json emb_body = {
        {"model", "smollm2-q4"},
        {"input", "embed me"},
    };
    auto emb = client_->Post("/v1/embeddings", emb_body.dump(), "application/json");
    ASSERT_TRUE(emb);
    EXPECT_EQ(emb->status, 200);
    auto emb_json = nlohmann::json::parse(emb->body);
    EXPECT_EQ(emb_json["object"], "list");
    ASSERT_FALSE(emb_json["data"].empty());
    EXPECT_FALSE(emb_json["data"][0]["embedding"].empty());

    auto metrics = client_->Get("/metrics");
    ASSERT_TRUE(metrics);
    EXPECT_EQ(metrics->status, 200);
    EXPECT_NE(metrics->body.find("llm_requests_total"), std::string::npos);
}

// Streaming: SSE chunks are valid JSON deltas and the stream ends with [DONE].
TEST_F(E2EServer, StreamingSSE)
{
    nlohmann::json body = {
        {"model", "smollm2-q4"},
        {"messages", {{{"role", "user"}, {"content", "Count to three."}}}},
        {"max_tokens", 12},
        // temperature 0 + fixed seed: like the other tests in this fixture the
        // run must be deterministic — at the default temp 0.7 the sampler can
        // legitimately pick a stop token first, producing a valid but empty
        // stream and a flaky failure.
        {"temperature", 0.0},
        {"seed", 7},
        {"stream", true},
    };
    auto res = client_->Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);

    std::istringstream in(res->body);
    std::string line;
    int data_events = 0;
    bool saw_done = false;
    std::string accumulated;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("data: ", 0) != 0) {
            continue;
        }
        std::string payload = line.substr(6);
        if (payload == "[DONE]") {
            saw_done = true;
            continue;
        }
        ++data_events;
        auto chunk = nlohmann::json::parse(payload);
        // Skip non-content frames (e.g. {"error": ...} emitted mid-stream).
        if (!chunk.contains("choices") || chunk["choices"].empty()) {
            continue;
        }
        const auto &delta = chunk["choices"][0]["delta"];
        if (delta.contains("content") && delta["content"].is_string()) {
            accumulated += delta["content"].get<std::string>();
        }
    }
    EXPECT_GT(data_events, 0);
    EXPECT_TRUE(saw_done);
    if (accumulated.empty()) {
        ADD_FAILURE() << "stream produced no content; raw SSE body:\n"
                      << res->body;
    }
}

// Error handling: malformed JSON -> 400, unknown route -> 404.
TEST_F(E2EServer, ErrorHandling)
{
    auto bad = client_->Post("/v1/chat/completions", "{not json", "application/json");
    ASSERT_TRUE(bad);
    EXPECT_EQ(bad->status, 400);

    auto missing = client_->Post("/v1/chat/completions", "{}", "application/json");
    ASSERT_TRUE(missing);
    EXPECT_TRUE(missing->status == 400 || missing->status == 200);  // schema tolerated or rejected

    auto nf = client_->Get("/v1/nonexistent");
    ASSERT_TRUE(nf);
    EXPECT_EQ(nf->status, 404);
}

// Wrong JSON scalar types must produce a clean 400, never a 500 from a
// nlohmann::json::type_error escaping the handlers.
TEST_F(E2EServer, WrongScalarTypesReturn400)
{
    const std::vector<std::pair<std::string, std::string>> bad_bodies = {
        {"temperature string", R"({"model":"smollm2-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":"hot"})"},
        {"top_p object", R"({"model":"smollm2-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"top_p":{"x":1}})"},
        {"max_tokens string", R"({"model":"smollm2-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":"many"})"},
        {"max_new_tokens bool", R"({"model":"smollm2-q4","messages":[{"role":"user","content":"hi"}],"max_new_tokens":true})"},
        {"stream number", R"({"model":"smollm2-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":7})"},
        {"model number", R"({"model":42,"messages":[{"role":"user","content":"hi"}],"max_tokens":4})"},
        {"seed string", R"({"model":"smollm2-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"seed":"abc"})"},
        {"messages object", R"({"model":"smollm2-q4","messages":{"role":"user"},"max_tokens":4})"},
        {"completions prompt number", R"({"model":"smollm2-q4","prompt":123,"max_tokens":4})"},
        {"completions temperature array", R"({"model":"smollm2-q4","prompt":"hi","max_tokens":4,"temperature":[0.5]})"},
    };
    for (const auto &[name, body] : bad_bodies) {
        const std::string path = name.rfind("completions", 0) == 0 ? "/v1/completions" : "/v1/chat/completions";
        auto res = client_->Post(path, body, "application/json");
        ASSERT_TRUE(res) << name;
        EXPECT_EQ(res->status, 400) << name << " body=" << res->body;
    }

    // A non-string message role is tolerated (treated as unknown role),
    // not a 500 — prompt builders must degrade gracefully.
    auto res = client_->Post("/v1/chat/completions",
                             R"({"model":"smollm2-q4","messages":[{"role":42,"content":"hi"}],"max_tokens":4})",
                             "application/json");
    ASSERT_TRUE(res);
    EXPECT_NE(res->status, 500);
}

// 1.3 CLI lifecycle: spawn the real `serius serve` binary, wait for health,
// exercise one completion, then terminate it.
TEST(E2ECli, ServeBinaryLifecycle)
{
#ifdef _WIN32
    const auto model_dir = find_model_dir("models/smollm2-q4");
    if (model_dir.empty()) {
        GTEST_SKIP() << "models/smollm2-q4 not found";
    }
#ifndef SERIUS_EXE_PATH
#error "SERIUS_EXE_PATH compile definition required"
#endif
    const int port = find_free_port();
    ASSERT_GT(port, 0);

    std::string port_str = std::to_string(port);
    std::string model_str = model_dir.string();
    std::replace(model_str.begin(), model_str.end(), '/', '\\');

    const std::string cmd = "\"" + std::string(SERIUS_EXE_PATH) + "\" serve"
                            " --model \"" + model_str + "\""
                            " --port " + port_str + " --gpu cpu";

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    ASSERT_TRUE(CreateProcessA(nullptr, const_cast<LPSTR>(cmd.c_str()), nullptr, nullptr,
                               FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi));

    httplib::Client client("127.0.0.1", port);
    wait_for_server(client, 90);  // model load + tokenizer init can be slow on CPU

    auto health = client.Get("/health");
    ASSERT_TRUE(health);
    EXPECT_EQ(health->status, 200);
    auto health_json = nlohmann::json::parse(health->body);
    EXPECT_TRUE(health_json["model_loaded"].get<bool>());

    nlohmann::json body = {
        {"model", "smollm2-q4"},
        {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
        {"max_tokens", 8},
        {"temperature", 0.0},
    };
    auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    auto res_json = nlohmann::json::parse(res->body);
    EXPECT_EQ(res_json["object"], "chat.completion");

    TerminateProcess(pi.hProcess, 0);
    WaitForSingleObject(pi.hProcess, 5000);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
#else
    GTEST_SKIP() << "CLI lifecycle test is Windows-only";
#endif
}

// 1.4 Import -> Convert -> Infer: run the real `serius convert` binary on the
// GGUF, then serve the resulting safetensors directory and generate.
TEST(E2ECli, ConvertThenInfer)
{
#ifdef _WIN32
    const auto model_dir = find_model_dir("models/smollm2-q4");
    if (model_dir.empty()) {
        GTEST_SKIP() << "models/smollm2-q4 not found";
    }
    const auto gguf = model_dir / "SmolLM2-135M-Instruct-Q4_0.gguf";
    if (!std::filesystem::exists(gguf)) {
        GTEST_SKIP() << "GGUF file not found";
    }
#ifndef SERIUS_EXE_PATH
#error "SERIUS_EXE_PATH compile definition required"
#endif

    const auto out_dir = std::filesystem::temp_directory_path() /
                         ("serius_e2e_convert_" + std::to_string(::GetCurrentProcessId()));
    std::filesystem::remove_all(out_dir);

    std::string gguf_str = gguf.string();
    std::replace(gguf_str.begin(), gguf_str.end(), '/', '\\');
    std::string out_str = out_dir.string();
    std::replace(out_str.begin(), out_str.end(), '/', '\\');

    const std::string cmd = "\"" + std::string(SERIUS_EXE_PATH) + "\" convert"
                            " --input \"" + gguf_str + "\""
                            " --output \"" + out_str + "\"";

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    ASSERT_TRUE(CreateProcessA(nullptr, const_cast<LPSTR>(cmd.c_str()), nullptr, nullptr,
                               FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi));
    WaitForSingleObject(pi.hProcess, 120000);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    ASSERT_EQ(exit_code, 0u);

    // Converted directory must contain the HF-layout artifacts.
    EXPECT_TRUE(std::filesystem::exists(out_dir / "model.safetensors"));
    EXPECT_TRUE(std::filesystem::exists(out_dir / "config.json"));
    EXPECT_TRUE(std::filesystem::exists(out_dir / "tokenizer.json"));

    // Infer: serve the converted directory and run a chat completion.
    const int port = find_free_port();
    ASSERT_GT(port, 0);

    serius::server::ServerConfig config;
    config.port = static_cast<uint16_t>(port);
    config.host = "127.0.0.1";
    config.gpu_mode = serius::server::ServerConfig::GpuMode::kCpu;

    auto server = std::make_unique<serius::server::Server>(config);
    ASSERT_TRUE(server->load_model(out_dir.string(), "llama"));
    ASSERT_TRUE(server->has_model());

    std::thread server_thread([&] { server->run(); });
    httplib::Client client("127.0.0.1", port);
    wait_for_server(client, 30);

    nlohmann::json body = {
        // No "model" field: the loaded name is derived from the temp dir,
        // so an empty value resolves to the primary (only) model.
        {"messages", {{{"role", "user"}, {"content", "Say hello."}}}},
        {"max_tokens", 16},
        {"temperature", 0.0},
    };
    auto res = client.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(static_cast<bool>(res));
    EXPECT_EQ(res->status, 200);
    auto res_json = nlohmann::json::parse(res->body);
    EXPECT_EQ(res_json["object"], "chat.completion");
    ASSERT_FALSE(res_json["choices"].empty());
    EXPECT_TRUE(res_json["choices"][0]["message"]["content"].is_string());
    EXPECT_FALSE(res_json["choices"][0]["message"]["content"].get<std::string>().empty());

    server->stop();
    server_thread.join();
    std::filesystem::remove_all(out_dir);
#else
    GTEST_SKIP() << "convert pipeline test is Windows-only";
#endif
}

// 1.4 full pipeline via real binaries: import (local dir) -> convert ->
// generate, all offline.
TEST(E2ECli, ImportConvertGeneratePipeline)
{
#ifdef _WIN32
    const auto model_dir = find_model_dir("models/smollm2-q4");
    if (model_dir.empty()) {
        GTEST_SKIP() << "models/smollm2-q4 not found";
    }
    const auto gguf_name = "SmolLM2-135M-Instruct-Q4_0.gguf";
    const auto src_gguf = model_dir / gguf_name;
    if (!std::filesystem::exists(src_gguf)) {
        GTEST_SKIP() << "GGUF file not found";
    }
#ifndef SERIUS_EXE_PATH
#error "SERIUS_EXE_PATH compile definition required"
#endif

    const auto tmp = std::filesystem::temp_directory_path() /
                     ("serius_e2e_pipe_" + std::to_string(::GetCurrentProcessId()));
    std::filesystem::remove_all(tmp);
    const auto hub_dir = tmp / "hub";
    const auto hf_dir = tmp / "hf";

    auto win_path = [](std::filesystem::path p) {
        std::string s = p.string();
        std::replace(s.begin(), s.end(), '/', '\\');
        return s;
    };
    const std::string exe = "\"" + std::string(SERIUS_EXE_PATH) + "\"";
    std::string out;

    // 1. Import: local directory -> hub/<name>/
    DWORD code = run_capture(exe + " import --source \"" + win_path(model_dir) +
                                 "\" --output \"" + win_path(hub_dir) + "\"",
                             out);
    ASSERT_EQ(code, 0u) << out;
    const auto imported_gguf = hub_dir / model_dir.filename() / gguf_name;
    EXPECT_TRUE(std::filesystem::exists(imported_gguf));
    EXPECT_TRUE(std::filesystem::exists(hub_dir / model_dir.filename() / "tokenizer.json"));

    // 2. Convert: GGUF -> safetensors dir
    out.clear();
    code = run_capture(exe + " convert --input \"" + win_path(imported_gguf) +
                           "\" --output \"" + win_path(hf_dir) + "\"",
                       out);
    ASSERT_EQ(code, 0u) << out;
    EXPECT_TRUE(std::filesystem::exists(hf_dir / "model.safetensors"));
    EXPECT_TRUE(std::filesystem::exists(hf_dir / "config.json"));
    EXPECT_TRUE(std::filesystem::exists(hf_dir / "tokenizer.json"));

    // 3. Generate: non-interactive inference on the converted model.
    out.clear();
    code = run_capture(exe + " generate --model \"" + win_path(hf_dir) +
                           "\" --prompt \"Say hello.\" --max-tokens 8"
                           " --temperature 0 --gpu cpu",
                       out);
    ASSERT_EQ(code, 0u) << out;
    EXPECT_NE(out.find("Assistant:"), std::string::npos);
    const auto pos = out.find("Assistant:");
    const auto rest = out.substr(pos + std::string("Assistant:").size());
    EXPECT_NE(rest.find_first_not_of(" \t\r\n"), std::string::npos);

    std::filesystem::remove_all(tmp);
#else
    GTEST_SKIP() << "pipeline test is Windows-only";
#endif
}

}  // namespace
