#include "server/server.h"

#include <gtest/gtest.h>
#include <httplib.h>

#include <chrono>
#include <thread>

namespace {

TEST(ServerTest, StopsFromAnotherThread)
{
    int port = 0;
    bool probe_result = false;
    {
        httplib::Server port_probe;
        port = port_probe.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);

        std::thread port_probe_thread([&] { probe_result = port_probe.listen_after_bind(); });
        port_probe.wait_until_ready();
        port_probe.stop();
        port_probe_thread.join();
    }
    ASSERT_TRUE(probe_result);

    serius::server::Server server(static_cast<uint16_t>(port));
    int run_result = -1;
    std::thread server_thread([&] { run_result = server.run(); });

    httplib::Client client("127.0.0.1", port);
    httplib::Result response;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        response = client.Get("/health");
        if (response) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    server.stop();
    server_thread.join();

    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 200);
    EXPECT_EQ(run_result, 0);
}

}  // namespace
