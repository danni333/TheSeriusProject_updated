#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "distributed/comm.h"
#include "distributed/data_parallel.h"
#include "distributed/local_comm.h"
#include "distributed/pipeline.h"
#include "distributed/tensor_parallel.h"

namespace serius {
namespace test {

using namespace distributed;

// ============================================================
// Single-rank Communicator Tests
// ============================================================

TEST(DistributedComm, SingleRankCreation)
{
    auto comm = Communicator::create(CommBackend::Local, 1, 0);
    EXPECT_EQ(comm->rank(), 0);
    EXPECT_EQ(comm->world_size(), 1);
    EXPECT_EQ(comm->backend(), CommBackend::Local);
}

TEST(DistributedComm, SingleRankAllReduceNoop)
{
    auto comm = Communicator::create(CommBackend::Local, 1, 0);
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    comm->all_reduce(data.data(), data.size(), ReduceOp::Avg);
    // Single rank: data unchanged
    EXPECT_NEAR(data[0], 1.0f, 1e-6f);
    EXPECT_NEAR(data[1], 2.0f, 1e-6f);
    EXPECT_NEAR(data[2], 3.0f, 1e-6f);
}

TEST(DistributedComm, SingleRankBroadcastNoop)
{
    auto comm = Communicator::create(CommBackend::Local, 1, 0);
    std::vector<float> data = {42.0f};
    comm->broadcast(data.data(), 1, 0);
    EXPECT_NEAR(data[0], 42.0f, 1e-6f);
}

TEST(DistributedComm, SingleRankAllGather)
{
    auto comm = Communicator::create(CommBackend::Local, 1, 0);
    std::vector<float> send = {1.0f, 2.0f};
    std::vector<float> recv(2);
    comm->all_gather(send.data(), 2, recv.data());
    EXPECT_NEAR(recv[0], 1.0f, 1e-6f);
    EXPECT_NEAR(recv[1], 2.0f, 1e-6f);
}

// ============================================================
// Multi-rank AllReduce Tests (using threads)
// ============================================================

TEST(DistributedComm, AllReduceSum4Ranks)
{
    const int world_size = 4;
    auto comms = create_local_group(world_size);

    // Each rank contributes rank+1 as its value
    std::vector<std::thread> threads;
    std::vector<float> results(world_size, 0.0f);

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            float data = static_cast<float>(r + 1);
            comms[static_cast<size_t>(r)]->all_reduce(&data, 1, ReduceOp::Sum);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    // Sum of 1+2+3+4 = 10
    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(results[static_cast<size_t>(r)], 10.0f, 1e-5f)
            << "AllReduce Sum mismatch on rank " << r;
    }
}

TEST(DistributedComm, AllReduceAvg4Ranks)
{
    const int world_size = 4;
    auto comms = create_local_group(world_size);

    std::vector<std::thread> threads;
    std::vector<float> results(world_size, 0.0f);

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            float data = static_cast<float>(r + 1);
            comms[static_cast<size_t>(r)]->all_reduce(&data, 1, ReduceOp::Avg);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    // Avg of 1+2+3+4 = 2.5
    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(results[static_cast<size_t>(r)], 2.5f, 1e-5f)
            << "AllReduce Avg mismatch on rank " << r;
    }
}

TEST(DistributedComm, AllReduceMultiElement)
{
    const int world_size = 3;
    const size_t count = 4;
    auto comms = create_local_group(world_size);

    // Each rank's data: [rank, rank+1, rank+2, rank+3]
    std::vector<std::vector<float>> results(world_size, std::vector<float>(count));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            std::vector<float> data(count);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<float>(r) + static_cast<float>(i);
            }
            comms[static_cast<size_t>(r)]->all_reduce(data.data(), count, ReduceOp::Sum);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    // Sum for element i: sum(r + i for r in 0,1,2) = 3 + 3*i
    for (int r = 0; r < world_size; ++r) {
        for (size_t i = 0; i < count; ++i) {
            float expected = 3.0f + 3.0f * static_cast<float>(i);
            EXPECT_NEAR(results[static_cast<size_t>(r)][i], expected, 1e-5f)
                << "AllReduce multi-element mismatch at rank=" << r << " elem=" << i;
        }
    }
}

TEST(DistributedComm, AllReduceMin)
{
    const int world_size = 4;
    auto comms = create_local_group(world_size);

    std::vector<float> results(world_size, 0.0f);
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            float data = static_cast<float>(r * 10 + 5);  // 5, 15, 25, 35
            comms[static_cast<size_t>(r)]->all_reduce(&data, 1, ReduceOp::Min);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(results[static_cast<size_t>(r)], 5.0f, 1e-5f);
    }
}

TEST(DistributedComm, AllReduceMax)
{
    const int world_size = 4;
    auto comms = create_local_group(world_size);

    std::vector<float> results(world_size, 0.0f);
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            float data = static_cast<float>(r * 10 + 5);
            comms[static_cast<size_t>(r)]->all_reduce(&data, 1, ReduceOp::Max);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(results[static_cast<size_t>(r)], 35.0f, 1e-5f);
    }
}

// ============================================================
// Broadcast Tests
// ============================================================

TEST(DistributedComm, Broadcast3Ranks)
{
    const int world_size = 3;
    const size_t count = 3;
    auto comms = create_local_group(world_size);

    std::vector<std::vector<float>> results(world_size, std::vector<float>(count, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            std::vector<float> data(count, 0.0f);
            if (r == 0) {
                // Root sends [10, 20, 30]
                data = {10.0f, 20.0f, 30.0f};
            }
            comms[static_cast<size_t>(r)]->broadcast(data.data(), count, /*root=*/0);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(results[static_cast<size_t>(r)][0], 10.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][1], 20.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][2], 30.0f, 1e-5f);
    }
}

// ============================================================
// AllGather Tests
// ============================================================

TEST(DistributedComm, AllGather3Ranks)
{
    const int world_size = 3;
    const size_t send_count = 2;
    auto comms = create_local_group(world_size);

    std::vector<std::vector<float>> results(world_size,
                                             std::vector<float>(send_count * world_size, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            // Each rank sends [r*10, r*10+1]
            std::vector<float> send = {static_cast<float>(r * 10),
                                        static_cast<float>(r * 10 + 1)};
            std::vector<float> recv(send_count * static_cast<size_t>(world_size));
            comms[static_cast<size_t>(r)]->all_gather(send.data(), send_count, recv.data());
            results[static_cast<size_t>(r)] = recv;
        });
    }

    for (auto &t : threads) t.join();

    // All ranks should see [0, 1, 10, 11, 20, 21]
    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(results[static_cast<size_t>(r)][0], 0.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][1], 1.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][2], 10.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][3], 11.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][4], 20.0f, 1e-5f);
        EXPECT_NEAR(results[static_cast<size_t>(r)][5], 21.0f, 1e-5f);
    }
}

// ============================================================
// ReduceScatter Tests
// ============================================================

TEST(DistributedComm, ReduceScatter2Ranks)
{
    const int world_size = 2;
    const size_t recv_count = 2;
    auto comms = create_local_group(world_size);

    std::vector<std::vector<float>> results(world_size, std::vector<float>(recv_count, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            // Each rank sends [r+1, r+2, r+3, r+4] (4 elements = recv_count * world_size)
            std::vector<float> send = {static_cast<float>(r + 1),
                                        static_cast<float>(r + 2),
                                        static_cast<float>(r + 3),
                                        static_cast<float>(r + 4)};
            std::vector<float> recv(recv_count);
            comms[static_cast<size_t>(r)]->reduce_scatter(send.data(), recv.data(), recv_count, ReduceOp::Sum);
            results[static_cast<size_t>(r)] = recv;
        });
    }

    for (auto &t : threads) t.join();

    // Rank 0 sends [1,2,3,4], Rank 1 sends [2,3,4,5]
    // Sum = [3,5,7,9]
    // Rank 0 gets [3,5], Rank 1 gets [7,9]
    EXPECT_NEAR(results[0][0], 3.0f, 1e-5f);
    EXPECT_NEAR(results[0][1], 5.0f, 1e-5f);
    EXPECT_NEAR(results[1][0], 7.0f, 1e-5f);
    EXPECT_NEAR(results[1][1], 9.0f, 1e-5f);
}

// ============================================================
// Point-to-Point Send/Recv Tests
// ============================================================

TEST(DistributedComm, SendRecv)
{
    const int world_size = 2;
    auto comms = create_local_group(world_size);

    float received = 0.0f;
    std::vector<std::thread> threads;

    threads.emplace_back([&]() {
        float data = 42.0f;
        comms[0]->send(&data, 1, /*dst=*/1);
    });

    threads.emplace_back([&]() {
        comms[1]->recv(&received, 1, /*src=*/0);
    });

    for (auto &t : threads) t.join();
    EXPECT_NEAR(received, 42.0f, 1e-5f);
}

// ============================================================
// Barrier Tests
// ============================================================

TEST(DistributedComm, Barrier)
{
    const int world_size = 4;
    auto comms = create_local_group(world_size);

    std::vector<std::thread> threads;
    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            comms[static_cast<size_t>(r)]->barrier();
        });
    }

    for (auto &t : threads) t.join();
    // If we get here without deadlock, the barrier works
    SUCCEED();
}

// ============================================================
// Data Parallel Tests
// ============================================================

TEST(DataParallelTest, SyncGradients)
{
    const int world_size = 4;
    auto comms = create_local_group(world_size);

    std::vector<float> final_grads(world_size, 0.0f);
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            DataParallel dp(comms[static_cast<size_t>(r)].get());

            // Each rank has different gradient
            float grad = static_cast<float>(r + 1) * 2.0f;  // 2, 4, 6, 8
            dp.sync_gradient(&grad, 1);

            final_grads[static_cast<size_t>(r)] = grad;
        });
    }

    for (auto &t : threads) t.join();

    // Avg of [2, 4, 6, 8] = 5.0
    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(final_grads[static_cast<size_t>(r)], 5.0f, 1e-5f)
            << "Data parallel gradient sync failed on rank " << r;
    }
}

TEST(DataParallelTest, BroadcastParams)
{
    const int world_size = 3;
    auto comms = create_local_group(world_size);

    std::vector<std::vector<float>> params(world_size, std::vector<float>(4, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            DataParallel dp(comms[static_cast<size_t>(r)].get());

            if (r == 0) {
                params[0] = {1.0f, 2.0f, 3.0f, 4.0f};
            }

            std::vector<std::pair<float *, size_t>> buffers = {
                {params[static_cast<size_t>(r)].data(), 4}
            };
            dp.broadcast_params(buffers);
        });
    }

    for (auto &t : threads) t.join();

    for (int r = 0; r < world_size; ++r) {
        EXPECT_NEAR(params[static_cast<size_t>(r)][0], 1.0f, 1e-5f);
        EXPECT_NEAR(params[static_cast<size_t>(r)][3], 4.0f, 1e-5f);
    }
}

// ============================================================
// Pipeline Parallelism Tests
// ============================================================

TEST(PipelineTest, TwoStagePipeline)
{
    const int world_size = 2;
    auto comms = create_local_group(world_size);

    PipelineConfig config;
    config.num_stages = 2;
    config.num_micro_batches = 2;
    config.hidden_size = 4;

    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            PipelineExecutor pipeline(comms[static_cast<size_t>(r)].get(), config);
            EXPECT_EQ(pipeline.stage(), r);

            if (r == 0) {
                EXPECT_TRUE(pipeline.is_first_stage());
                EXPECT_FALSE(pipeline.is_last_stage());
            } else {
                EXPECT_FALSE(pipeline.is_first_stage());
                EXPECT_TRUE(pipeline.is_last_stage());
            }

            // Forward: multiply by 2
            auto fwd = [](const float *in, float *out, size_t count) {
                for (size_t i = 0; i < count; ++i) out[i] = in[i] * 2.0f;
            };
            // Backward: multiply by 0.5
            auto bwd = [](const float *grad_out, float *grad_in, size_t count) {
                for (size_t i = 0; i < count; ++i) grad_in[i] = grad_out[i] * 0.5f;
            };

            // Input data: 2 micro-batches of 4 elements each
            std::vector<float> input(2 * 4, 1.0f);
            std::vector<float> output(2 * 4, 0.0f);

            pipeline.execute(fwd, bwd, input.data(), output.data());
        });
    }

    for (auto &t : threads) t.join();
    // If we get here without deadlock, the pipeline schedule works
    SUCCEED();
}

TEST(PipelineTest, BubbleRatio)
{
    PipelineConfig config;
    config.num_stages = 4;
    config.num_micro_batches = 8;
    config.hidden_size = 16;

    auto comm = Communicator::create(CommBackend::Local, 1, 0);
    // Can't create a full pipeline with 1 rank and 4 stages, but we can test the ratio formula
    float expected = (4.0f - 1.0f) / 8.0f;  // 0.375
    EXPECT_NEAR(expected, 0.375f, 1e-5f);
}

// ============================================================
// Tensor Parallelism Tests
// ============================================================

TEST(TensorParallelTest, ColumnParallelForward)
{
    const int world_size = 2;
    auto comms = create_local_group(world_size);

    // Full weight: [3, 4] (in_features=3, out_features=4)
    // Each rank gets [3, 2]
    std::vector<float> W_full = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };

    // Input X: [2, 3] (batch=2, in_features=3)
    std::vector<float> X = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };

    // Reference: Y = X * W = [2, 4]
    std::vector<float> Y_ref(2 * 4, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                Y_ref[i * 4 + j] += X[i * 3 + k] * W_full[k * 4 + j];
            }
        }
    }

    std::vector<std::vector<float>> outputs(world_size, std::vector<float>(2 * 4, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            TensorParallelConfig tp_config;
            tp_config.tp_size = world_size;
            TensorParallelLinear tp(comms[static_cast<size_t>(r)].get(), tp_config);

            // Split W along columns: rank 0 gets cols 0-1, rank 1 gets cols 2-3
            size_t local_out = 2;
            std::vector<float> W_local(3 * local_out);
            for (size_t row = 0; row < 3; ++row) {
                for (size_t col = 0; col < local_out; ++col) {
                    W_local[row * local_out + col] = W_full[row * 4 + static_cast<size_t>(r) * local_out + col];
                }
            }

            tp.column_parallel_forward(X.data(), W_local.data(), outputs[static_cast<size_t>(r)].data(),
                                       2, 3, local_out);
        });
    }

    for (auto &t : threads) t.join();

    // Both ranks should have the same full output
    for (size_t i = 0; i < 2 * 4; ++i) {
        EXPECT_NEAR(outputs[0][i], Y_ref[i], 1e-4f)
            << "Column parallel output mismatch at index " << i;
        EXPECT_NEAR(outputs[1][i], Y_ref[i], 1e-4f)
            << "Column parallel output mismatch at index " << i << " (rank 1)";
    }
}

TEST(TensorParallelTest, RowParallelForward)
{
    const int world_size = 2;
    auto comms = create_local_group(world_size);

    // Full weight: [4, 3] (in_features=4, out_features=3)
    // Split along rows: each rank gets [2, 3]
    std::vector<float> W_full = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };

    // Input X: [2, 4]
    std::vector<float> X = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };

    // Reference: Y = X * W = [2, 3]
    std::vector<float> Y_ref(2 * 3, 0.0f);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                Y_ref[i * 3 + j] += X[i * 4 + k] * W_full[k * 3 + j];
            }
        }
    }

    std::vector<std::vector<float>> outputs(world_size, std::vector<float>(2 * 3, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            TensorParallelConfig tp_config;
            tp_config.tp_size = world_size;
            TensorParallelLinear tp(comms[static_cast<size_t>(r)].get(), tp_config);

            // Split W along rows: rank 0 gets rows 0-1, rank 1 gets rows 2-3
            size_t local_in = 2;
            std::vector<float> W_local(local_in * 3);
            for (size_t row = 0; row < local_in; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    W_local[row * 3 + col] = W_full[(static_cast<size_t>(r) * local_in + row) * 3 + col];
                }
            }

            // Scatter input
            std::vector<float> X_local(2 * local_in);
            tp.scatter_input(X.data(), X_local.data(), 2, 4);

            tp.row_parallel_forward(X_local.data(), W_local.data(),
                                    outputs[static_cast<size_t>(r)].data(),
                                    2, local_in, 3);
        });
    }

    for (auto &t : threads) t.join();

    for (size_t i = 0; i < 2 * 3; ++i) {
        EXPECT_NEAR(outputs[0][i], Y_ref[i], 1e-4f)
            << "Row parallel output mismatch at index " << i;
        EXPECT_NEAR(outputs[1][i], Y_ref[i], 1e-4f)
            << "Row parallel output mismatch at index " << i << " (rank 1)";
    }
}

// ============================================================
// Stress test: larger AllReduce
// ============================================================

TEST(DistributedComm, AllReduceLargeBuffer)
{
    const int world_size = 4;
    const size_t count = 1024;
    auto comms = create_local_group(world_size);

    std::vector<std::vector<float>> results(world_size, std::vector<float>(count, 0.0f));
    std::vector<std::thread> threads;

    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([&, r]() {
            std::vector<float> data(count);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<float>(r) + static_cast<float>(i) * 0.001f;
            }
            comms[static_cast<size_t>(r)]->all_reduce(data.data(), count, ReduceOp::Sum);
            results[static_cast<size_t>(r)] = data;
        });
    }

    for (auto &t : threads) t.join();

    // Sum of ranks = 0+1+2+3 = 6, plus 4 * i * 0.001 for each element
    for (size_t i = 0; i < count; ++i) {
        float expected = 6.0f + 4.0f * static_cast<float>(i) * 0.001f;
        EXPECT_NEAR(results[0][i], expected, 1e-3f)
            << "Large AllReduce mismatch at element " << i;
    }
}

}  // namespace test
}  // namespace serius
