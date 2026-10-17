#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "attention/flash_attn.h"
#include "tensor/tensor.h"

namespace {
void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void benchmark_sequence(int sequence_length, int head_dim, const std::vector<std::array<int, 2>> &configurations)
{
    constexpr int warmup_iterations = 2;
    constexpr int measured_iterations = 20;

    serius::core::Tensor query(
        {1, 1, static_cast<size_t>(sequence_length), static_cast<size_t>(head_dim)});
    serius::core::Tensor key({1, 1, static_cast<size_t>(sequence_length), static_cast<size_t>(head_dim)});
    serius::core::Tensor value(
        {1, 1, static_cast<size_t>(sequence_length), static_cast<size_t>(head_dim)});
    query.rand();
    key.rand();
    value.rand();
    query.syncToDevice();
    key.syncToDevice();
    value.syncToDevice();

    std::cout << "\nSequence length: " << sequence_length << ", head dimension: " << head_dim << '\n';
    std::cout << "block_m  block_n  average_ms\n";

    for (const auto &blocks : configurations) {
        serius::model::attention::FlashAttentionConfig config;
        config.is_causal = false;
        config.softmax_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
        config.block_m = blocks[0];
        config.block_n = blocks[1];

        for (int i = 0; i < warmup_iterations; ++i) {
            auto output = serius::model::attention::flash_attention_forward(query, key, value, config);
        }

        cudaEvent_t start;
        cudaEvent_t stop;
        check_cuda(cudaEventCreate(&start), "Create CUDA start event");
        check_cuda(cudaEventCreate(&stop), "Create CUDA stop event");
        check_cuda(cudaEventRecord(start), "Record CUDA start event");
        for (int i = 0; i < measured_iterations; ++i) {
            auto output = serius::model::attention::flash_attention_forward(query, key, value, config);
        }
        check_cuda(cudaEventRecord(stop), "Record CUDA stop event");
        check_cuda(cudaEventSynchronize(stop), "Wait for Flash Attention benchmark");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "Measure Flash Attention benchmark");
        check_cuda(cudaEventDestroy(start), "Destroy CUDA start event");
        check_cuda(cudaEventDestroy(stop), "Destroy CUDA stop event");
        std::cout << std::setw(7) << blocks[0] << std::setw(9) << blocks[1] << std::setw(13)
                  << std::fixed << std::setprecision(3) << elapsed_ms / measured_iterations << '\n';
    }
}
}  // namespace

int main(int argc, char **argv)
{
    try {
        const std::array<std::vector<std::array<int, 2>>, 3> configurations = {{
            {{16, 16}, {32, 32}, {64, 32}, {128, 64}, {128, 128}},
            {{16, 16}, {32, 32}, {64, 32}, {64, 64}, {128, 32}},
            {{16, 16}, {16, 32}, {32, 16}, {32, 32}, {64, 16}},
        }};
        const std::array<int, 3> head_dimensions = {32, 64, 128};
        const auto run_all = [&](int sequence_length) {
            for (size_t i = 0; i < head_dimensions.size(); ++i) {
                benchmark_sequence(sequence_length, head_dimensions[i], configurations[i]);
            }
        };
        if (argc == 1) {
            for (int sequence_length : {128, 512, 1024}) run_all(sequence_length);
        } else {
            for (int i = 1; i < argc; ++i) {
                const int sequence_length = std::stoi(argv[i]);
                if (sequence_length <= 0) throw std::invalid_argument("Sequence lengths must be positive");
                run_all(sequence_length);
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Flash Attention benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
