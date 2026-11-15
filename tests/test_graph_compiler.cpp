#include <gtest/gtest.h>

#include "autograd/ops.h"
#include "optim/graph_compiler.h"
#include "tensor/tensor.h"

using namespace serius;
using namespace serius::optim;
using namespace serius::core;

TEST(GraphCompiler, TraceCreatesOps)
{
    GraphCompiler compiler;
    std::vector<size_t> input_shape = {10, 64};

    compiler.trace(nullptr, input_shape);

    const auto& ops = compiler.get_ops();
    EXPECT_GT(ops.size(), 0);
}

TEST(GraphCompiler, FuseElementwise)
{
    GraphCompiler::Config cfg;
    cfg.fuse_elementwise = true;
    GraphCompiler compiler(cfg);

    std::vector<size_t> input_shape = {10, 64};
    compiler.trace(nullptr, input_shape);
    compiler.optimize();

    const auto& ops = compiler.get_ops();

    // Should have fused bias + gelu
    bool has_fused = false;
    for (const auto& op : ops) {
        if (op.type == "fused_bias_gelu") {
            has_fused = true;
            break;
        }
    }
    EXPECT_TRUE(has_fused);
}

TEST(GraphCompiler, EliminateRedundantCopies)
{
    GraphCompiler::Config cfg;
    cfg.eliminate_redundant_copies = true;
    GraphCompiler compiler(cfg);

    std::vector<size_t> input_shape = {10, 64};
    compiler.trace(nullptr, input_shape);
    compiler.optimize();

    const auto& ops = compiler.get_ops();

    // Should not have contiguous operations
    for (const auto& op : ops) {
        EXPECT_NE(op.type, "contiguous");
    }
}

TEST(ExecutionPlan, CreateFromCompiler)
{
    GraphCompiler compiler;
    std::vector<size_t> input_shape = {10, 64};
    compiler.trace(nullptr, input_shape);
    compiler.optimize();

    auto plan = compiler.compile();
    EXPECT_NE(plan, nullptr);
    EXPECT_GT(plan->num_ops(), 0);
}

TEST(ExecutionPlan, Execute)
{
    GraphCompiler compiler;
    std::vector<size_t> input_shape = {10, 64};
    compiler.trace(nullptr, input_shape);
    compiler.optimize();

    auto plan = compiler.compile();

    std::vector<Tensor> inputs;
    inputs.emplace_back(input_shape);
    inputs[0].fill(1.0f);

    std::vector<Tensor> outputs;
    outputs.emplace_back(input_shape);

    plan->execute(inputs, outputs);

    // Output should be valid (not all zeros)
    bool has_nonzero = false;
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        if (outputs[0].item(i) != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(GraphCompiler, ConfigDefaults)
{
    GraphCompiler::Config cfg;
    EXPECT_TRUE(cfg.fuse_elementwise);
    EXPECT_TRUE(cfg.fuse_layernorm_linear);
    EXPECT_TRUE(cfg.eliminate_redundant_copies);
    EXPECT_TRUE(cfg.preallocate_buffers);
}

TEST(GraphCompiler, TraceGraphFromRealAutogradGraph)
{
    // Build: relu(gelu(matmul(a, b))) and trace the real graph.
    core::Tensor a_data({4, 8}), b_data({8, 4});
    a_data.fill(0.5f);
    b_data.fill(0.25f);
    auto *a = new core::autograd::Node(std::move(a_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    a->requires_grad = b->requires_grad = false;

    auto *mm = core::autograd::ops::matmul(a, b);   // "matmul"
    auto *g1 = core::autograd::ops::gelu(mm);       // "gelu"
    auto *out = core::autograd::ops::relu(g1);      // "relu"

    GraphCompiler compiler;
    compiler.trace_graph(out);

    const auto &ops = compiler.get_ops();
    ASSERT_EQ(ops.size(), 3u);
    EXPECT_EQ(ops[0].type, "matmul");
    EXPECT_EQ(ops[1].type, "gelu");
    EXPECT_EQ(ops[2].type, "relu");
    // matmul has two leaf inputs; gelu/relu chain single inputs.
    EXPECT_EQ(ops[0].inputs.size(), 2u);
    EXPECT_EQ(ops[1].inputs.size(), 1u);
    EXPECT_EQ(ops[2].inputs.size(), 1u);
    EXPECT_EQ(ops[2].output_shape, std::vector<size_t>({4, 4}));

    delete out;
    delete g1;
    delete mm;
    delete a;
    delete b;
}

TEST(GraphCompiler, TraceGraphNullIsEmpty)
{
    GraphCompiler compiler;
    compiler.trace_graph(nullptr);
    EXPECT_TRUE(compiler.get_ops().empty());
}

TEST(GraphCompiler, DisableOptimizations)
{
    GraphCompiler::Config cfg;
    cfg.fuse_elementwise = false;
    cfg.eliminate_redundant_copies = false;

    GraphCompiler compiler(cfg);
    std::vector<size_t> input_shape = {10, 64};
    compiler.trace(nullptr, input_shape);
    compiler.optimize();

    const auto& ops = compiler.get_ops();

    // Should not have fused operations
    for (const auto& op : ops) {
        EXPECT_NE(op.type, "fused_bias_gelu");
    }
}
