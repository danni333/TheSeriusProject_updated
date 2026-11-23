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

TEST(ExecutionPlan, ExecuteThrowsNotImplemented)
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

    // Synthetic trace() ops carry no replay function, so execution must fail
    // loudly rather than silently copying inputs to outputs.
    EXPECT_THROW(plan->execute(inputs, outputs), std::runtime_error);
}

TEST(ExecutionPlan, IsExecutable)
{
    // Synthetic trace() plans are descriptive only.
    GraphCompiler synth;
    synth.trace(nullptr, {4, 4});
    auto synth_plan = synth.compile();
    ASSERT_NE(synth_plan, nullptr);
    EXPECT_FALSE(synth_plan->is_executable());

    // Plans from trace_graph() carry replay functions.
    core::Tensor a_data({4, 4}), b_data({4, 4});
    a_data.rand();
    b_data.rand();
    auto *a = new core::autograd::Node(std::move(a_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    a->requires_grad = b->requires_grad = false;
    auto *mm = core::autograd::ops::matmul(a, b);
    auto *out = core::autograd::ops::relu(mm);

    GraphCompiler real;
    real.trace_graph(out);
    auto real_plan = real.compile();
    ASSERT_NE(real_plan, nullptr);
    EXPECT_TRUE(real_plan->is_executable());

    delete out;
    delete mm;
    delete a;
    delete b;
}

TEST(ExecutionPlan, ExecuteReplaysTracedGraph)
{
    // Build: relu(gelu(matmul(a, b))) — same graph as TraceGraphFromRealAutogradGraph.
    core::Tensor a_data({4, 8}), b_data({8, 4});
    a_data.fill(0.5f);
    b_data.fill(0.25f);
    auto *a = new core::autograd::Node(std::move(a_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    a->requires_grad = b->requires_grad = false;

    auto *mm = core::autograd::ops::matmul(a, b);
    auto *g1 = core::autograd::ops::gelu(mm);
    auto *out = core::autograd::ops::relu(g1);

    GraphCompiler compiler;
    compiler.trace_graph(out);
    auto plan = compiler.compile();
    ASSERT_NE(plan, nullptr);

    // Two leaf operands (a, b), one sink output.
    EXPECT_EQ(plan->num_inputs(), 2u);
    EXPECT_EQ(plan->num_outputs(), 1u);
    EXPECT_EQ(plan->num_ops(), 3u);

    std::vector<Tensor> inputs;
    inputs.push_back(a->data);
    inputs.push_back(b->data);
    std::vector<Tensor> outputs;
    ASSERT_NO_THROW(plan->execute(inputs, outputs));

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].shape(), out->data.shape());
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], out->data[i]);
    }

    // Re-executing with different inputs must actually recompute, not replay
    // the traced tensors: a=1, b=1 -> matmul = 8 -> gelu(8) ~ 8 -> relu = 8.
    inputs[0].fill(1.0f);
    inputs[1].fill(1.0f);
    ASSERT_NO_THROW(plan->execute(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1u);
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        EXPECT_NEAR(outputs[0][i], 8.0f, 1e-3f);
    }

    // Wrong input count must fail loudly.
    std::vector<Tensor> bad_inputs(inputs.begin(), inputs.end() - 1);
    EXPECT_THROW(plan->execute(bad_inputs, outputs), std::runtime_error);

    delete out;
    delete g1;
    delete mm;
    delete a;
    delete b;
}

TEST(ExecutionPlan, ExecuteBranchingGraph)
{
    // y = relu(a + b) * 2 style graph exercising multi-sink-free DAG:
    // out = gelu(add(matmul(a,b), c))
    core::Tensor a_data({2, 3}), b_data({3, 2}), c_data({2, 2});
    a_data.rand();
    b_data.rand();
    c_data.rand();
    auto *a = new core::autograd::Node(std::move(a_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    auto *c = new core::autograd::Node(std::move(c_data));
    a->requires_grad = b->requires_grad = c->requires_grad = false;

    auto *mm = core::autograd::ops::matmul(a, b);
    auto *sum = core::autograd::ops::add(mm, c);
    auto *out = core::autograd::ops::gelu(sum);

    GraphCompiler compiler;
    compiler.trace_graph(out);
    auto plan = compiler.compile();
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->num_inputs(), 3u);
    EXPECT_EQ(plan->num_outputs(), 1u);

    std::vector<Tensor> inputs;
    inputs.push_back(a->data);
    inputs.push_back(b->data);
    inputs.push_back(c->data);
    std::vector<Tensor> outputs;
    ASSERT_NO_THROW(plan->execute(inputs, outputs));

    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].shape(), out->data.shape());
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], out->data[i]);
    }

    delete out;
    delete sum;
    delete mm;
    delete a;
    delete b;
    delete c;
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

TEST(GraphCompiler, FuseLayernormLinear)
{
    // out = relu(matmul(layernorm(x, g, b), w))
    core::Tensor x_data({2, 4}), g_data({4}), b_data({4}), w_data({4, 3});
    x_data.rand();
    g_data.rand();
    b_data.rand();
    w_data.rand();
    auto *x = new core::autograd::Node(std::move(x_data));
    auto *g = new core::autograd::Node(std::move(g_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    auto *w = new core::autograd::Node(std::move(w_data));
    x->requires_grad = g->requires_grad = b->requires_grad = w->requires_grad = false;

    auto *ln = core::autograd::ops::layernorm(x, g, b);
    auto *mm = core::autograd::ops::matmul(ln, w);
    auto *out = core::autograd::ops::relu(mm);

    GraphCompiler compiler;
    compiler.trace_graph(out);
    compiler.optimize();

    const auto& ops = compiler.get_ops();
    ASSERT_EQ(ops.size(), 2u);
    EXPECT_EQ(ops[0].type, "fused_layernorm_linear");
    EXPECT_EQ(ops[1].type, "relu");
    // fused inputs = layernorm's 3 inputs + matmul's weight
    EXPECT_EQ(ops[0].inputs.size(), 4u);

    // The fused plan must still produce the eager result.
    auto plan = compiler.compile();
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->num_inputs(), 4u);
    EXPECT_EQ(plan->num_outputs(), 1u);

    std::vector<Tensor> inputs;
    inputs.push_back(x->data);
    inputs.push_back(g->data);
    inputs.push_back(b->data);
    inputs.push_back(w->data);
    std::vector<Tensor> outputs;
    ASSERT_NO_THROW(plan->execute(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].shape(), out->data.shape());
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], out->data[i]);
    }

    delete out;
    delete mm;
    delete ln;
    delete x;
    delete g;
    delete b;
    delete w;
}

TEST(GraphCompiler, PreallocateBuffersUsesLivenessPeak)
{
    // out = relu(matmul(relu(matmul(a, a)), ...)) chain where each {8,8}
    // intermediate dies as soon as it is consumed: peak live = one tensor,
    // far below the sum of all op outputs.
    core::Tensor a_data({8, 8});
    a_data.rand();
    auto *a = new core::autograd::Node(std::move(a_data));
    a->requires_grad = false;

    auto *m1 = core::autograd::ops::matmul(a, a);
    auto *r = core::autograd::ops::relu(m1);
    auto *m2 = core::autograd::ops::matmul(r, r);
    auto *out = core::autograd::ops::relu(m2);

    GraphCompiler compiler;
    compiler.trace_graph(out);
    compiler.optimize();
    auto plan = compiler.compile();
    ASSERT_NE(plan, nullptr);

    const size_t tensor_bytes = 8 * 8 * sizeof(float);
    // Three computed {8,8} tensors exist, but at most ~2 are live at once
    // (a leaf + current intermediate, plus the final output): the peak must
    // be well under the naive sum of all outputs (4 tensors).
    EXPECT_LT(plan->workspace_size(), 4 * tensor_bytes);
    EXPECT_GE(plan->workspace_size(), tensor_bytes);

    delete out;
    delete m2;
    delete r;
    delete m1;
    delete a;
}

TEST(GraphCompiler, EliminateCopiesPreservesExecution)
{
    // out = relu(contiguous(transpose(matmul(a, b)))) — transpose yields a
    // non-contiguous tensor so contiguous() creates a real node; the pass
    // must remove it AND rewire consumers, not leave dangling slots.
    core::Tensor a_data({4, 4}), b_data({4, 4});
    a_data.rand();
    b_data.rand();
    auto *a = new core::autograd::Node(std::move(a_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    a->requires_grad = b->requires_grad = false;

    auto *mm = core::autograd::ops::matmul(a, b);
    auto *t = core::autograd::ops::transpose(mm, 0, 1);
    auto *c = core::autograd::ops::contiguous(t);
    ASSERT_NE(c, t);  // contiguous must be a real node, not an alias
    auto *out = core::autograd::ops::relu(c);

    GraphCompiler::Config cfg;
    cfg.eliminate_redundant_copies = true;
    GraphCompiler compiler(cfg);
    compiler.trace_graph(out);
    compiler.optimize();

    for (const auto& op : compiler.get_ops()) {
        EXPECT_FALSE(op.type == "contiguous" ||
                     op.type.rfind("contiguous(", 0) == 0);
    }

    auto plan = compiler.compile();
    ASSERT_NE(plan, nullptr);
    std::vector<Tensor> inputs;
    inputs.push_back(a->data);
    inputs.push_back(b->data);
    std::vector<Tensor> outputs;
    ASSERT_NO_THROW(plan->execute(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1u);
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], out->data[i]);
    }

    delete out;
    delete c;
    delete t;
    delete mm;
    delete a;
    delete b;
}

TEST(GraphCompiler, ReorderForLocalityPreservesExecution)
{
    // Branching graph: out = add(relu(mm), gelu(mm)) where mm feeds two
    // consumers — reordering must keep a valid topological order.
    core::Tensor a_data({4, 4}), b_data({4, 4});
    a_data.rand();
    b_data.rand();
    auto *a = new core::autograd::Node(std::move(a_data));
    auto *b = new core::autograd::Node(std::move(b_data));
    a->requires_grad = b->requires_grad = false;

    auto *mm = core::autograd::ops::matmul(a, b);
    auto *r = core::autograd::ops::relu(mm);
    auto *g = core::autograd::ops::gelu(mm);
    auto *out = core::autograd::ops::add(r, g);

    GraphCompiler compiler;
    compiler.trace_graph(out);
    const size_t ops_before = compiler.get_ops().size();
    compiler.optimize();
    ASSERT_EQ(compiler.get_ops().size(), ops_before);

    auto plan = compiler.compile();
    ASSERT_NE(plan, nullptr);
    std::vector<Tensor> inputs;
    inputs.push_back(a->data);
    inputs.push_back(b->data);
    std::vector<Tensor> outputs;
    ASSERT_NO_THROW(plan->execute(inputs, outputs));
    ASSERT_EQ(outputs.size(), 1u);
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], out->data[i]);
    }

    delete out;
    delete g;
    delete r;
    delete mm;
    delete a;
    delete b;
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
