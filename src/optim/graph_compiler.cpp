#include "graph_compiler.h"

#include <algorithm>

namespace serius {
namespace optim {

ExecutionPlan::ExecutionPlan()
    : workspace_(nullptr), workspace_size_(0)
{
}

ExecutionPlan::~ExecutionPlan()
{
    if (workspace_) {
        // Would free workspace memory here
        // For now, using standard allocator
        workspace_ = nullptr;
    }
}

void ExecutionPlan::execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const
{
    // Simplified execution: just copy inputs to outputs
    // In a real implementation, this would execute fused kernels
    if (inputs.size() == outputs.size()) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            outputs[i] = inputs[i].copy();
        }
    }
}

GraphCompiler::GraphCompiler(const Config& config)
    : config_(config)
{
}

GraphCompiler::~GraphCompiler()
{
}

void GraphCompiler::trace(void* model, const std::vector<size_t>& input_shape)
{
    (void)model;  // Suppress unused parameter warning
    // Simplified tracing: create a dummy graph
    // In a real implementation, this would execute a forward pass
    // and record all operations

    ops_.clear();

    // Example: linear layer + gelu + bias
    GraphOp op1;
    op1.type = "matmul";
    op1.inputs = {0};
    op1.outputs = {1};
    op1.output_shape = input_shape;
    ops_.push_back(op1);

    GraphOp op2;
    op2.type = "bias";
    op2.inputs = {1};
    op2.outputs = {2};
    op2.output_shape = input_shape;
    ops_.push_back(op2);

    GraphOp op3;
    op3.type = "gelu";
    op3.inputs = {2};
    op3.outputs = {3};
    op3.output_shape = input_shape;
    ops_.push_back(op3);
}

void GraphCompiler::optimize()
{
    if (config_.fuse_elementwise) {
        fuse_elementwise_chains();
    }

    if (config_.fuse_layernorm_linear) {
        fuse_layernorm_linear();
    }

    if (config_.eliminate_redundant_copies) {
        eliminate_redundant_copies();
    }

    if (config_.preallocate_buffers) {
        preallocate_buffers();
    }
}

void GraphCompiler::fuse_elementwise_chains()
{
    // Fuse bias + gelu into a single operation
    for (size_t i = 0; i + 1 < ops_.size(); ++i) {
        if (ops_[i].type == "bias" && ops_[i + 1].type == "gelu") {
            ops_[i].type = "fused_bias_gelu";
            ops_[i].fused_kernel = "fused_bias_gelu";
            ops_[i].outputs = ops_[i + 1].outputs;
            ops_.erase(ops_.begin() + i + 1);
            break;  // Only fuse one chain for simplicity
        }
    }
}

void GraphCompiler::fuse_layernorm_linear()
{
    // In a real implementation, this would find layernorm + matmul patterns
    // and fuse them into a single kernel
}

void GraphCompiler::eliminate_redundant_copies()
{
    // Remove contiguous operations that are unnecessary
    std::vector<GraphOp> optimized;
    for (const auto& op : ops_) {
        if (op.type != "contiguous") {
            optimized.push_back(op);
        }
    }
    ops_ = std::move(optimized);
}

void GraphCompiler::reorder_for_locality()
{
    // In a real implementation, this would reorder operations
    // to improve cache locality
}

void GraphCompiler::preallocate_buffers()
{
    // In a real implementation, this would calculate required workspace
    // and pre-allocate it
}

std::unique_ptr<ExecutionPlan> GraphCompiler::compile()
{
    auto plan = std::make_unique<ExecutionPlan>();
    plan->ops_ = ops_;

    // Calculate workspace size
    size_t total_size = 0;
    for (const auto& op : ops_) {
        size_t op_size = 1;
        for (size_t dim : op.output_shape) {
            op_size *= dim;
        }
        total_size += op_size * sizeof(float);
    }
    plan->workspace_size_ = total_size;

    // Allocate workspace (simplified)
    if (total_size > 0) {
        plan->workspace_ = ::operator new(total_size);
    }

    return plan;
}

}  // namespace optim
}  // namespace serius
