#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace optim {

using core::Tensor;

/// Represents a single operation in the computation graph.
struct GraphOp {
    std::string type;  // "matmul", "layernorm", "gelu", "bias", etc.
    std::vector<size_t> inputs;  // Indices of input tensors
    std::vector<size_t> outputs;  // Indices of output tensors
    std::vector<size_t> output_shape;
    std::string fused_kernel;  // Name of fused kernel if applicable
};

/// Optimized execution plan for a model.
class ExecutionPlan
{
public:
    ExecutionPlan();
    ~ExecutionPlan();

    /// Execute the plan with the given input tensors.
    /// @param inputs Input tensors
    /// @param outputs Output tensors (pre-allocated)
    void execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const;

    /// Get the number of operations in the plan.
    size_t num_ops() const { return ops_.size(); }

private:
    std::vector<GraphOp> ops_;
    void* workspace_;  // Pre-allocated workspace memory
    size_t workspace_size_;

    friend class GraphCompiler;
};

/// Compiles and optimizes a computation graph.
class GraphCompiler
{
public:
    struct Config {
        bool fuse_elementwise = true;
        bool fuse_layernorm_linear = true;
        bool eliminate_redundant_copies = true;
        bool preallocate_buffers = true;
    };

    explicit GraphCompiler(const Config& config = Config{});
    ~GraphCompiler();

    /// Trace the graph by running a dummy forward pass.
    /// @param model Reference to the model (not implemented in this version)
    /// @param input_shape Shape of the input tensor
    void trace(void* model, const std::vector<size_t>& input_shape);

    /// Optimize the traced graph.
    void optimize();

    /// Compile the optimized graph into an execution plan.
    std::unique_ptr<ExecutionPlan> compile();

    /// Get the current graph operations (for debugging).
    const std::vector<GraphOp>& get_ops() const { return ops_; }

private:
    Config config_;
    std::vector<GraphOp> ops_;

    // Optimization passes
    void fuse_elementwise_chains();
    void fuse_layernorm_linear();
    void eliminate_redundant_copies();
    void reorder_for_locality();
    void preallocate_buffers();
};

}  // namespace optim
}  // namespace serius
