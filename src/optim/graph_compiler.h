#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../autograd/node.h"
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
    /// Replays this op: maps operand tensors (in `inputs` order) to the op's
    /// output tensor. Copied from the traced node's Node::forward_fn; null for
    /// synthetic trace() ops and nodes whose op provides no replay —
    /// ExecutionPlan::execute throws on those.
    std::function<Tensor(const std::vector<const Tensor *> &)> executor;
};

/// Optimized execution plan for a model.
class ExecutionPlan
{
public:
    ExecutionPlan();
    ~ExecutionPlan();

    /// Execute the plan.
    /// Plans built from trace_graph() re-run the traced autograd ops: `inputs`
    /// must contain one tensor per leaf (operand) node, in topological order
    /// (see num_inputs()); `outputs` is cleared and filled with one tensor per
    /// sink node, in topological order (see num_outputs()). Ops without a
    /// replay function cause a std::runtime_error.
    /// @param inputs Leaf/operand tensors
    /// @param outputs Receives the sink node outputs
    void execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const;

    /// Number of leaf tensors execute() expects in `inputs`.
    size_t num_inputs() const { return input_slots_.size(); }

    /// Number of tensors execute() produces in `outputs`.
    size_t num_outputs() const { return output_slots_.size(); }

    /// True when every op carries a replay function, i.e. the plan came from
    /// trace_graph() (or an equivalent real trace) and execute() can run it.
    /// Synthetic trace() plans return false.
    bool is_executable() const;

    /// Get the number of operations in the plan.
    size_t num_ops() const { return ops_.size(); }

    /// Workspace bytes required by the plan: the liveness peak computed by
    /// GraphCompiler::preallocate_buffers(), or the conservative sum of all
    /// op outputs when that pass did not run.
    size_t workspace_size() const { return workspace_size_; }

private:
    std::vector<GraphOp> ops_;
    void* workspace_;  // Pre-allocated workspace memory
    size_t workspace_size_;
    std::vector<size_t> input_slots_;   // values index per execute() input
    std::vector<size_t> output_slots_;  // values index per execute() output
    size_t slot_count_ = 0;             // total traced node slots

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
        bool reorder_for_locality = true;
    };

    explicit GraphCompiler(const Config& config);
    GraphCompiler();
    ~GraphCompiler();

    /// Trace the graph by running a dummy forward pass.
    /// NOTE: `model` is unused — this emits a fixed synthetic op sequence
    /// (matmul → bias → gelu) purely so the optimizer passes can be exercised.
    /// For real graphs use trace_graph(Node*) instead.
    /// @param model Unused placeholder (kept for API compatibility)
    /// @param input_shape Shape of the input tensor
    void trace(void* model, const std::vector<size_t>& input_shape);

    /// Trace a real autograd graph: walks `output`'s ancestors in topological
    /// order and emits one GraphOp per computed node (op type = node name,
    /// inputs = child indices, output_shape = node data shape).
    /// Leaf/input nodes (no children) are indexed but emit no op.
    void trace_graph(core::autograd::Node *output);

    /// Optimize the traced graph.
    void optimize();

    /// Compile the optimized graph into an execution plan.
    /// For graphs traced with trace_graph(), the plan can re-execute the
    /// computation via ExecutionPlan::execute (ops lacking a replay function
    /// still throw). Plans from the synthetic trace() are descriptive only.
    std::unique_ptr<ExecutionPlan> compile();

    /// Get the current graph operations (for debugging).
    const std::vector<GraphOp>& get_ops() const { return ops_; }

private:
    Config config_;
    std::vector<GraphOp> ops_;
    // Slot mapping produced by trace_graph(): total node slots, leaf slots
    // (execute() inputs, in topological order) and sink slots (outputs).
    std::vector<size_t> input_slots_;
    std::vector<size_t> output_slots_;
    size_t slot_count_ = 0;
    // Per-slot shapes (indexed like the topological order) used by
    // preallocate_buffers() for liveness analysis; empty for synthetic traces.
    std::vector<std::vector<size_t>> slot_shapes_;
    // Peak live-tensor bytes computed by preallocate_buffers(); 0 = not run.
    size_t peak_workspace_bytes_ = 0;

    // Optimization passes
    void fuse_elementwise_chains();
    void fuse_layernorm_linear();
    void eliminate_redundant_copies();
    void reorder_for_locality();
    void preallocate_buffers();
};

}  // namespace optim
}  // namespace serius
