#include "graph_compiler.h"

#include "../autograd/engine.h"
#include "../autograd/ops.h"
#include "../hal/device.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

namespace serius {
namespace optim {

ExecutionPlan::ExecutionPlan()
    : workspace_(nullptr), workspace_size_(0)
{
}

ExecutionPlan::~ExecutionPlan()
{
    if (workspace_ != nullptr) {
        hal::get_backend(hal::DeviceType::CPU)->free(workspace_);
        workspace_ = nullptr;
    }
}

bool ExecutionPlan::is_executable() const
{
    for (const GraphOp& op : ops_) {
        if (!op.executor) { return false; }
    }
    return true;
}

void ExecutionPlan::execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const
{
    if (inputs.size() != input_slots_.size()) {
        throw std::runtime_error("ExecutionPlan::execute: expected " +
                                 std::to_string(input_slots_.size()) + " input tensor(s), got " +
                                 std::to_string(inputs.size()));
    }

    // values_[i] holds the tensor of traced node i. Leaf slots are filled
    // from `inputs` (as views when possible); op slots are computed in
    // topological order, which ops_ already follows.
    std::vector<Tensor> values(slot_count_);
    for (size_t i = 0; i < inputs.size(); ++i) {
        values[input_slots_[i]] = core::autograd::ops::detail::replay_tensor_ref(inputs[i]);
    }

    for (const GraphOp& op : ops_) {
        if (!op.executor) {
            throw std::runtime_error("ExecutionPlan::execute: op '" + op.type +
                                     "' has no replay function (traced node "
                                     "lacks forward_fn)");
        }
        if (op.outputs.size() != 1 || op.outputs[0] >= slot_count_) {
            throw std::runtime_error("ExecutionPlan::execute: malformed op '" + op.type + "'");
        }
        std::vector<const Tensor*> args;
        args.reserve(op.inputs.size());
        for (size_t idx : op.inputs) {
            if (idx >= slot_count_) {
                throw std::runtime_error("ExecutionPlan::execute: op '" + op.type +
                                         "' references an invalid input slot");
            }
            args.push_back(&values[idx]);
        }
        values[op.outputs[0]] = op.executor(args);
    }

    outputs.clear();
    outputs.reserve(output_slots_.size());
    for (size_t slot : output_slots_) {
        outputs.push_back(std::move(values[slot]));
    }
}

GraphCompiler::GraphCompiler(const Config& config)
    : config_(config)
{
}

GraphCompiler::GraphCompiler()
    : GraphCompiler(Config{})
{
}

GraphCompiler::~GraphCompiler()
{
}

void GraphCompiler::trace_graph(core::autograd::Node *output)
{
    ops_.clear();
    input_slots_.clear();
    output_slots_.clear();
    slot_count_ = 0;
    slot_shapes_.clear();
    peak_workspace_bytes_ = 0;
    if (output == nullptr) { return; }

    const auto order = core::autograd::topological_sort(output);
    slot_count_ = order.size();
    slot_shapes_.reserve(order.size());
    for (core::autograd::Node *node : order) {
        slot_shapes_.push_back(node->data.shape());
    }
    std::map<core::autograd::Node *, size_t> index;
    for (size_t i = 0; i < order.size(); ++i) { index[order[i]] = i; }

    // consumed[i] marks slots used as an operand; unconsumed slots are the
    // graph's sink nodes and become execute() outputs.
    std::vector<bool> consumed(slot_count_, false);
    for (core::autograd::Node *node : order) {
        if (node->children.empty()) {
            input_slots_.push_back(index[node]);  // leaf/input node
            continue;
        }
        GraphOp op;
        op.type = node->name.empty() ? "unknown" : node->name;
        op.executor = node->forward_fn;
        op.inputs.reserve(node->children.size());
        for (core::autograd::Node *child : node->children) {
            if (child == nullptr) { continue; }
            op.inputs.push_back(index[child]);
            consumed[index[child]] = true;
        }
        op.outputs = {index[node]};
        op.output_shape = node->data.shape();
        ops_.push_back(std::move(op));
    }

    for (size_t i = 0; i < slot_count_; ++i) {
        if (!consumed[i]) { output_slots_.push_back(i); }
    }
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

    if (config_.reorder_for_locality) {
        reorder_for_locality();
    }

    if (config_.preallocate_buffers) {
        preallocate_buffers();
    }
}

void GraphCompiler::fuse_elementwise_chains()
{
    // Fuse add(x, bias) -> gelu into a single fused_bias_gelu plan op.
    // Matches the broadcast-bias pattern of Tensor::operator+
    // ((..., D) + (D,)): the add's rank-1 operand whose size equals the last
    // dim of the add output is the bias. The fused executor composes the two
    // original executors, so op parameters stay captured inside them and no
    // numeric behavior changes (same math as ops::fused_bias_gelu).
    //
    // Safety conditions mirror fuse_layernorm_linear():
    //  - the add output is consumed exactly once, by the gelu;
    //  - the add output is not itself a graph output (output_slots_);
    //  - both ops have executors;
    //  - slot shapes are available (trace_graph() always fills them).
    if (slot_shapes_.size() != slot_count_) { return; }

    for (;;) {
        bool fused_any = false;
        for (size_t i = 0; i < ops_.size() && !fused_any; ++i) {
            GraphOp &gelu_op = ops_[i];
            if (gelu_op.type != "gelu" || gelu_op.inputs.size() != 1 ||
                gelu_op.outputs.size() != 1 || !gelu_op.executor) {
                continue;
            }
            const size_t add_out = gelu_op.inputs[0];
            if (add_out >= slot_count_ ||
                std::find(output_slots_.begin(), output_slots_.end(), add_out) !=
                    output_slots_.end()) {
                continue;
            }

            // Locate the unique producer op of the gelu input and count its
            // consumers (must be exactly the gelu).
            size_t add_idx = ops_.size();
            size_t uses = 0;
            for (size_t j = 0; j < ops_.size(); ++j) {
                if (ops_[j].outputs.size() == 1 &&
                    ops_[j].outputs[0] == add_out) {
                    add_idx = j;
                }
                for (size_t s : ops_[j].inputs) {
                    if (s == add_out) { ++uses; }
                }
            }
            if (uses != 1 || add_idx == ops_.size() || add_idx == i) { continue; }

            GraphOp &add_op = ops_[add_idx];
            if (add_op.type != "add" || add_op.inputs.size() != 2 ||
                !add_op.executor || add_op.output_shape.empty()) {
                continue;
            }

            // Exactly one add operand must be a rank-1 bias whose size equals
            // the last dim of the (broadcast) add output.
            const size_t last_dim = add_op.output_shape.back();
            int bias_pos = -1;
            for (int p = 0; p < 2; ++p) {
                const size_t s = add_op.inputs[p];
                if (s >= slot_shapes_.size()) { continue; }
                const auto &sh = slot_shapes_[s];
                if (sh.size() == 1 && sh[0] == last_dim) {
                    if (bias_pos >= 0) { bias_pos = -2; break; }  // ambiguous
                    bias_pos = p;
                }
            }
            if (bias_pos < 0) { continue; }

            GraphOp fused;
            fused.type = "fused_bias_gelu";
            fused.fused_kernel = fused.type;
            fused.inputs = add_op.inputs;
            fused.outputs = gelu_op.outputs;
            fused.output_shape = gelu_op.output_shape;

            auto add_exec = add_op.executor;
            auto gelu_exec = gelu_op.executor;
            fused.executor = [add_exec, gelu_exec](
                                 const std::vector<const Tensor *> &in) -> Tensor {
                Tensor mid = add_exec(in);
                const std::vector<const Tensor *> gargs = {&mid};
                return gelu_exec(gargs);
            };

            // Place the fused op at the gelu's (later) position, like
            // fuse_layernorm_linear(): the bias operand may be produced by an
            // op sitting between the add and the gelu.
            ops_[i] = std::move(fused);
            ops_.erase(ops_.begin() + add_idx);
            fused_any = true;
        }
        if (!fused_any) { break; }
    }
}

void GraphCompiler::fuse_layernorm_linear()
{
    // Fuse a normalization op (layernorm / rmsnorm / fused_add_layernorm)
    // feeding a matmul into a single plan op. The fused executor composes the
    // two original executors, so op parameters (eps, dims, ...) stay captured
    // inside them and no numeric behavior changes.
    //
    // Safety conditions:
    //  - the norm output is consumed exactly once, by the matmul;
    //  - the norm output is not itself a graph output (output_slots_);
    //  - the matmul has exactly two inputs and both ops have executors.
    static const std::set<std::string> kNormTypes = {
        "layernorm", "rmsnorm", "fused_add_layernorm"};

    for (size_t i = 0; i < ops_.size(); ++i) {
        GraphOp &norm_op = ops_[i];
        if (!kNormTypes.count(norm_op.type) || norm_op.outputs.size() != 1 ||
            !norm_op.executor) {
            continue;
        }
        const size_t norm_slot = norm_op.outputs[0];
        if (norm_slot >= slot_count_) { continue; }
        if (std::find(output_slots_.begin(), output_slots_.end(), norm_slot) !=
            output_slots_.end()) {
            continue;
        }

        // Count consumers of the norm output; find the unique consumer op.
        size_t uses = 0;
        size_t consumer = ops_.size();
        for (size_t j = 0; j < ops_.size(); ++j) {
            for (size_t s : ops_[j].inputs) {
                if (s == norm_slot) {
                    ++uses;
                    consumer = j;
                }
            }
        }
        if (uses != 1 || consumer == ops_.size() || consumer == i) { continue; }

        GraphOp &mm_op = ops_[consumer];
        if (mm_op.type != "matmul" || mm_op.inputs.size() != 2 || !mm_op.executor) {
            continue;
        }
        const size_t mm_norm_pos = (mm_op.inputs[0] == norm_slot) ? 0 : 1;
        const size_t norm_arity = norm_op.inputs.size();

        GraphOp fused;
        fused.type = "fused_" + norm_op.type + "_linear";
        fused.fused_kernel = fused.type;
        fused.inputs = norm_op.inputs;
        fused.inputs.push_back(mm_op.inputs[1 - mm_norm_pos]);
        fused.outputs = mm_op.outputs;
        fused.output_shape = mm_op.output_shape;

        auto norm_exec = norm_op.executor;
        auto mm_exec = mm_op.executor;
        fused.executor = [norm_exec, mm_exec, norm_arity, mm_norm_pos](
                             const std::vector<const Tensor *> &in) -> Tensor {
            std::vector<const Tensor *> norm_args(in.begin(),
                                                  in.begin() + norm_arity);
            Tensor mid = norm_exec(norm_args);
            const Tensor *other = in[norm_arity];
            std::vector<const Tensor *> mm_args =
                mm_norm_pos == 0
                    ? std::vector<const Tensor *>{&mid, other}
                    : std::vector<const Tensor *>{other, &mid};
            return mm_exec(mm_args);
        };

        // Place the fused op at the matmul's (later) position: its second
        // input may be produced by an op between the norm and the matmul.
        ops_[consumer] = std::move(fused);
        ops_.erase(ops_.begin() + i);
        break;
    }
}

void GraphCompiler::eliminate_redundant_copies()
{
    // Remove contiguous ops that are unnecessary. Consumers are rewired to
    // the contiguous's input slot so no slot is left without a producer;
    // contiguous ops whose output is a graph output are kept.
    // contiguous nodes are named "contiguous(<child>)" by the autograd op.
    auto is_contiguous_op = [](const GraphOp& op) {
        return op.type == "contiguous" ||
               op.type.rfind("contiguous(", 0) == 0;
    };
    std::vector<GraphOp> optimized;
    optimized.reserve(ops_.size());
    for (const auto& op : ops_) {
        if (is_contiguous_op(op) && op.inputs.size() == 1 &&
            op.outputs.size() == 1 &&
            std::find(output_slots_.begin(), output_slots_.end(),
                      op.outputs[0]) == output_slots_.end()) {
            for (GraphOp& other : ops_) {
                for (size_t& s : other.inputs) {
                    if (s == op.outputs[0]) { s = op.inputs[0]; }
                }
            }
            continue;
        }
        optimized.push_back(op);
    }
    ops_ = std::move(optimized);
}

void GraphCompiler::reorder_for_locality()
{
    // Greedy list-scheduling for cache locality: repeatedly emit the ready op
    // whose inputs were produced most recently, so producer->consumer chains
    // stay adjacent in ops_ order. Kahn's algorithm guarantees the result is
    // still a valid topological order; execute() resolves operands by slot
    // index, so any topological order is correct.
    const size_t n = ops_.size();
    if (n < 2 || slot_count_ == 0) { return; }

    std::vector<int> producer(slot_count_, -1);
    for (size_t i = 0; i < n; ++i) {
        for (size_t s : ops_[i].outputs) {
            if (s < slot_count_) { producer[s] = static_cast<int>(i); }
        }
    }

    std::vector<std::vector<int>> consumers(n);
    std::vector<int> remaining(n, 0);
    std::vector<int> score(n, -1);  // most recent producer index among inputs
    for (size_t j = 0; j < n; ++j) {
        std::set<int> deps;
        for (size_t s : ops_[j].inputs) {
            if (s >= slot_count_) { continue; }
            const int p = producer[s];
            if (p >= 0 && static_cast<size_t>(p) != j) {
                deps.insert(p);
                score[j] = std::max(score[j], p);
            }
        }
        remaining[j] = static_cast<int>(deps.size());
        for (int p : deps) { consumers[p].push_back(static_cast<int>(j)); }
    }

    // Max-heap on (score, -index): highest score first, ties -> lower index.
    std::priority_queue<std::pair<int, int>> ready;
    for (size_t i = 0; i < n; ++i) {
        if (remaining[i] == 0) {
            ready.emplace(score[i], -static_cast<int>(i));
        }
    }

    std::vector<GraphOp> reordered;
    reordered.reserve(n);
    while (!ready.empty()) {
        const auto [sc, neg_i] = ready.top();
        ready.pop();
        const size_t i = static_cast<size_t>(-neg_i);
        (void)sc;
        reordered.push_back(std::move(ops_[i]));
        for (int c : consumers[i]) {
            if (--remaining[c] == 0) {
                ready.emplace(score[c], -c);
            }
        }
    }

    if (reordered.size() == n) {
        ops_ = std::move(reordered);
    }
    // On a cycle (should not happen for traced graphs) keep the original order.
}

void GraphCompiler::preallocate_buffers()
{
    // Liveness analysis over the slot buffer used by ExecutionPlan::execute:
    // a slot is live from the op that produces it (inputs are live from the
    // start) until its last consumer, or until the end for graph outputs.
    // peak_workspace_bytes_ is the maximum simultaneously-live tensor bytes;
    // compile() uses it instead of the conservative sum-of-all-outputs.
    peak_workspace_bytes_ = 0;
    if (ops_.empty() || slot_shapes_.size() != slot_count_) { return; }

    const size_t n_ops = ops_.size();
    const size_t n_slots = slot_count_;

    std::vector<int> producer(n_slots, -1);
    for (size_t i = 0; i < n_ops; ++i) {
        for (size_t s : ops_[i].outputs) {
            if (s < n_slots) { producer[s] = static_cast<int>(i); }
        }
    }

    std::vector<int> last_use(n_slots, -1);
    for (size_t i = 0; i < n_ops; ++i) {
        for (size_t s : ops_[i].inputs) {
            if (s < n_slots) { last_use[s] = static_cast<int>(i); }
        }
    }

    std::vector<char> is_output(n_slots, 0);
    for (size_t s : output_slots_) {
        if (s < n_slots) { is_output[s] = 1; }
    }

    std::vector<size_t> bytes(n_slots, 0);
    for (size_t s = 0; s < n_slots; ++s) {
        size_t numel = 1;
        for (size_t d : slot_shapes_[s]) { numel *= d; }
        bytes[s] = numel * sizeof(float);
    }

    size_t peak = 0;
    for (size_t t = 0; t <= n_ops; ++t) {
        size_t live = 0;
        for (size_t s = 0; s < n_slots; ++s) {
            if (bytes[s] == 0) { continue; }
            const int avail = producer[s] >= 0 ? producer[s] + 1 : 0;
            const int death = is_output[s] ? static_cast<int>(n_ops) : last_use[s];
            if (static_cast<int>(t) >= avail && static_cast<int>(t) <= death) {
                live += bytes[s];
            }
        }
        peak = std::max(peak, live);
    }
    peak_workspace_bytes_ = peak;
}

std::unique_ptr<ExecutionPlan> GraphCompiler::compile()
{
    auto plan = std::make_unique<ExecutionPlan>();
    plan->ops_ = ops_;
    plan->input_slots_ = input_slots_;
    plan->output_slots_ = output_slots_;
    plan->slot_count_ = slot_count_;

    // Calculate workspace size
    size_t total_size = 0;
    for (const auto& op : ops_) {
        size_t op_size = 1;
        for (size_t dim : op.output_shape) {
            op_size *= dim;
        }
        total_size += op_size * sizeof(float);
    }
    // Use the liveness peak when preallocate_buffers() ran; otherwise fall
    // back to the conservative sum of all op outputs.
    plan->workspace_size_ = peak_workspace_bytes_ > 0 ? peak_workspace_bytes_
                                                    : total_size;

    // workspace_size_ stays as metadata (liveness peak or conservative sum).
    // The buffer itself is NOT allocated here: execute() never references
    // workspace_ — op outputs allocate via Tensor/hal backend — so an eager
    // malloc would reserve (potentially GBs of) memory that is never touched.
    // If a fused kernel later needs scratch it can allocate on demand.
    plan->workspace_ = nullptr;

    return plan;
}

}  // namespace optim
}  // namespace serius
