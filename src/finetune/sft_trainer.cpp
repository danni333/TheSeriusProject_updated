#include "sft_trainer.h"

#include <iostream>
#include <stdexcept>

#include "../autograd/engine.h"
#include "../autograd/ops.h"
#include "../stability/grad_clip.h"
#include "../train/checkpoint.h"

namespace serius {
namespace finetune {

SFTTrainer::SFTTrainer(model::TransformerModel &model,
                       optim::AdamW &optimizer,
                       const SFTTrainerConfig &config)
    : model_(model), optimizer_(optimizer), config_(config)
{
    if (config_.learning_rate > 0.0f) {
        optimizer_.set_lr(config_.learning_rate);
    }
    refresh_model_params_set();
}

void SFTTrainer::refresh_model_params_set()
{
    model_params_set_.clear();
    const auto params = model_.parameters();
    model_params_set_.insert(params.begin(), params.end());
}

void SFTTrainer::cleanup_graph(core::autograd::Node *loss_node,
                               core::autograd::Node *input_node,
                               core::autograd::Node *target_node) const
{
    if (!loss_node) return;
    std::vector<core::autograd::Node *> order = core::autograd::topological_sort(loss_node);
    std::unordered_set<core::autograd::Node *> to_delete(order.begin(), order.end());
    if (input_node) to_delete.insert(input_node);
    if (target_node) to_delete.insert(target_node);

    for (core::autograd::Node *node : to_delete) {
        if (node && model_params_set_.find(node) == model_params_set_.end()) {
            delete node;
        }
    }
}

float SFTTrainer::train_step(const SFTSample &sample)
{
    if (sample.input_ids.empty() || sample.target_ids.empty()) {
        throw std::invalid_argument("SFTTrainer::train_step: sample input_ids and target_ids must not be empty.");
    }
    if (sample.input_ids.size() != sample.target_ids.size() || sample.input_ids.size() != sample.loss_mask.size()) {
        throw std::invalid_argument("SFTTrainer::train_step: sample lengths mismatch.");
    }

    if (accum_step_ == 0) {
        optimizer_.zero_grad();
    }

    const size_t seq_len = sample.input_ids.size();

    // 1. Prepare input tensor node
    core::Tensor input_tensor({seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        input_tensor.data()[i] = static_cast<float>(sample.input_ids[i]);
    }
    auto *input_node = new core::autograd::Node(std::move(input_tensor), false);
    input_node->name = "sft_input_ids";

    // 2. Prepare target tensor node
    core::Tensor target_tensor({seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        target_tensor.data()[i] = static_cast<float>(sample.target_ids[i]);
    }
    auto *target_node = new core::autograd::Node(std::move(target_tensor), false);
    target_node->name = "sft_target_ids";

    // 3. Prepare loss mask tensor
    core::Tensor mask_tensor({seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        mask_tensor.data()[i] = sample.loss_mask[i] ? 1.0f : 0.0f;
    }

    // 4. Forward pass
    core::autograd::Node *logits = model_.forward(input_node);

    // 5. Masked cross-entropy (computed strictly over response tokens)
    core::autograd::Node *loss_node = core::autograd::ops::masked_cross_entropy(logits, target_node, mask_tensor);
    const float step_loss = loss_node->data[0];

    // 6. Backward pass
    core::autograd::backward(loss_node);

    // 7. Gradient accumulation & optimizer step
    accum_step_++;
    const size_t accum_target = config_.gradient_accumulation_steps > 0 ? config_.gradient_accumulation_steps : 1;
    if (accum_step_ >= accum_target) {
        if (!stability::has_nan_or_inf(model_.parameters())) {
            if (config_.grad_clip > 0.0f) {
                stability::clip_grad_norm(model_.parameters(), config_.grad_clip);
            }
            optimizer_.step();
        } else {
            optimizer_.zero_grad();
        }
        accum_step_ = 0;
    }

    // 8. Cleanup dynamic computation graph nodes
    cleanup_graph(loss_node, input_node, target_node);

    current_step_++;
    loss_history_.push_back(step_loss);

    if (config_.verbose && (current_step_ % config_.log_interval == 0 || current_step_ == 1)) {
        std::cout << "[SFT Step " << current_step_ << "] Loss: " << step_loss << "\n";
    }

    // 9. Optional checkpointing
    if (!config_.checkpoint_dir.empty() && config_.checkpoint_interval > 0 &&
        (current_step_ % config_.checkpoint_interval == 0)) {
        std::string ckpt_path = config_.checkpoint_dir + "/sft_step_" + std::to_string(current_step_) + ".ckpt";
        training::save_checkpoint(ckpt_path, model_, optimizer_, 0, static_cast<int>(current_step_));
    }

    return step_loss;
}

std::vector<float> SFTTrainer::train(const std::vector<SFTSample> &dataset)
{
    if (dataset.empty()) return {};

    std::vector<float> run_losses;
    for (size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        for (const auto &sample : dataset) {
            float loss = train_step(sample);
            run_losses.push_back(loss);
            if (config_.max_steps > 0 && current_step_ >= config_.max_steps) {
                return run_losses;
            }
        }
    }
    return run_losses;
}

std::vector<float> SFTTrainer::train(const std::vector<SFTExample> &examples, const SFTDataLoader &loader)
{
    std::vector<SFTSample> dataset;
    dataset.reserve(examples.size());
    for (const auto &ex : examples) {
        dataset.push_back(loader.prepare(ex));
    }
    return train(dataset);
}

float SFTTrainer::evaluate(const std::vector<SFTSample> &val_dataset) const
{
    if (val_dataset.empty()) {
        return 0.0f;
    }

    float total_loss = 0.0f;
    size_t count = 0;

    for (const auto &sample : val_dataset) {
        if (sample.input_ids.empty() || sample.target_ids.empty()) continue;
        const size_t seq_len = sample.input_ids.size();

        core::Tensor input_tensor({seq_len});
        for (size_t i = 0; i < seq_len; ++i) {
            input_tensor.data()[i] = static_cast<float>(sample.input_ids[i]);
        }
        auto *input_node = new core::autograd::Node(std::move(input_tensor), false);

        core::Tensor target_tensor({seq_len});
        for (size_t i = 0; i < seq_len; ++i) {
            target_tensor.data()[i] = static_cast<float>(sample.target_ids[i]);
        }
        auto *target_node = new core::autograd::Node(std::move(target_tensor), false);

        core::Tensor mask_tensor({seq_len});
        for (size_t i = 0; i < seq_len; ++i) {
            mask_tensor.data()[i] = sample.loss_mask[i] ? 1.0f : 0.0f;
        }

        core::autograd::Node *logits = model_.forward(input_node);
        core::autograd::Node *loss_node = core::autograd::ops::masked_cross_entropy(logits, target_node, mask_tensor);
        total_loss += loss_node->data[0];
        count++;

        cleanup_graph(loss_node, input_node, target_node);
    }

    return count > 0 ? (total_loss / static_cast<float>(count)) : 0.0f;
}

void SFTTrainer::freeze_parameters(size_t count)
{
    auto params = model_.parameters();
    for (size_t i = 0; i < count && i < params.size(); ++i) {
        params[i]->requires_grad = false;
    }
    refresh_model_params_set();
}

void SFTTrainer::freeze_embedding()
{
    if (model_.embedding) {
        model_.embedding->requires_grad = false;
    }
    refresh_model_params_set();
}

void SFTTrainer::unfreeze_all()
{
    for (auto *p : model_.parameters()) {
        p->requires_grad = true;
    }
    refresh_model_params_set();
}

void SFTTrainer::set_learning_rate(float lr)
{
    config_.learning_rate = lr;
    optimizer_.set_lr(lr);
}

}  // namespace finetune
}  // namespace serius
