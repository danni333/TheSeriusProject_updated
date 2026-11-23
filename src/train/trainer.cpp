#include "trainer.h"

#include <iomanip>
#include <iostream>
#include <optional>

#include "../autograd/engine.h"
#include "../autograd/ops.h"
#include "../stability/grad_clip.h"
#include "checkpoint.h"

namespace serius {
namespace training {

Trainer::Trainer(model::TransformerModel &model, optim::AdamW &optimizer, optim::WSDScheduler &scheduler,
                 const TrainerConfig &config)
    : model_(model), optimizer_(optimizer), scheduler_(scheduler), config_(config)
{
    const auto params = model_.parameters();
    model_params_set_.insert(params.begin(), params.end());
}

void Trainer::cleanup_graph(core::autograd::Node *loss_node) const
{
    if (!loss_node) return;
    std::vector<core::autograd::Node *> order = core::autograd::topological_sort(loss_node);
    for (core::autograd::Node *node : order) {
        if (node && model_params_set_.find(node) == model_params_set_.end()) {
            delete node;
        }
    }
}

void Trainer::train(DataLoader &dataloader)
{
    std::cout << "Starting training...\n";

    for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
        while (auto batch_opt = dataloader.next_batch()) {
            auto &batch = batch_opt.value();

            // 1. Zero gradients
            optimizer_.zero_grad();

            // 2. Prepare inputs as autograd nodes
            core::autograd::Node *input_node = new core::autograd::Node(std::move(batch.input_ids), false);
            input_node->name = "input_ids";

            core::autograd::Node *target_node = new core::autograd::Node(std::move(batch.target_ids), false);
            target_node->name = "target_ids";

            // 3. Forward pass
            core::autograd::Node *logits = model_.forward(input_node);
            core::autograd::Node *loss_node = core::autograd::ops::cross_entropy(logits, target_node);

            // 4. Backward pass (with AMP scaling if enabled)
            core::autograd::Node *scaled_loss = loss_node;
            if (config_.use_amp) {
                scaled_loss = amp_.scale(loss_node);
            }

            core::autograd::backward(scaled_loss);

            // 5. AMP Unscale and Update
            bool should_step = true;
            if (config_.use_amp) {
                should_step = amp_.unscale_and_update(model_.parameters());
            }

            float current_lr = scheduler_.get_lr(current_step_);

            if (should_step) {
                // 6. Gradient clipping + optimizer step
                stability::clip_grad_norm(model_.parameters(), config_.grad_clip);
                optimizer_.set_lr(current_lr);
                optimizer_.step();
            } else {
                std::cout << "Warning: Gradient overflow detected at step " << current_step_
                          << ". Skipping optimizer step.\n";
            }

            // 7. Logging
            const int LOG_INTERVAL = 10;
            if (current_step_ % LOG_INTERVAL == 0) {
                std::cout << "Step: " << current_step_ << " | Loss: " << std::fixed << std::setprecision(4)
                          << loss_node->data[0] << " | LR: " << std::scientific << current_lr << "\n";
            }

            // 8. Checkpointing
            if (current_step_ > 0 && current_step_ % config_.checkpoint_interval == 0) {
                std::string path = config_.checkpoint_dir + "/model_step_" + std::to_string(current_step_) + ".ckpt";
                save_checkpoint(path, model_, optimizer_, epoch, current_step_);
                std::cout << "Saved checkpoint to " << path << "\n";
            }

            // 9. Free the forward graph (parameters are kept: they are shared
            // across steps and owned by the model). cross_entropy does not
            // register target_node as a graph child, so it is unreachable
            // from the sink and must be freed explicitly.
            cleanup_graph(scaled_loss);
            delete target_node;

            current_step_++;
        }
    }
}

void Trainer::freeze_parameters(size_t count)
{
    std::vector<core::autograd::Node *> params = model_.parameters();
    for (size_t i = 0; i < count && i < params.size(); ++i) {
        params[i]->requires_grad = false;
    }
}

void Trainer::set_parameters_requires_grad(bool value)
{
    for (core::autograd::Node *p : model_.parameters()) {
        p->requires_grad = value;
    }
}

}  // namespace training
}  // namespace serius
