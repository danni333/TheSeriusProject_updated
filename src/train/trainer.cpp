#include "trainer.h"

#include <iomanip>
#include <iostream>

#include "../autograd/engine.h"
#include "../autograd/ops.h"
#include "checkpoint.h"

namespace serius {
namespace training {

Trainer::Trainer(model::TransformerModel &model, optim::AdamW &optimizer, optim::WSDScheduler &scheduler,
                 const TrainerConfig &config)
    : model_(model), optimizer_(optimizer), scheduler_(scheduler), config_(config)
{
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

            // Forward pass
            core::autograd::Node *logits = model_.forward(input_node);

            // Cross entropy handles (B, S, V) and (B, S) natively.
            core::autograd::Node *loss_node = core::autograd::ops::cross_entropy(logits, target_node);

            // 4. Backward pass
            core::autograd::backward(loss_node);

            // 5. Gradient clipping
            clip_grad_norm(model_.parameters(), config_.grad_clip);

            // 6. Update learning rate
            float current_lr = scheduler_.get_lr(current_step_);
            optimizer_.set_lr(current_lr);

            // 7. Optimizer step
            optimizer_.step();

            // 8. Logging
            const int LOG_INTERVAL = 10;
            if (current_step_ % LOG_INTERVAL == 0) {
                std::cout << "Step: " << current_step_ << " | Loss: " << std::fixed << std::setprecision(4)
                          << loss_node->data[0] << " | LR: " << std::scientific << current_lr << "\n";
            }

            // 9. Checkpointing
            if (current_step_ > 0 && current_step_ % config_.checkpoint_interval == 0) {
                std::string path = config_.checkpoint_dir + "/model_step_" + std::to_string(current_step_) + ".ckpt";
                save_checkpoint(path, model_, optimizer_, epoch, current_step_);
                std::cout << "Saved checkpoint to " << path << "\n";
            }

            current_step_++;

            // Cleanup nodes (in a real engine, this might be managed better)
            // Note: need to be careful with Node pointers if they are deleted or not.
            // For now, we'll assume they need manual cleanup if not managed.
            // ... cleanup logic (topological sort delete)
        }
    }
}

}  // namespace training
}  // namespace serius
