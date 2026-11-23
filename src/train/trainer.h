#pragma once

#include <unordered_set>

#include "../autograd/node.h"
#include "../data/dataloader.h"
#include "../optim/adamw.h"
#include "../optim/amp.h"
#include "../optim/lr_schedule.h"
#include "../transformer/model.h"


namespace serius {
namespace training {

struct TrainerConfig
{
    int max_epochs = 10;
    int checkpoint_interval = 1000;  // steps
    float grad_clip = 1.0F;
    bool use_amp = false;
    std::string checkpoint_dir = "checkpoints";
};

class Trainer
{
   public:
    Trainer(model::TransformerModel &model, optim::AdamW &optimizer, optim::WSDScheduler &scheduler,
            const TrainerConfig &config = TrainerConfig());

    /// Run the training loop
    void train(DataLoader &dataloader);

    /// Freeze the first \p count parameters (e.g. for finetuning: freeze embedding and early layers).
    void freeze_parameters(size_t count);

    /// Set requires_grad for all model parameters.
    void set_parameters_requires_grad(bool value);

   private:
    /// Delete every node reachable from \p loss_node except model parameters
    /// (which are shared across steps). Prevents leaking the whole forward
    /// graph on every training step.
    void cleanup_graph(core::autograd::Node *loss_node) const;

    model::TransformerModel &model_;
    std::unordered_set<core::autograd::Node *> model_params_set_;
    optim::AdamW &optimizer_;
    optim::WSDScheduler &scheduler_;
    optim::AutomaticMixedPrecision amp_;
    TrainerConfig config_;
    int current_step_ = 0;
};

}  // namespace training
}  // namespace serius
