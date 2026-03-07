#pragma once

#include "../data/dataloader.h"  // Assuming it exists from Phase 4
#include "../optim/adamw.h"
#include "../optim/lr_schedule.h"
#include "../transformer/model.h"


namespace serius {
namespace training {

struct TrainerConfig
{
    int max_epochs = 10;
    int checkpoint_interval = 1000;  // steps
    float grad_clip = 1.0f;
    std::string checkpoint_dir = "checkpoints";
};

class Trainer
{
   public:
    Trainer(transformer::TransformerModel &model, optim::AdamW &optimizer, optim::WSDScheduler &scheduler,
            const TrainerConfig &config = TrainerConfig());

    /// Run the training loop
    void train(void *dataloader);  // Dataloader type placeholder as I need to check its API

   private:
    transformer::TransformerModel &model_;
    optim::AdamW &optimizer_;
    optim::WSDScheduler &scheduler_;
    TrainerConfig config_;
    int current_step_ = 0;
};

}  // namespace training
}  // namespace serius
