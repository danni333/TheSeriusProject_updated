#pragma once

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

#include "../autograd/node.h"
#include "../optim/adamw.h"
#include "../transformer/model.h"
#include "sft_loader.h"

namespace serius {
namespace finetune {

struct SFTTrainerConfig
{
    size_t epochs = 1;
    size_t max_steps = 0;              // 0 = run until end of epochs * dataset
    float learning_rate = 1e-4f;
    float grad_clip = 1.0f;
    size_t gradient_accumulation_steps = 1;
    size_t log_interval = 10;
    bool verbose = false;
    std::string checkpoint_dir = "";   // Empty = no checkpointing
    size_t checkpoint_interval = 0;    // 0 = disabled
};

class SFTTrainer
{
public:
    SFTTrainer(model::TransformerModel &model,
               optim::AdamW &optimizer,
               const SFTTrainerConfig &config = SFTTrainerConfig());
    ~SFTTrainer() = default;

    /// Execute a single supervised training step on a preprocessed SFTSample.
    /// Computes masked cross-entropy loss ONLY on response tokens.
    /// Returns the step loss.
    float train_step(const SFTSample &sample);

    /// Train over a dataset of SFTSamples for config.epochs.
    /// Returns vector of step loss values.
    std::vector<float> train(const std::vector<SFTSample> &dataset);

    /// Helper: formats SFTExamples with loader and trains over them.
    std::vector<float> train(const std::vector<SFTExample> &examples, const SFTDataLoader &loader);

    /// Evaluate model on validation samples without updating parameters.
    /// Returns the mean loss over masked response tokens.
    float evaluate(const std::vector<SFTSample> &val_dataset) const;

    /// Freeze the first `count` model parameters (e.g. embedding and first layers).
    void freeze_parameters(size_t count);

    /// Freeze only the embedding layer weights.
    void freeze_embedding();

    /// Unfreeze all model parameters (set requires_grad = true).
    void unfreeze_all();

    /// Dynamically update optimizer learning rate.
    void set_learning_rate(float lr);

    /// Inspection accessors
    size_t current_step() const noexcept { return current_step_; }
    const std::vector<float> &loss_history() const noexcept { return loss_history_; }
    const SFTTrainerConfig &config() const noexcept { return config_; }

private:
    model::TransformerModel &model_;
    optim::AdamW &optimizer_;
    SFTTrainerConfig config_;
    size_t current_step_ = 0;
    size_t accum_step_ = 0;
    std::vector<float> loss_history_;
    std::unordered_set<core::autograd::Node *> model_params_set_;

    void refresh_model_params_set();
    void cleanup_graph(core::autograd::Node *loss_node,
                       core::autograd::Node *input_node,
                       core::autograd::Node *target_node) const;
};

}  // namespace finetune
}  // namespace serius
