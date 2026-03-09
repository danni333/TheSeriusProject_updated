#pragma once

#include <string>
#include <vector>

#include "../optim/adamw.h"
#include "../transformer/model.h"

namespace serius {
namespace training {

/// Save model and optimizer state to a binary file
void save_checkpoint(const std::string &path, const model::TransformerModel &model, const optim::AdamW &optimizer,
                     int epoch, int step);

/// Load model and optimizer state from a binary file
void load_checkpoint(const std::string &path, model::TransformerModel &model, optim::AdamW &optimizer, int &epoch,
                     int &step);

/// Perform gradient clipping in-place on a list of parameters
void clip_grad_norm(const std::vector<core::autograd::Node *> &params, float max_norm);

}  // namespace training
}  // namespace serius
