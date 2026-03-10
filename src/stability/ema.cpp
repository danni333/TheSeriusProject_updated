#include "ema.h"

#include <stdexcept>

namespace serius {
namespace stability {

EMAWeights::EMAWeights(float decay) : decay_(decay) {}

void EMAWeights::init(const std::vector<core::autograd::Node *> &model_params)
{
    shadow_params_.clear();
    shadow_params_.reserve(model_params.size());
    for (const auto *p : model_params) {
        shadow_params_.push_back(p->data.copy());
    }
}

void EMAWeights::update(const std::vector<core::autograd::Node *> &model_params)
{
    if (shadow_params_.size() != model_params.size()) {
        throw std::runtime_error("EMAWeights::update: parameter count mismatch. Call init() first.");
    }

    for (size_t i = 0; i < shadow_params_.size(); i++) {
        float *s = shadow_params_[i].data();
        const float *p = model_params[i]->data.data();
        size_t n = shadow_params_[i].size();
        for (size_t j = 0; j < n; j++) {
            s[j] = decay_ * s[j] + (1.0f - decay_) * p[j];
        }
    }
}

void EMAWeights::apply(std::vector<core::autograd::Node *> &model_params)
{
    if (shadow_params_.size() != model_params.size()) {
        throw std::runtime_error("EMAWeights::apply: parameter count mismatch.");
    }
    if (applied_) {
        throw std::runtime_error("EMAWeights::apply: already applied. Call restore() first.");
    }

    // Save current weights
    backup_params_.clear();
    backup_params_.reserve(model_params.size());
    for (const auto *p : model_params) {
        backup_params_.push_back(p->data.copy());
    }

    // Copy shadow -> model
    for (size_t i = 0; i < model_params.size(); i++) {
        float *dst = model_params[i]->data.data();
        const float *src = shadow_params_[i].data();
        size_t n = model_params[i]->data.size();
        for (size_t j = 0; j < n; j++) {
            dst[j] = src[j];
        }
    }

    applied_ = true;
}

void EMAWeights::restore(std::vector<core::autograd::Node *> &model_params)
{
    if (!applied_) {
        throw std::runtime_error("EMAWeights::restore: not applied. Call apply() first.");
    }
    if (backup_params_.size() != model_params.size()) {
        throw std::runtime_error("EMAWeights::restore: parameter count mismatch.");
    }

    // Restore original weights
    for (size_t i = 0; i < model_params.size(); i++) {
        float *dst = model_params[i]->data.data();
        const float *src = backup_params_[i].data();
        size_t n = model_params[i]->data.size();
        for (size_t j = 0; j < n; j++) {
            dst[j] = src[j];
        }
    }

    backup_params_.clear();
    applied_ = false;
}

}  // namespace stability
}  // namespace serius
