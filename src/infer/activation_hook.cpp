#include "activation_hook.h"

namespace serius {
namespace infer {

// Process-global: attention rows are emitted from OpenMP worker threads,
// so per-thread storage would silently drop captures. Consumers that
// record data must synchronize (see experimental/interpret/inference_capture).
namespace {
ActivationHooks g_hooks{};
}

void set_activation_hooks(const ActivationHooks &hooks)
{
    g_hooks = hooks;
}

void clear_activation_hooks()
{
    g_hooks = ActivationHooks{};
}

bool activation_hooks_enabled()
{
    return g_hooks.layer_hidden != nullptr || g_hooks.attention_row != nullptr;
}

const ActivationHooks &activation_hooks()
{
    return g_hooks;
}

}  // namespace infer
}  // namespace serius
