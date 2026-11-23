#include "activation_hook.h"

#include <atomic>

namespace serius {
namespace infer {

// Process-global: attention rows are emitted from OpenMP worker threads,
// so per-thread storage would silently drop captures. Consumers that
// record data must synchronize (see experimental/interpret/inference_capture).
// The hooks struct is read from OpenMP workers while the main thread may
// install/clear it, so accesses go through an atomic snapshot.
namespace {
std::atomic<ActivationHooks> g_hooks{ActivationHooks{}};
}

void set_activation_hooks(const ActivationHooks &hooks)
{
    g_hooks.store(hooks, std::memory_order_release);
}

void clear_activation_hooks()
{
    g_hooks.store(ActivationHooks{}, std::memory_order_release);
}

bool activation_hooks_enabled()
{
    const ActivationHooks h = g_hooks.load(std::memory_order_acquire);
    return h.layer_hidden != nullptr || h.attention_row != nullptr;
}

ActivationHooks activation_hooks()
{
    return g_hooks.load(std::memory_order_acquire);
}

}  // namespace infer
}  // namespace serius
