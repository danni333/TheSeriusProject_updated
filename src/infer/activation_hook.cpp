#include "activation_hook.h"

#include <atomic>
#include <mutex>

namespace serius {
namespace infer {

// Process-global: attention rows are emitted from OpenMP worker threads,
// so per-thread storage would silently drop captures. Consumers that
// record data must synchronize (see experimental/interpret/inference_capture).
// A 1-byte atomic flag gives a lock-free fast path when no hooks are active
// (avoids 16-byte std::atomic<ActivationHooks>, which requires -latomic on GCC),
// while the mutex serializes updates and non-empty snapshot copies.
namespace {
std::mutex g_hooks_mu;
ActivationHooks g_hooks{};
std::atomic<bool> g_hooks_active{false};
}  // namespace

void set_activation_hooks(const ActivationHooks &hooks)
{
    std::lock_guard<std::mutex> lk(g_hooks_mu);
    g_hooks = hooks;
    g_hooks_active.store(hooks.layer_hidden != nullptr || hooks.attention_row != nullptr,
                         std::memory_order_release);
}

void clear_activation_hooks()
{
    set_activation_hooks(ActivationHooks{});
}

bool activation_hooks_enabled()
{
    return g_hooks_active.load(std::memory_order_acquire);
}

ActivationHooks activation_hooks()
{
    if (!g_hooks_active.load(std::memory_order_acquire)) {
        return ActivationHooks{};
    }
    std::lock_guard<std::mutex> lk(g_hooks_mu);
    return g_hooks;
}

}  // namespace infer
}  // namespace serius
