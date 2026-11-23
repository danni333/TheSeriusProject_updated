#include "bootstrap.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "local_comm.h"
#include "mpi_comm.h"

#ifdef SERIUS_USE_NCCL
#include "nccl_comm.h"
#endif

namespace serius {
namespace distributed {

namespace {

const char *env_get(const std::function<const char *(const char *)> &fn, const char *name)
{
    if (fn != nullptr) {
        return fn(name);
    }
#if defined(_MSC_VER)
    // getenv() is deprecated under /W4 — use the bounds-checked variant.
    static thread_local std::string buf;
    buf.resize(4096);
    size_t len = 0;
    if (::getenv_s(&len, buf.data(), buf.size(), name) != 0 || len == 0) {
        return nullptr;
    }
    buf.resize(len);
    return buf.c_str();
#else
    return std::getenv(name);
#endif
}

int env_int(const std::function<const char *(const char *)> &fn, const char *primary, const char *fallback, int dflt,
            const char *what)
{
    const char *v = env_get(fn, primary);
    if (v == nullptr && fallback != nullptr) {
        v = env_get(fn, fallback);
    }
    if (v == nullptr || *v == '\0') {
        return dflt;
    }
    char *end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || parsed < 0 || parsed > 0x7FFFFFFF) {
        throw std::invalid_argument(std::string("Invalid ") + what + " environment value: '" + v + "'");
    }
    return static_cast<int>(parsed);
}

std::string env_str(const std::function<const char *(const char *)> &fn, const char *name, const std::string &dflt)
{
    const char *v = env_get(fn, name);
    return (v != nullptr && *v != '\0') ? std::string(v) : dflt;
}

void validate_backend_name(const std::string &name)
{
    if (name != "auto" && name != "mpi" && name != "nccl" && name != "local") {
        throw std::invalid_argument("SERIUS_COMM_BACKEND must be one of: auto, mpi, nccl, local (got '" + name + "')");
    }
}

}  // namespace

DistributedEnv distributed_env_from_getenv(std::function<const char *(const char *)> getenv_fn)
{
    DistributedEnv env;
    env.rank = env_int(getenv_fn, "SERIUS_RANK", "RANK", 0, "rank");
    env.world_size = env_int(getenv_fn, "SERIUS_WORLD_SIZE", "WORLD_SIZE", 1, "world size");
    env.local_rank = env_int(getenv_fn, "SERIUS_LOCAL_RANK", "LOCAL_RANK", -1, "local rank");
    env.tp_size = env_int(getenv_fn, "SERIUS_TP_SIZE", nullptr, 1, "TP size");
    env.pp_size = env_int(getenv_fn, "SERIUS_PP_SIZE", nullptr, 1, "PP size");
    env.backend = env_str(getenv_fn, "SERIUS_COMM_BACKEND", "auto");
    validate_backend_name(env.backend);

    if (env.world_size < 1) {
        throw std::invalid_argument("SERIUS_WORLD_SIZE must be >= 1");
    }
    if (env.rank < 0 || env.rank >= env.world_size) {
        throw std::invalid_argument("SERIUS_RANK must be in [0, SERIUS_WORLD_SIZE)");
    }
    if (env.tp_size < 1 || env.pp_size < 1) {
        throw std::invalid_argument("SERIUS_TP_SIZE and SERIUS_PP_SIZE must be >= 1");
    }
    if (env.tp_size > 1 && env.pp_size > 1) {
        throw std::invalid_argument(
            "Combined TP+PP (tp_size>1 && pp_size>1) is not supported yet — set only one of "
            "SERIUS_TP_SIZE / SERIUS_PP_SIZE");
    }
    const int axis_product = env.tp_size * env.pp_size;
    if (axis_product != env.world_size) {
        throw std::invalid_argument(
            "SERIUS_WORLD_SIZE (" + std::to_string(env.world_size) + ") must equal "
            "SERIUS_TP_SIZE * SERIUS_PP_SIZE (" + std::to_string(axis_product) + ")");
    }
    return env;
}

DistributedTopology topology_from_env(const DistributedEnv &env, Communicator *tp_comm, Communicator *pp_comm)
{
    DistributedTopology topo;
    topo.tp_size = env.tp_size;
    topo.pp_size = env.pp_size;
    topo.tp_comm = tp_comm;
    topo.pp_comm = pp_comm;
    if (env.pp_size > 1) {
        topo.pp_rank = env.rank;
        topo.tp_rank = 0;
    } else {
        topo.tp_rank = env.rank;
        topo.pp_rank = 0;
    }
    return topo;
}

std::unique_ptr<DistributedContext> DistributedContext::bootstrap(int *argc, char ***argv)
{
    DistributedEnv env = distributed_env_from_getenv();
    if (!env.is_distributed()) {
        return nullptr;
    }

    auto ctx = std::make_unique<DistributedContext>();
    ctx->env_ = env;

    const bool use_pp = env.pp_size > 1;
    const bool use_tp = env.tp_size > 1;

    // --- Select and create the world communicator ---
    auto create_mpi = [&]() -> std::unique_ptr<Communicator> {
#ifdef SERIUS_USE_MPI
        return std::make_unique<MPICommunicator>(argc, argv);
#else
        (void)argc;
        (void)argv;
        throw std::runtime_error("MPI backend requested but Serius was built without SERIUS_USE_MPI");
#endif
    };
    auto create_nccl = [&]() -> std::unique_ptr<Communicator> {
#ifdef SERIUS_USE_NCCL
#ifdef SERIUS_USE_MPI
        return Communicator::create_nccl(env.world_size, env.rank);
#else
        throw std::runtime_error(
            "NCCL bootstrap requires SERIUS_USE_MPI for ncclUniqueId distribution "
            "(or pass a shared ncclUniqueId manually)");
#endif
#else
        throw std::runtime_error("NCCL backend requested but Serius was built without SERIUS_USE_NCCL");
#endif
    };

    std::unique_ptr<Communicator> world;
    if (env.backend == "auto") {
        // Prefer MPI (inter-node capable), then NCCL (needs MPI anyway for
        // ID bootstrap). Fail honestly when nothing is compiled in.
#ifdef SERIUS_USE_MPI
        world = create_mpi();
#elif defined(SERIUS_USE_NCCL)
        world = create_nccl();
#else
        throw std::runtime_error(
            "Distributed launch requested (world_size=" + std::to_string(env.world_size) +
            ") but no communicator backend is available — rebuild with "
            "-DSERIUS_USE_MPI=ON or -DSERIUS_USE_NCCL=ON (NCCL also needs MPI for bootstrap)");
#endif
    } else if (env.backend == "mpi") {
        world = create_mpi();
    } else if (env.backend == "nccl") {
        world = create_nccl();
    } else {  // "local"
        throw std::runtime_error(
            "SERIUS_COMM_BACKEND=local cannot span processes — use mpi or nccl "
            "(LocalCommunicator is in-process simulation only)");
    }
    ctx->world_comm_ = std::move(world);

    // Sanity: the communicator's own view of the world must match the env.
    if (ctx->world_comm_->world_size() != env.world_size) {
        throw std::runtime_error(
            "Communicator world_size (" + std::to_string(ctx->world_comm_->world_size()) +
            ") does not match SERIUS_WORLD_SIZE (" + std::to_string(env.world_size) + ")");
    }
    if (ctx->world_comm_->rank() != env.rank) {
        throw std::runtime_error(
            "Communicator rank (" + std::to_string(ctx->world_comm_->rank()) +
            ") does not match SERIUS_RANK (" + std::to_string(env.rank) + ")");
    }

    Communicator *comm = ctx->world_comm_.get();
    if (use_pp) {
        ctx->topology_ = topology_from_env(env, nullptr, comm);
    } else if (use_tp) {
        ctx->topology_ = topology_from_env(env, comm, nullptr);
    } else {
        // world_size>1 with tp=pp=1 is rejected by env validation
        // (world must equal tp*pp), so this is unreachable.
        ctx->topology_ = topology_from_env(env, nullptr, nullptr);
    }
    return ctx;
}

}  // namespace distributed
}  // namespace serius
