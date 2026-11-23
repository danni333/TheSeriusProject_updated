#pragma once

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include "comm.h"
#include "context.h"

namespace serius {
namespace distributed {

/// Distributed launch configuration read from environment variables.
///
/// Recognized variables (all optional; absent => single-node):
///   SERIUS_WORLD_SIZE / WORLD_SIZE   total ranks (default 1)
///   SERIUS_RANK        / RANK        this rank (default 0)
///   SERIUS_LOCAL_RANK  / LOCAL_RANK  node-local rank (default = rank)
///   SERIUS_TP_SIZE                     tensor-parallel group size (default 1)
///   SERIUS_PP_SIZE                     pipeline-parallel group size (default 1)
///   SERIUS_COMM_BACKEND                auto | mpi | nccl (default auto)
///
/// world_size must equal tp_size * pp_size when TP/PP are requested.
/// TP and PP cannot be combined (tp_size>1 && pp_size>1) — the model
/// only supports one parallel axis at a time.
struct DistributedEnv
{
    int rank = 0;
    int world_size = 1;
    int local_rank = -1;  // -1 => same as rank
    int tp_size = 1;
    int pp_size = 1;
    std::string backend = "auto";

    bool is_distributed() const { return world_size > 1 || tp_size > 1 || pp_size > 1; }
};

/// Parse the environment. `getenv_fn` is injectable for tests
/// (pass nullptr to use std::getenv). Throws std::invalid_argument on
/// inconsistent values.
DistributedEnv distributed_env_from_getenv(std::function<const char *(const char *)> getenv_fn = nullptr);

/// Build a DistributedTopology for `env`. `tp_comm`/`pp_comm` are raw
/// pointers to communicators owned elsewhere (see DistributedContext);
/// pass nullptr for the axis that is not in use.
DistributedTopology topology_from_env(const DistributedEnv &env, Communicator *tp_comm, Communicator *pp_comm);

/// Owns the communicators and topology for a distributed launch.
/// The topology's raw Communicator* stay valid while the context lives —
/// keep the context alive for the whole process lifetime (e.g. store it
/// next to the model).
class DistributedContext
{
   public:
    /// Returns nullptr when the environment requests single-node
    /// (no SERIUS_*/RANK/WORLD_SIZE vars or all defaults). Throws on
    /// invalid config or when the requested backend is not compiled in.
    /// `argc`/`argv` are forwarded to MPI_Init when the MPI backend is used
    /// (may be nullptr — MPI implementations accept that).
    static std::unique_ptr<DistributedContext> bootstrap(int *argc = nullptr, char ***argv = nullptr);

    const DistributedTopology &topology() const { return topology_; }
    const DistributedEnv &env() const { return env_; }
    bool is_distributed() const { return world_comm_ != nullptr; }

    DistributedContext() = default;

   private:

    DistributedEnv env_;
    std::unique_ptr<Communicator> world_comm_;
    DistributedTopology topology_;
};

}  // namespace distributed
}  // namespace serius
