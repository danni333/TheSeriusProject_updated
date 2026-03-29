#pragma once

#include "comm.h"

namespace serius {
namespace distributed {

/// Holds the configuration and communicators for distributed training.
struct DistributedTopology {
    Communicator* tp_comm = nullptr;  // Tensor parallel communicator
    Communicator* pp_comm = nullptr;  // Pipeline parallel communicator
    
    int tp_rank = 0;
    int tp_size = 1;
    
    int pp_rank = 0;
    int pp_size = 1;

    /// Helper to get a default single-node topology
    static DistributedTopology single_node() {
        return DistributedTopology{};
    }

    bool is_tensor_parallel() const {
        return tp_size > 1 && tp_comm != nullptr;
    }

    bool is_pipeline_parallel() const {
        return pp_size > 1 && pp_comm != nullptr;
    }
};

} // namespace distributed
} // namespace serius
