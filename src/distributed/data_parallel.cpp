#include "data_parallel.h"

namespace serius {
namespace distributed {

DataParallel::DataParallel(Communicator *comm) : comm_(comm)
{
}

void DataParallel::sync_gradients(const std::vector<std::pair<float *, size_t>> &grad_buffers)
{
    for (const auto &[buffer, count] : grad_buffers) {
        comm_->all_reduce(buffer, count, ReduceOp::Avg);
    }
}

void DataParallel::sync_gradient(float *buffer, size_t count)
{
    comm_->all_reduce(buffer, count, ReduceOp::Avg);
}

void DataParallel::broadcast_params(const std::vector<std::pair<float *, size_t>> &param_buffers)
{
    for (const auto &[buffer, count] : param_buffers) {
        comm_->broadcast(buffer, count, /*root=*/0);
    }
}

}  // namespace distributed
}  // namespace serius
