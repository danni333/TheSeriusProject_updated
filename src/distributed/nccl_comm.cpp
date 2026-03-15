#ifdef SERIUS_USE_NCCL

#include "nccl_comm.h"

#include <stdexcept>
#include <string>

namespace serius {
namespace distributed {

#define NCCL_CHECK(cmd)                                                               \
    do {                                                                              \
        ncclResult_t res = cmd;                                                       \
        if (res != ncclSuccess) {                                                     \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(res)); \
        }                                                                             \
    } while (0)

#define CUDA_CHECK(cmd)                                                              \
    do {                                                                             \
        cudaError_t err = cmd;                                                       \
        if (err != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        }                                                                            \
    } while (0)

NCCLCommunicator::NCCLCommunicator(int world_size, int rank, ncclUniqueId id)
    : rank_(rank), world_size_(world_size)
{
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size, id, rank));
}

NCCLCommunicator::~NCCLCommunicator()
{
    ncclCommDestroy(comm_);
    cudaStreamDestroy(stream_);
}

ncclRedOp_t NCCLCommunicator::to_nccl_op(ReduceOp op)
{
    switch (op) {
        case ReduceOp::Sum: return ncclSum;
        case ReduceOp::Avg: return ncclAvg;
        case ReduceOp::Min: return ncclMin;
        case ReduceOp::Max: return ncclMax;
    }
    return ncclSum;
}

void NCCLCommunicator::all_reduce(float *buffer, size_t count, ReduceOp op)
{
    NCCL_CHECK(ncclAllReduce(buffer, buffer, count, ncclFloat,
                             to_nccl_op(op), comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::all_reduce(core::Tensor &tensor, ReduceOp op)
{
    tensor.syncToDevice();
    all_reduce(tensor.data(), tensor.size(), op);
    // Data is already on device, mark it dirty so future host syncs are correct
    tensor.mark_dirty(hal::DeviceType::CUDA);
}

void NCCLCommunicator::broadcast(float *buffer, size_t count, int root)
{
    NCCL_CHECK(ncclBroadcast(buffer, buffer, count, ncclFloat, root, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::broadcast(core::Tensor &tensor, int root)
{
    tensor.syncToDevice();
    broadcast(tensor.data(), tensor.size(), root);
    tensor.mark_dirty(hal::DeviceType::CUDA);
}

void NCCLCommunicator::all_gather(const float *send_buf, size_t send_count, float *recv_buf)
{
    NCCL_CHECK(ncclAllGather(send_buf, recv_buf, send_count, ncclFloat, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::all_gather(const core::Tensor &send_tensor, core::Tensor &recv_tensor)
{
    const_cast<core::Tensor&>(send_tensor).syncToDevice();
    recv_tensor.syncToDevice();
    all_gather(send_tensor.data(), send_tensor.size(), recv_tensor.data());
    recv_tensor.mark_dirty(hal::DeviceType::CUDA);
}

void NCCLCommunicator::reduce_scatter(const float *send_buf, float *recv_buf,
                                      size_t recv_count, ReduceOp op)
{
    NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count, ncclFloat,
                                  to_nccl_op(op), comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::send(const float *buffer, size_t count, int dst)
{
    NCCL_CHECK(ncclSend(buffer, count, ncclFloat, dst, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::recv(float *buffer, size_t count, int src)
{
    NCCL_CHECK(ncclRecv(buffer, count, ncclFloat, src, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::barrier()
{
    // NCCL doesn't have a native barrier; use a zero-sized AllReduce
    NCCL_CHECK(ncclAllReduce(nullptr, nullptr, 0, ncclFloat, ncclSum, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace distributed
}  // namespace serius

#endif  // SERIUS_USE_NCCL
