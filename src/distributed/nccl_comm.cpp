#ifdef SERIUS_USE_NCCL

#include "nccl_comm.h"

#include <limits>
#include <stdexcept>
#include <string>

#ifdef SERIUS_USE_MPI
#include <mpi.h>
#endif

namespace serius {
namespace distributed {

#define NCCL_CHECK(cmd)                                                               \
    do {                                                                              \
        ncclResult_t res = cmd;                                                       \
        if (res != ncclSuccess) {                                                     \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(res)); \
        }                                                                             \
    } while (0)

#ifdef SERIUS_USE_MPI
#define MPI_CHECK(cmd)                                                            \
    do {                                                                          \
        int err = cmd;                                                            \
        if (err != MPI_SUCCESS) {                                                 \
            char errstr[MPI_MAX_ERROR_STRING];                                    \
            int len = 0;                                                          \
            MPI_Error_string(err, errstr, &len);                                  \
            throw std::runtime_error(std::string("MPI error: ") +                \
                                     std::string(errstr, static_cast<size_t>(len))); \
        }                                                                         \
    } while (0)
#endif

#define CUDA_CHECK(cmd)                                                              \
    do {                                                                             \
        cudaError_t err = cmd;                                                       \
        if (err != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        }                                                                            \
    } while (0)

namespace {

void validate_device_buffer(const float *buffer, size_t count, const char *name)
{
    if (count != 0 && buffer == nullptr) {
        throw std::invalid_argument(std::string("NCCL ") + name +
                                    " buffer must not be null when count is non-zero");
    }
}

}  // namespace

NCCLCommunicator::NCCLCommunicator(int world_size, int rank, ncclUniqueId id,
                                   int local_rank)
    : rank_(rank), local_rank_(local_rank < 0 ? rank : local_rank),
      world_size_(world_size)
{
    if (world_size_ <= 0 || rank_ < 0 || rank_ >= world_size_ || local_rank_ < 0) {
        throw std::invalid_argument("Invalid NCCL rank, local_rank, or world_size");
    }
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (local_rank_ >= device_count) {
        throw std::invalid_argument("NCCL local_rank does not identify a CUDA device");
    }
    CUDA_CHECK(cudaSetDevice(local_rank_));
    try {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        NCCL_CHECK(ncclCommInitRank(&comm_, world_size, id, rank));
    } catch (...) {
        if (comm_ != nullptr) {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        throw;
    }
}

#ifdef SERIUS_USE_MPI
namespace {

ncclUniqueId create_and_distribute_nccl_id(int world_size, int rank)
{
    int initialized = 0;
    MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized) {
        throw std::runtime_error(
            "NCCL ID bootstrap requires MPI to be initialized and all ranks to participate");
    }

    int finalized = 0;
    MPI_CHECK(MPI_Finalized(&finalized));
    if (finalized) {
        throw std::runtime_error("NCCL ID bootstrap cannot run after MPI has been finalized");
    }

    int actual_rank = -1;
    int actual_world_size = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &actual_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &actual_world_size));

    int arguments_valid = world_size == actual_world_size &&
                          rank == actual_rank && rank >= 0 && rank < world_size;
    int all_arguments_valid = 0;
    MPI_CHECK(MPI_Allreduce(&arguments_valid, &all_arguments_valid, 1, MPI_INT,
                            MPI_MIN, MPI_COMM_WORLD));
    if (!all_arguments_valid) {
        throw std::invalid_argument(
            "NCCL bootstrap rank and world_size must match MPI_COMM_WORLD");
    }

    ncclUniqueId id{};
    int result_code = static_cast<int>(ncclSuccess);
    if (actual_rank == 0) {
        result_code = static_cast<int>(ncclGetUniqueId(&id));
    }
    MPI_CHECK(MPI_Bcast(&result_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (result_code != static_cast<int>(ncclSuccess)) {
        const auto result = static_cast<ncclResult_t>(result_code);
        throw std::runtime_error(std::string("NCCL error creating unique ID: ") +
                                 ncclGetErrorString(result));
    }

    MPI_CHECK(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0,
                        MPI_COMM_WORLD));
    return id;
}

int get_mpi_local_rank()
{
    int rank = -1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local_comm));
    int local_rank = -1;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    return local_rank;
}

}  // namespace
#endif

NCCLCommunicator::~NCCLCommunicator()
{
    if (comm_ != nullptr) ncclCommDestroy(comm_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
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
    validate_device_buffer(buffer, count, "AllReduce");
    NCCL_CHECK(ncclAllReduce(buffer, buffer, count, ncclFloat,
                             to_nccl_op(op), comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::all_reduce(core::Tensor &tensor, ReduceOp op)
{
    if (tensor.dtype() != hal::DType::Float32) {
        throw std::invalid_argument("NCCL AllReduce tensor must use Float32");
    }
    tensor.syncToDevice();
    all_reduce(tensor.data_gpu(), tensor.size(), op);
    tensor.mark_dirty(core::Tensor::DataLocation::GPU);
}

void NCCLCommunicator::broadcast(float *buffer, size_t count, int root)
{
    validate_device_buffer(buffer, count, "Broadcast");
    if (root < 0 || root >= world_size_) {
        throw std::invalid_argument("NCCL Broadcast root is outside the communicator");
    }
    NCCL_CHECK(ncclBroadcast(buffer, buffer, count, ncclFloat, root, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::broadcast(core::Tensor &tensor, int root)
{
    if (tensor.dtype() != hal::DType::Float32) {
        throw std::invalid_argument("NCCL Broadcast tensor must use Float32");
    }
    tensor.syncToDevice();
    broadcast(tensor.data_gpu(), tensor.size(), root);
    tensor.mark_dirty(core::Tensor::DataLocation::GPU);
}

void NCCLCommunicator::all_gather(const float *send_buf, size_t send_count, float *recv_buf)
{
    validate_device_buffer(send_buf, send_count, "AllGather send");
    if (send_count > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(world_size_)) {
        throw std::overflow_error("NCCL AllGather receive size overflows size_t");
    }
    validate_device_buffer(recv_buf, send_count * static_cast<size_t>(world_size_),
                           "AllGather receive");
    NCCL_CHECK(ncclAllGather(send_buf, recv_buf, send_count, ncclFloat, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::all_gather(const core::Tensor &send_tensor, core::Tensor &recv_tensor)
{
    if (send_tensor.dtype() != hal::DType::Float32 ||
        recv_tensor.dtype() != hal::DType::Float32) {
        throw std::invalid_argument("NCCL AllGather tensors must use Float32");
    }
    if (send_tensor.size() > std::numeric_limits<size_t>::max() /
                                 static_cast<size_t>(world_size_) ||
        recv_tensor.size() != send_tensor.size() * static_cast<size_t>(world_size_)) {
        throw std::invalid_argument(
            "NCCL AllGather output size must equal input size times world_size");
    }
    send_tensor.syncToDevice();
    recv_tensor.syncToDevice();
    all_gather(send_tensor.data_gpu(), send_tensor.size(), recv_tensor.data_gpu());
    recv_tensor.mark_dirty(core::Tensor::DataLocation::GPU);
}

void NCCLCommunicator::reduce_scatter(const float *send_buf, float *recv_buf,
                                      size_t recv_count, ReduceOp op)
{
    if (recv_count > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(world_size_)) {
        throw std::overflow_error("NCCL ReduceScatter send size overflows size_t");
    }
    validate_device_buffer(send_buf, recv_count * static_cast<size_t>(world_size_),
                           "ReduceScatter send");
    validate_device_buffer(recv_buf, recv_count, "ReduceScatter receive");
    NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, recv_count, ncclFloat,
                                  to_nccl_op(op), comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::send(const float *buffer, size_t count, int dst)
{
    validate_device_buffer(buffer, count, "Send");
    if (dst < 0 || dst >= world_size_) {
        throw std::invalid_argument("NCCL Send destination is outside the communicator");
    }
    NCCL_CHECK(ncclSend(buffer, count, ncclFloat, dst, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::recv(float *buffer, size_t count, int src)
{
    validate_device_buffer(buffer, count, "Receive");
    if (src < 0 || src >= world_size_) {
        throw std::invalid_argument("NCCL Receive source is outside the communicator");
    }
    NCCL_CHECK(ncclRecv(buffer, count, ncclFloat, src, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void NCCLCommunicator::barrier()
{
    // NCCL doesn't have a native barrier; use a zero-sized AllReduce
    NCCL_CHECK(ncclAllReduce(nullptr, nullptr, 0, ncclFloat, ncclSum, comm_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

std::unique_ptr<Communicator> Communicator::create_nccl(int world_size, int rank)
{
#ifdef SERIUS_USE_MPI
    const ncclUniqueId id = create_and_distribute_nccl_id(world_size, rank);
    return std::make_unique<NCCLCommunicator>(world_size, rank, id,
                                               get_mpi_local_rank());
#else
    (void)world_size;
    (void)rank;
    throw std::runtime_error(
        "NCCL ID bootstrap requires building with SERIUS_USE_MPI; alternatively, "
        "pass a shared ncclUniqueId to create_nccl()");
#endif
}

}  // namespace distributed
}  // namespace serius

#endif  // SERIUS_USE_NCCL
