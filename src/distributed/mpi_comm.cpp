#ifdef SERIUS_USE_MPI

#include "mpi_comm.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace serius {
namespace distributed {

#define MPI_CHECK(cmd)                                                            \
    do {                                                                          \
        int err = cmd;                                                            \
        if (err != MPI_SUCCESS) {                                                 \
            char errstr[MPI_MAX_ERROR_STRING];                                    \
            int len;                                                              \
            MPI_Error_string(err, errstr, &len);                                  \
            throw std::runtime_error(std::string("MPI error: ") + errstr);        \
        }                                                                         \
    } while (0)

MPICommunicator::MPICommunicator(int *argc, char ***argv)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_CHECK(MPI_Init(argc, argv));
        owns_mpi_ = true;
    } else {
        owns_mpi_ = false;
    }

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size_));

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_,
                                  MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank_));
    MPI_CHECK(MPI_Comm_free(&local_comm));
}

MPICommunicator::~MPICommunicator()
{
    if (owns_mpi_) {
        MPI_Finalize();
    }
}

MPI_Op MPICommunicator::to_mpi_op(ReduceOp op)
{
    switch (op) {
        case ReduceOp::Sum:
        case ReduceOp::Avg:
            return MPI_SUM;
        case ReduceOp::Min:
            return MPI_MIN;
        case ReduceOp::Max:
            return MPI_MAX;
    }
    return MPI_SUM;
}

void MPICommunicator::all_reduce(float *buffer, size_t count, ReduceOp op)
{
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buffer, static_cast<int>(count),
                            MPI_FLOAT, to_mpi_op(op), MPI_COMM_WORLD));

    // MPI doesn't have native Avg, so divide by world_size
    if (op == ReduceOp::Avg) {
        float inv = 1.0f / static_cast<float>(world_size_);
        for (size_t i = 0; i < count; ++i) {
            buffer[i] *= inv;
        }
    }
}

void MPICommunicator::broadcast(float *buffer, size_t count, int root)
{
    MPI_CHECK(MPI_Bcast(buffer, static_cast<int>(count), MPI_FLOAT, root, MPI_COMM_WORLD));
}

void MPICommunicator::all_gather(const float *send_buf, size_t send_count, float *recv_buf)
{
    MPI_CHECK(MPI_Allgather(send_buf, static_cast<int>(send_count), MPI_FLOAT,
                            recv_buf, static_cast<int>(send_count), MPI_FLOAT,
                            MPI_COMM_WORLD));
}

void MPICommunicator::reduce_scatter(const float *send_buf, float *recv_buf,
                                     size_t recv_count, ReduceOp op)
{
    // MPI_Reduce_scatter_block: each rank gets recv_count elements
    MPI_CHECK(MPI_Reduce_scatter_block(send_buf, recv_buf,
                                       static_cast<int>(recv_count),
                                       MPI_FLOAT, to_mpi_op(op),
                                       MPI_COMM_WORLD));

    if (op == ReduceOp::Avg) {
        float inv = 1.0f / static_cast<float>(world_size_);
        for (size_t i = 0; i < recv_count; ++i) {
            recv_buf[i] *= inv;
        }
    }
}

void MPICommunicator::send(const float *buffer, size_t count, int dst)
{
    MPI_CHECK(MPI_Send(buffer, static_cast<int>(count), MPI_FLOAT, dst, 0, MPI_COMM_WORLD));
}

void MPICommunicator::recv(float *buffer, size_t count, int src)
{
    MPI_CHECK(MPI_Recv(buffer, static_cast<int>(count), MPI_FLOAT, src, 0,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE));
}

void MPICommunicator::barrier()
{
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

}  // namespace distributed
}  // namespace serius

#endif  // SERIUS_USE_MPI
