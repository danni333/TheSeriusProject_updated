#include <mpi.h>

#include <exception>
#include <iostream>

#include "distributed/comm.h"

int main(int argc, char **argv)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_result = 0;
    try {
        {
            auto communicator =
                serius::distributed::Communicator::create_nccl(world_size, rank);
            if (communicator->rank() != rank ||
                communicator->world_size() != world_size) {
                local_result = 1;
            }
            communicator->barrier();
        }
    } catch (const std::exception &error) {
        std::cerr << "NCCL bootstrap failed on rank " << rank << ": "
                  << error.what() << '\n';
        local_result = 1;
    }

    int result = 0;
    MPI_Allreduce(&local_result, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
