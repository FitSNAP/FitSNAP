#include <slate/slate.hh>
#include <mpi.h>
#include <cmath>

extern "C" {

void slate_ridge_solve(double* local_ata_data, double* local_atb_data, double* solution,
                      int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Initialize SLATE
    slate::initialize();
    
    // Set up process grid (as square as possible)
    int p = static_cast<int>(std::sqrt(mpi_size));
    int q = mpi_size / p;
    while (p * q != mpi_size && p > 1) {
        --p;
        q = mpi_size / p;
    }
    
    // Create SLATE matrices
    auto A = slate::HermitianMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, n, local_ata_data, n, tile_size, p, q, comm);
    
    auto B = slate::Matrix<double>::fromScaLAPACK(
        n, 1, local_atb_data, n, tile_size, p, q, comm);
    
    // Reduce local A^T A and A^T b across all nodes
    MPI_Allreduce(MPI_IN_PLACE, local_ata_data, n*n, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, local_atb_data, n, MPI_DOUBLE, MPI_SUM, comm);
    
    // Add ridge regularization to diagonal
    for (int i = 0; i < n; ++i) {
        local_ata_data[i * n + i] += alpha;
    }
    
    // Copy reduced data to SLATE matrices
    A.insertLocalTiles();
    B.insertLocalTiles();
    
    // Solve using Cholesky factorization (positive definite solver)
    slate::posv(A, B);
    
    // Extract solution
    B.copyData(solution);
    
    // Finalize SLATE
    slate::finalize();
}

} // extern "C"
