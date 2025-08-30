#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>

extern "C" {

void slate_ridge_solve(double* local_ata_data, double* local_atb_data, double* solution,
                      int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Set up 2D process grid (as square as possible)
    int p = static_cast<int>(std::sqrt(mpi_size));
    int q = mpi_size / p;
    while (p * q != mpi_size && p > 1) {
        --p;
        q = mpi_size / p;
    }
    
    // First, perform MPI reduction to get global A^T A and A^T b
    std::vector<double> global_ata(n * n);
    std::vector<double> global_atb(n);
    
    MPI_Allreduce(local_ata_data, global_ata.data(), n*n, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_atb_data, global_atb.data(), n, MPI_DOUBLE, MPI_SUM, comm);
    
    // Add ridge regularization to diagonal
    for (int i = 0; i < n; ++i) {
        global_ata[i * n + i] += alpha;
    }
    
    // Create SLATE matrices from the global data
    // Each MPI rank will have the full matrices, but SLATE will distribute internally
    // Based on page 38 of the docs - fromScaLAPACK creates distributed matrices
    int lld = n; // local leading dimension
    
    auto A = slate::HermitianMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, n,        // matrix properties
        global_ata.data(), lld,        // data and leading dimension
        tile_size, tile_size,          // tile sizes
        slate::GridOrder::Col,         // column-major process grid
        p, q,                          // process grid dimensions
        comm                           // MPI communicator
    );
    
    auto B = slate::Matrix<double>::fromScaLAPACK(
        n, 1,                          // dimensions
        global_atb.data(), lld,        // data and leading dimension
        tile_size, tile_size,          // tile sizes
        slate::GridOrder::Col,         // column-major process grid
        p, q,                          // process grid dimensions
        comm                           // MPI communicator
    );
    
    // Solve using SLATE's distributed Cholesky solver
    // From page 45 of docs: slate::posv(A, B) for positive definite solve
    slate::posv(A, B);
    
    // Copy solution back from B
    // The solution is now distributed in B, gather it
    for (int i = 0; i < n; ++i) {
        solution[i] = global_atb[i];
    }
}

} // extern "C"
