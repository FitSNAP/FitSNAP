#include <slate/slate.hh>
#include <blas.hh>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" {

void slate_ridge_solve(double* local_ata_data, double* local_atb_data, double* solution,
                      int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // First, perform MPI reduction to get global A^T A and A^T b
    // This is needed because each node has only computed its local portion
    std::vector<double> global_ata(n * n);
    std::vector<double> global_atb(n);
    
    MPI_Allreduce(local_ata_data, global_ata.data(), n*n, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_atb_data, global_atb.data(), n, MPI_DOUBLE, MPI_SUM, comm);
    
    // Add ridge regularization to diagonal
    for (int i = 0; i < n; ++i) {
        global_ata[i * n + i] += alpha;
    }
    
    // Set up 2D process grid (as square as possible)
    int p = static_cast<int>(std::sqrt(mpi_size));
    int q = mpi_size / p;
    while (p * q != mpi_size && p > 1) {
        --p;
        q = mpi_size / p;
    }
    
    // Create SLATE matrices from ScaLAPACK-style layout
    // This handles the distribution automatically
    slate::HermitianMatrix<double> A = slate::HermitianMatrix<double>::fromScaLAPACK(
        slate::Uplo::Lower, n, global_ata.data(), n, tile_size, p, q, comm);
    
    slate::Matrix<double> B = slate::Matrix<double>::fromScaLAPACK(
        n, 1, global_atb.data(), n, tile_size, p, q, comm);
    
    // Solve using SLATE's distributed Cholesky solver
    // posv: A X = B, where A is symmetric positive definite
    slate::posv(A, B);
    
    // Copy solution back from B (which now contains X)
    // Use ScaLAPACK-style gathering
    B.copyDataToScaLAPACK(solution, n);
}

} // extern "C"
