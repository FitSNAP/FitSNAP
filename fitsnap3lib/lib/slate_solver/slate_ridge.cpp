#include <slate/slate.hh>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <cstring>

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
    // This is needed because each node has only computed its local portion
    std::vector<double> global_ata(n * n);
    std::vector<double> global_atb(n);
    
    MPI_Allreduce(local_ata_data, global_ata.data(), n*n, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_atb_data, global_atb.data(), n, MPI_DOUBLE, MPI_SUM, comm);
    
    // Add ridge regularization to diagonal
    for (int i = 0; i < n; ++i) {
        global_ata[i * n + i] += alpha;
    }
    
    // Create SLATE distributed matrices
    // Each process will own tiles according to 2D block-cyclic distribution
    slate::HermitianMatrix<double> A(slate::Uplo::Lower, n, tile_size, p, q, comm);
    slate::Matrix<double> B(n, 1, tile_size, p, q, comm);
    slate::Matrix<double> X(n, 1, tile_size, p, q, comm);
    
    // Insert local tiles for this process
    A.insertLocalTiles();
    B.insertLocalTiles();
    X.insertLocalTiles();
    
    // Distribute the global matrix data to SLATE's 2D block-cyclic layout
    // Each process fills its local tiles
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = j; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                auto T = A(i, j);
                
                // Get the global indices for this tile
                int64_t i_start = i * tile_size;
                int64_t j_start = j * tile_size;
                int64_t i_end = std::min(i_start + tile_size, (int64_t)n);
                int64_t j_end = std::min(j_start + tile_size, (int64_t)n);
                
                // Copy data from global matrix to this tile
                for (int64_t jj = j_start; jj < j_end; ++jj) {
                    for (int64_t ii = std::max(jj, i_start); ii < i_end; ++ii) {
                        T(ii - i_start, jj - j_start) = global_ata[ii * n + jj];
                    }
                }
            }
        }
    }
    
    // Distribute the RHS vector
    for (int64_t i = 0; i < B.mt(); ++i) {
        if (B.tileIsLocal(i, 0)) {
            auto T = B(i, 0);
            
            int64_t i_start = i * tile_size;
            int64_t i_end = std::min(i_start + tile_size, (int64_t)n);
            
            for (int64_t ii = i_start; ii < i_end; ++ii) {
                T(ii - i_start, 0) = global_atb[ii];
            }
        }
    }
    
    // Solve using SLATE's distributed Cholesky solver
    // This performs the solve across all nodes in parallel
    slate::posv(A, B);
    
    // Gather solution from distributed B matrix
    // Each process contributes its local tiles
    std::fill(solution, solution + n, 0.0);
    
    for (int64_t i = 0; i < B.mt(); ++i) {
        if (B.tileIsLocal(i, 0)) {
            auto T = B(i, 0);
            
            int64_t i_start = i * tile_size;
            int64_t i_end = std::min(i_start + tile_size, (int64_t)n);
            
            for (int64_t ii = i_start; ii < i_end; ++ii) {
                solution[ii] = T(ii - i_start, 0);
            }
        }
    }
    
    // All-reduce to ensure all processes have the complete solution
    MPI_Allreduce(MPI_IN_PLACE, solution, n, MPI_DOUBLE, MPI_SUM, comm);
}

} // extern "C"
