#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int n, double alpha, void* comm_ptr, int tile_size) {
    
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
    
    // Get total number of rows across all processes
    int m_total;
    MPI_Allreduce(&m_local, &m_total, 1, MPI_INT, MPI_SUM, comm);
    
    // Ridge regression via augmented least squares:
    // Solve: [A    ] x ≈ [b]
    //        [√α*I ]     [0]
    // This is an (m_total + n) × n system
    
    int m_augmented = m_total + n;
    double sqrt_alpha = std::sqrt(alpha);
    
    // Create the augmented matrix A_aug of size (m_total + n) × n
    slate::Matrix<double> A_aug(m_augmented, n, tile_size, p, q, comm);
    
    // For least squares, we need a BX matrix of size max(m_augmented, n) × 1
    int max_mn = std::max(m_augmented, n);
    slate::Matrix<double> BX(max_mn, 1, tile_size, p, q, comm);
    
    // Allocate local tiles
    A_aug.insertLocalTiles(slate::Target::Host);
    BX.insertLocalTiles(slate::Target::Host);
    
    // First, we need to figure out the global row offset for this rank's data
    int row_offset;
    MPI_Scan(&m_local, &row_offset, 1, MPI_INT, MPI_SUM, comm);
    row_offset -= m_local; // Exclusive scan
    
    // Fill the augmented matrix A_aug with local data from A
    for (int64_t j = 0; j < A_aug.nt(); ++j) {
        for (int64_t i = 0; i < A_aug.mt(); ++i) {
            if (A_aug.tileIsLocal(i, j)) {
                auto tile = A_aug(i, j);
                double* tile_data = tile.data();
                int64_t mb = tile.mb();
                int64_t nb = tile.nb();
                int64_t stride = tile.stride();
                
                // Global row and column indices for this tile
                int64_t tile_row_start = i * tile_size;
                int64_t tile_col_start = j * tile_size;
                
                // Initialize tile to zero
                for (int64_t jj = 0; jj < nb; ++jj) {
                    for (int64_t ii = 0; ii < mb; ++ii) {
                        tile_data[ii + jj * stride] = 0.0;
                    }
                }
                
                // Fill with local A data if this tile overlaps with our local rows
                for (int64_t jj = 0; jj < nb && tile_col_start + jj < n; ++jj) {
                    for (int64_t ii = 0; ii < mb && tile_row_start + ii < m_augmented; ++ii) {
                        int64_t global_row = tile_row_start + ii;
                        int64_t global_col = tile_col_start + jj;
                        
                        if (global_row >= row_offset && global_row < row_offset + m_local) {
                            // This is our local data from A
                            int local_row = global_row - row_offset;
                            tile_data[ii + jj * stride] = local_a_data[local_row * n + global_col];
                        } else if (global_row >= m_total && global_row < m_augmented) {
                            // This is the regularization part: √α*I
                            if (global_col == global_row - m_total) {
                                tile_data[ii + jj * stride] = sqrt_alpha;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fill the augmented RHS vector b_aug
    for (int64_t i = 0; i < BX.mt(); ++i) {
        if (BX.tileIsLocal(i, 0)) {
            auto tile = BX(i, 0);
            double* tile_data = tile.data();
            int64_t mb = tile.mb();
            int64_t stride = tile.stride();
            
            int64_t tile_row_start = i * tile_size;
            
            // Initialize to zero
            for (int64_t ii = 0; ii < mb; ++ii) {
                tile_data[ii] = 0.0;
            }
            
            // Fill with local b data
            for (int64_t ii = 0; ii < mb && tile_row_start + ii < m_augmented; ++ii) {
                int64_t global_row = tile_row_start + ii;
                
                if (global_row >= row_offset && global_row < row_offset + m_local) {
                    // This is our local data from b
                    int local_row = global_row - row_offset;
                    tile_data[ii] = local_b_data[local_row];
                }
                // Rows m_total to m_augmented are already zero (regularization RHS)
            }
        }
    }
    
    // Set options for SLATE
    slate::Options opts;
    opts[slate::Option::Target] = slate::Target::HostTask;
    
    // Solve the least squares problem using QR factorization
    // From SLATE docs page 46-47: gels solves overdetermined systems with QR
    slate::least_squares_solve(A_aug, BX, opts);
    
    // Extract the solution from BX
    // For overdetermined systems, the solution is in the first n rows of BX
    std::vector<double> local_solution(n, 0.0);
    
    for (int64_t i = 0; i < BX.mt(); ++i) {
        if (BX.tileIsLocal(i, 0)) {
            auto tile = BX(i, 0);
            double* tile_data = tile.data();
            int64_t mb = tile.mb();
            int64_t tile_row_start = i * tile_size;
            
            for (int64_t ii = 0; ii < mb && tile_row_start + ii < n; ++ii) {
                int64_t global_row = tile_row_start + ii;
                local_solution[global_row] = tile_data[ii];
            }
        }
    }
    
    // All-reduce to get complete solution on all processes
    MPI_Allreduce(local_solution.data(), solution, n, MPI_DOUBLE, MPI_SUM, comm);
}

} // extern "C"
