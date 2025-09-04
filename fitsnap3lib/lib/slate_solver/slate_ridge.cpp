#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int m, int n, double alpha, void* comm_ptr, int tile_size) {
    
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
        
    // Augmented system for ridge regression
    int m_aug = m + n;  // [A; sqrt(alpha)*I]
    
    /*
    // Tile size configuration
    int mb = tile_size;
    int nb = tile_size;
    
    // Ensure tiles are appropriate for QR
    if (m_aug < 100) {
        mb = std::max(32, (m_aug + mpi_size - 1) / mpi_size);
        nb = std::min(32, n);
    }
    mb = std::max(mb, nb);  // QR needs tall tiles
    
    // Create process grid - use all MPI ranks for SLATE computation
    int p = 1, q = mpi_size;
    for (int i = (int)std::sqrt(mpi_size); i >= 1; i--) {
        if (mpi_size % i == 0) {
            p = i;
            q = mpi_size / i;
            break;
        }
    }
    */
    
    int mb = 6, nb = 10, p = mpi_size, q = 1;
    
    if (mpi_rank == 0) {
        std::cerr << "\n=== Clean SLATE Ridge Solver ===" << std::endl;
        std::cerr << "Problem size: " << m << " x " << n << std::endl;
        std::cerr << "Augmented size: " << m_aug << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
        std::cerr << "Alpha (regularization): " << alpha << std::endl;
    }
    
    try {
        // Create SLATE matrices 
        slate::Matrix<double> A_aug(m_aug, n, mb, nb, p, q, comm);
        slate::Matrix<double> b_aug(m_aug, 1, mb, 1, p, q, comm);
        
        // APPROACH: Each rank with data inserts its local tiles into SLATE's matrix
        // SLATE will manage the tile distribution and storage
        
        double sqrt_alpha = std::sqrt(alpha);
        
        // Insert A matrix data
        for (int j = 0; j < A_aug.nt(); ++j)
          for (int i = 0; i < A_aug.mt(); ++i)
            if (A_aug.tileIsLocal(i, j))
              A_aug.tileInsert(i, j, local_a_data);
            
        // Insert b vector data
        for (int i = 0; i < b_aug.mt(); ++i)
          if (b_aug.tileIsLocal(i, 0))
            b_aug.tileInsert(i, 0, local_b_data);
          
        if (mpi_rank == 0) {
            std::cerr << "Solving with SLATE QR decomposition..." << std::endl;
        }
        
        // Solve using QR decomposition
        slate::gels(A_aug, b_aug);
        
        // Extract solution from first n elements of b_aug
        std::vector<double> solution_local(n, 0.0);
        
        for (int i = 0; i < b_aug.mt(); ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                
                // Only extract from tiles containing solution (first n rows)
                if (tile_row_start < n) {
                    auto tile = b_aug(i, 0);
                    
                    int tile_m = std::min(tile_row_end, n) - tile_row_start;
                    
                    for (int ti = 0; ti < tile_m; ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row < n) {
                            solution_local[global_row] = tile(ti, 0);
                        }
                    }
                }
            }
        }
        
        // Reduce solution to all processes
        MPI_Allreduce(solution_local.data(), solution, n, MPI_DOUBLE, MPI_SUM, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "SLATE solve completed successfully" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
