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
    
    // Gather all local sizes to understand data distribution
    std::vector<int> all_m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, all_m_locals.data(), 1, MPI_INT, comm);
    
    // Count ranks with actual data (in shared memory architecture, only rank 0 per node has data)
    int ranks_with_data = 0;
    for (int i = 0; i < mpi_size; i++) {
        if (all_m_locals[i] > 0) {
            ranks_with_data++;
        }
    }
    
    // Build cumulative row offsets for global indexing
    std::vector<int> row_offsets(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; i++) {
        row_offsets[i + 1] = row_offsets[i] + all_m_locals[i];
    }
    
    // Augmented system for ridge regression
    int m_aug = m + n;  // [A; sqrt(alpha)*I]
    
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
    
    if (mpi_rank == 0) {
        std::cerr << "\n=== Clean SLATE Ridge Solver ===" << std::endl;
        std::cerr << "Problem size: " << m << " x " << n << std::endl;
        std::cerr << "Augmented size: " << m_aug << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
        std::cerr << "Ranks with data: " << ranks_with_data << " out of " << mpi_size << std::endl;
        std::cerr << "Alpha (regularization): " << alpha << std::endl;
    }
    
    try {
        // Create SLATE matrices - let SLATE manage its own storage
        slate::Matrix<double> A_aug(m_aug, n, mb, nb, p, q, comm);
        slate::Matrix<double> b_aug(m_aug, 1, mb, 1, p, q, comm);
        
        // APPROACH: Each rank with data inserts its local tiles into SLATE's matrix
        // SLATE will manage the tile distribution and storage
        
        double sqrt_alpha = std::sqrt(alpha);
        
        // Insert data tiles from local filtered training data
        if (m_local > 0) {
            // This rank has actual data to contribute
            int my_start_row = row_offsets[mpi_rank];
            
            // Insert A matrix data
            for (int j = 0; j < A_aug.nt(); ++j) {
                for (int i = 0; i < A_aug.mt(); ++i) {
                    if (A_aug.tileIsLocal(i, j)) {
                        int tile_row_start = i * mb;
                        int tile_row_end = std::min((i + 1) * mb, m_aug);
                        int tile_col_start = j * nb;
                        int tile_col_end = std::min((j + 1) * nb, n);
                        
                        if (tile_row_start >= m) {
                            // Regularization part (sqrt(alpha) * I)
                            continue;  // Will handle separately
                        }
                        
                        // Check if any part of this tile overlaps with our data
                        if (tile_row_end > my_start_row && tile_row_start < my_start_row + m_local) {
                            // This tile needs some of our data
                            int tile_m = tile_row_end - tile_row_start;
                            int tile_n = tile_col_end - tile_col_start;
                            
                            // Allocate tile storage
                            std::vector<double> tile_data(tile_m * tile_n, 0.0);
                            
                            // Fill tile with our data (where it overlaps)
                            for (int ti = 0; ti < tile_m; ++ti) {
                                int global_row = tile_row_start + ti;
                                if (global_row >= my_start_row && global_row < my_start_row + m_local) {
                                    int local_row = global_row - my_start_row;
                                    for (int tj = 0; tj < tile_n; ++tj) {
                                        int global_col = tile_col_start + tj;
                                        if (global_col < n && local_row < m_local) {
                                            // Column-major storage for SLATE
                                            tile_data[ti + tj * tile_m] = local_a_data[local_row * n + global_col];
                                        }
                                    }
                                }
                            }
                            
                            // Insert this tile into SLATE matrix
                            A_aug.tileInsert(i, j, tile_data.data());
                        }
                    }
                }
            }
            
            // Insert b vector data
            for (int i = 0; i < b_aug.mt(); ++i) {
                if (b_aug.tileIsLocal(i, 0)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    
                    if (tile_row_start >= m) {
                        // Regularization part (zeros)
                        continue;  // Will handle separately
                    }
                    
                    // Check if this tile overlaps with our data
                    if (tile_row_end > my_start_row && tile_row_start < my_start_row + m_local) {
                        int tile_m = tile_row_end - tile_row_start;
                        
                        std::vector<double> tile_data(tile_m, 0.0);
                        
                        for (int ti = 0; ti < tile_m; ++ti) {
                            int global_row = tile_row_start + ti;
                            if (global_row >= my_start_row && global_row < my_start_row + m_local) {
                                int local_row = global_row - my_start_row;
                                if (local_row < m_local) {
                                    tile_data[ti] = local_b_data[local_row];
                                }
                            }
                        }
                        
                        b_aug.tileInsert(i, 0, tile_data.data());
                    }
                }
            }
        }
        
        // ALL ranks handle regularization tiles (bottom part of augmented matrix)
        // These are sqrt(alpha) * I for A_aug and zeros for b_aug
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                if (A_aug.tileIsLocal(i, j)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_col_start = j * nb;
                    int tile_col_end = std::min((j + 1) * nb, n);
                    
                    // Only process regularization rows
                    if (tile_row_start >= m) {
                        int tile_m = tile_row_end - tile_row_start;
                        int tile_n = tile_col_end - tile_col_start;
                        
                        std::vector<double> tile_data(tile_m * tile_n, 0.0);
                        
                        // Fill diagonal with sqrt(alpha)
                        for (int ti = 0; ti < tile_m; ++ti) {
                            for (int tj = 0; tj < tile_n; ++tj) {
                                int global_row = tile_row_start + ti;
                                int global_col = tile_col_start + tj;
                                if (global_row - m == global_col) {
                                    // Diagonal element - column major storage
                                    tile_data[ti + tj * tile_m] = sqrt_alpha;
                                }
                            }
                        }
                        
                        A_aug.tileInsert(i, j, tile_data.data());
                    }
                }
            }
        }
        
        // Insert zero tiles for b regularization part
        for (int i = 0; i < b_aug.mt(); ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                
                if (tile_row_start >= m) {
                    int tile_m = tile_row_end - tile_row_start;
                    std::vector<double> tile_data(tile_m, 0.0);
                    b_aug.tileInsert(i, 0, tile_data.data());
                }
            }
        }
        
        // Synchronize before solving
        MPI_Barrier(comm);
        
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
