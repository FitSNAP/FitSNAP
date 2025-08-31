#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <cstring>

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Debug output
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Regression Solver (Zero-Copy Row-Major) ===" << std::endl;
        std::cerr << "Features (n): " << n << std::endl;
        std::cerr << "Alpha: " << alpha << std::endl;
        std::cerr << "MPI ranks: " << mpi_size << std::endl;
        std::cerr << "Tile size: " << tile_size << std::endl;
        std::cerr.flush();
    }
    
    // Gather all local row counts
    std::vector<int> m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, m_locals.data(), 1, MPI_INT, comm);
    
    int m_total = 0;
    std::vector<int> m_offsets(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i) {
        m_total += m_locals[i];
        m_offsets[i + 1] = m_offsets[i] + m_locals[i];
    }
    
    if (mpi_rank == 0) {
        std::cerr << "Total rows (m_total): " << m_total << std::endl;
        std::cerr << "Using tileInsert for zero-copy with row-major data" << std::endl;
        std::cerr.flush();
    }
    
    // ZERO-COPY APPROACH WITH ROW-MAJOR DATA
    // Instead of using fromScaLAPACK (which expects column-major),
    // we create an empty SLATE matrix and use tileInsert to wrap our row-major data
    
    // Set up process grid
    int p = mpi_size;
    int q = 1;  // 1D row distribution
    
    // Tile sizes
    int mb = tile_size;  // Row tile size
    int nb = tile_size;  // Column tile size
    
    // Augmented system dimensions
    int m_aug_total = m_total + n;  // Add n regularization rows
    
    if (mpi_rank == 0) {
        std::cerr << "\nCreating augmented system:" << std::endl;
        std::cerr << "  Original: " << m_total << " x " << n << std::endl;
        std::cerr << "  Augmented: " << m_aug_total << " x " << n << std::endl;
        std::cerr << "  Process grid: " << p << " x " << q << std::endl;
        std::cerr << "  Tile sizes: " << mb << " x " << nb << std::endl;
        std::cerr.flush();
    }
    
    try {
        // Create empty SLATE matrices
        auto A_aug = slate::Matrix<double>(m_aug_total, n, mb, nb, 
                                           slate::GridOrder::Col, p, q, comm);
        auto b_aug = slate::Matrix<double>(m_aug_total, 1, mb, 1,
                                           slate::GridOrder::Col, p, q, comm);
        
        // Calculate tile distribution
        int mt = (m_total + mb - 1) / mb;  // Number of tile rows for original data
        int nt = (n + nb - 1) / nb;         // Number of tile columns
        
        if (mpi_rank == 0) {
            std::cerr << "Inserting tiles from row-major data..." << std::endl;
            std::cerr.flush();
        }
        
        // INSERT TILES FOR ORIGINAL DATA (A matrix)
        // Each rank inserts tiles for its local rows
        int local_start_row = m_offsets[mpi_rank];
        int local_end_row = m_offsets[mpi_rank + 1];
        
        for (int i = local_start_row; i < local_end_row; i += mb) {
            int tile_i = i / mb;
            int rows_in_tile = std::min(mb, local_end_row - i);
            
            for (int j = 0; j < n; j += nb) {
                int tile_j = j / nb;
                int cols_in_tile = std::min(nb, n - j);
                
                // Check if this tile belongs to this rank in the process grid
                if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                    // For row-major data, we need to handle tiles carefully
                    // Each tile covers rows_in_tile x cols_in_tile
                    // But our data is stored row-major with stride n
                    
                    // If tile fits perfectly in our contiguous data, we can use zero-copy
                    if (cols_in_tile == nb) {
                        // Calculate pointer to the start of this tile in row-major data
                        int local_i = i - local_start_row;
                        double* tile_data = &local_a_data[local_i * n + j];
                        A_aug.tileInsert(tile_i, tile_j, slate::HostNum, 
                                        slate::Layout::RowMajor, tile_data, n);
                    } else {
                        // For partial tiles, we need to copy data to ensure correct layout
                        std::vector<double> tile_buffer(rows_in_tile * cols_in_tile);
                        for (int r = 0; r < rows_in_tile; ++r) {
                            int local_i = (i - local_start_row) + r;
                            for (int c = 0; c < cols_in_tile; ++c) {
                                tile_buffer[r * cols_in_tile + c] = local_a_data[local_i * n + j + c];
                            }
                        }
                        A_aug.tileInsert(tile_i, tile_j, slate::HostNum, 
                                        slate::Layout::RowMajor, tile_buffer.data(), cols_in_tile);
                    }
                }
            }
        }
        
        // INSERT TILES FOR b vector
        for (int i = local_start_row; i < local_end_row; i += mb) {
            int tile_i = i / mb;
            int rows_in_tile = std::min(mb, local_end_row - i);
            
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                int local_i = i - local_start_row;
                double* tile_data = &local_b_data[local_i];
                
                // For vector, use column-major (it's a single column)
                // Leading dimension is the number of rows in the tile
                // Use HostNum (-1) for CPU-only operations
                b_aug.tileInsert(tile_i, 0, slate::HostNum, 
                                slate::Layout::ColMajor, tile_data, rows_in_tile);
            }
        }
        
        // ADD REGULARIZATION ROWS
        // Distribute regularization rows across ranks
        int reg_rows_per_rank = n / mpi_size;
        int reg_rows_remainder = n % mpi_size;
        int my_reg_rows = reg_rows_per_rank + (mpi_rank < reg_rows_remainder ? 1 : 0);
        int my_reg_start = mpi_rank * reg_rows_per_rank + std::min(mpi_rank, reg_rows_remainder);
        
        if (mpi_rank == 0) {
            std::cerr << "Adding regularization rows..." << std::endl;
            std::cerr.flush();
        }
        
        // Allocate memory for regularization tiles
        double sqrt_alpha = std::sqrt(alpha);
        std::vector<double> reg_tile_data;
        std::vector<double> reg_b_data;
        
        for (int i = 0; i < my_reg_rows; i += mb) {
            int global_row = m_total + my_reg_start + i;
            int tile_i = global_row / mb;
            int rows_in_tile = std::min(mb, my_reg_rows - i);
            
            // Allocate and fill regularization tile
            reg_tile_data.assign(rows_in_tile * n, 0.0);
            reg_b_data.assign(rows_in_tile, 0.0);
            
            // Set diagonal elements to sqrt(alpha)
            for (int r = 0; r < rows_in_tile; ++r) {
                int diag_col = my_reg_start + i + r;
                if (diag_col < n) {
                    reg_tile_data[r * n + diag_col] = sqrt_alpha;
                }
            }
            
            // Insert regularization tiles
            for (int j = 0; j < n; j += nb) {
                int tile_j = j / nb;
                int cols_in_tile = std::min(nb, n - j);
                
                if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                    if (cols_in_tile == nb) {
                        // Full tile - can use direct pointer
                        double* tile_ptr = &reg_tile_data[j];
                        A_aug.tileInsert(tile_i, tile_j, slate::HostNum, 
                                        slate::Layout::RowMajor, tile_ptr, n);
                    } else {
                        // Partial tile - need to copy with correct stride
                        std::vector<double> partial_tile(rows_in_tile * cols_in_tile);
                        for (int r = 0; r < rows_in_tile; ++r) {
                            for (int c = 0; c < cols_in_tile; ++c) {
                                partial_tile[r * cols_in_tile + c] = reg_tile_data[r * n + j + c];
                            }
                        }
                        A_aug.tileInsert(tile_i, tile_j, slate::HostNum, 
                                        slate::Layout::RowMajor, partial_tile.data(), cols_in_tile);
                    }
                }
            }
            
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                b_aug.tileInsert(tile_i, 0, slate::HostNum, 
                                slate::Layout::ColMajor, reg_b_data.data(), rows_in_tile);
            }
        }
        
        MPI_Barrier(comm);
        if (mpi_rank == 0) {
            std::cerr << "Solving augmented system with QR..." << std::endl;
            std::cerr.flush();
        }
        
        // SOLVE WITH QR DECOMPOSITION
        slate::gels(A_aug, b_aug);
        
        if (mpi_rank == 0) {
            std::cerr << "QR solve completed. Extracting solution..." << std::endl;
            std::cerr.flush();
        }
        
        // Extract solution from first n elements of b_aug
        std::fill(solution, solution + n, 0.0);
        
        for (int i = 0; i < n; ++i) {
            int tile_i = i / mb;
            int local_i = i % mb;
            
            if (b_aug.tileRank(tile_i, 0) == mpi_rank && b_aug.tileIsLocal(tile_i, 0)) {
                auto tile = b_aug(tile_i, 0);
                if (local_i < tile.mb()) {
                    solution[i] = tile(local_i, 0);
                }
            }
        }
        
        // Gather solution to all ranks
        MPI_Allreduce(MPI_IN_PLACE, solution, n, MPI_DOUBLE, MPI_SUM, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "\nRidge regression solved successfully!" << std::endl;
            std::cerr << "Solution vector has " << n << " coefficients" << std::endl;
            
            // Print first few coefficients
            std::cerr << "First 5 coefficients: ";
            for (int i = 0; i < std::min(5, n); ++i) {
                std::cerr << std::fixed << std::setprecision(6) << solution[i] << " ";
            }
            std::cerr << std::endl;
            std::cerr << "=====================================\n" << std::endl;
            std::cerr.flush();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        std::cerr.flush();
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
