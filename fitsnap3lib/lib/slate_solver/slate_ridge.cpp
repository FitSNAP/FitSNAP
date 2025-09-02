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

// Create a RowMajorMatrix class that inherits from slate::Matrix
template<typename scalar_t>
class RowMajorMatrix : public slate::Matrix<scalar_t> {
public:
    RowMajorMatrix(int64_t m, int64_t n, int64_t mb, int64_t nb,
                   slate::GridOrder order, int p, int q, MPI_Comm mpi_comm)
        : slate::Matrix<scalar_t>(m, n, mb, nb, order, p, q, mpi_comm) {
        // Set the layout to RowMajor
        this->layout_ = slate::Layout::RowMajor;
    }
};

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Gather all local sizes to compute global m
    // Only rank 0 within each node has non-zero m_local
    std::vector<int> all_m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, all_m_locals.data(), 1, MPI_INT, comm);
    
    int m_total = 0;
    std::vector<int> row_offsets(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; i++) {
        m_total += all_m_locals[i];
        row_offsets[i + 1] = row_offsets[i] + all_m_locals[i];
    }
    
    // Debug output
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Solver (Row-Major, Zero-Copy) ===" << std::endl;
        std::cerr << "Global dimensions: " << m_total << " x " << n << std::endl;
        std::cerr << "Ridge parameter (alpha): " << alpha << std::endl;
        std::cerr << "MPI ranks: " << mpi_size << std::endl;
        std::cerr << "Tile size: " << tile_size << std::endl;
    }
    
    // For ridge regression, solve the augmented system:
    // [  A  ]     [  b  ]
    // [√α*I ] x = [ 0  ]
    int m_aug = m_total + n;
    
    // Use appropriate tile size
    int mb, nb;
    if (m_total < tile_size) {
        // Small problem: make tiles small enough to distribute
        mb = std::max(1, (m_total + mpi_size - 1) / mpi_size);
        nb = std::min(n, tile_size);
    } else {
        mb = tile_size;
        nb = tile_size;
    }
    
    // Create process grid for SLATE's 2D block-cyclic distribution
    int p = 1, q = 1;
    for (int i = (int)std::sqrt(mpi_size); i >= 1; i--) {
        if (mpi_size % i == 0) {
            p = i;
            q = mpi_size / i;
            break;
        }
    }
    
    if (mpi_rank == 0) {
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
        std::cerr << "Augmented system: " << m_aug << " x " << n << std::endl;
        std::cerr << "Tile sizes: " << mb << " x " << nb << std::endl;
    }
    
    try {
        // Create SLATE matrices with ROW-MAJOR layout
        RowMajorMatrix<double> A_aug(m_aug, n, mb, nb, slate::GridOrder::Col, p, q, comm);
        slate::Matrix<double> b_aug(m_aug, 1, mb, 1, slate::GridOrder::Col, p, q, comm);
        
        // Only ranks with data (rank 0 within each node) insert tiles
        if (m_local > 0 && local_a_data != nullptr && local_b_data != nullptr) {
            int my_row_start = row_offsets[mpi_rank];
            int my_row_end = row_offsets[mpi_rank + 1];
            
            // Insert tiles for matrix A - tiles point directly to shared array memory
            for (int j = 0; j < A_aug.nt(); ++j) {
                for (int i = 0; i < A_aug.mt(); ++i) {
                    if (A_aug.tileIsLocal(i, j)) {
                        int tile_row_start = i * mb;
                        int tile_row_end = std::min((i + 1) * mb, m_aug);
                        int tile_col_start = j * nb;
                        int tile_col_end = std::min((j + 1) * nb, n);
                        
                        int tile_m = tile_row_end - tile_row_start;
                        int tile_n = tile_col_end - tile_col_start;
                        
                        // Check if this tile overlaps with my data rows
                        if (tile_row_start < m_total && 
                            tile_row_start >= my_row_start && 
                            tile_row_start < my_row_end) {
                            
                            // Point directly to shared array memory - zero copy!
                            int local_row_offset = tile_row_start - my_row_start;
                            double* tile_ptr = local_a_data + local_row_offset * n + tile_col_start;
                            
                            // Insert with row-major layout, stride = n
                            A_aug.tileInsert(i, j, tile_ptr, n);
                        }
                    }
                }
            }
            
            // Insert tiles for vector b
            for (int i = 0; i < b_aug.mt(); ++i) {
                if (b_aug.tileIsLocal(i, 0)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_m = tile_row_end - tile_row_start;
                    
                    // Check if this tile overlaps with my b data
                    if (tile_row_start < m_total && 
                        tile_row_start >= my_row_start && 
                        tile_row_start < my_row_end) {
                        
                        // Point directly to shared array memory
                        int local_row_offset = tile_row_start - my_row_start;
                        double* tile_ptr = local_b_data + local_row_offset;
                        
                        b_aug.tileInsert(i, 0, tile_ptr, tile_m);
                    }
                }
            }
        }
        
        // All ranks (including those without data) handle regularization rows
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                if (A_aug.tileIsLocal(i, j)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_col_start = j * nb;
                    int tile_col_end = std::min((j + 1) * nb, n);
                    
                    int tile_m = tile_row_end - tile_row_start;
                    int tile_n = tile_col_end - tile_col_start;
                    
                    // Handle regularization rows (√α * I)
                    if (tile_row_start >= m_total) {
                        double* reg_tile = new double[tile_m * tile_n];
                        std::fill(reg_tile, reg_tile + tile_m * tile_n, 0.0);
                        
                        double sqrt_alpha = std::sqrt(alpha);
                        for (int ti = 0; ti < tile_m; ++ti) {
                            for (int tj = 0; tj < tile_n; ++tj) {
                                int global_row = tile_row_start + ti;
                                int global_col = tile_col_start + tj;
                                if (global_row >= m_total && global_row - m_total == global_col) {
                                    reg_tile[ti * tile_n + tj] = sqrt_alpha;
                                }
                            }
                        }
                        A_aug.tileInsert(i, j, reg_tile, tile_n);
                    }
                }
            }
        }
        
        // Regularization part of b (zeros)
        for (int i = 0; i < b_aug.mt(); ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                int tile_m = tile_row_end - tile_row_start;
                
                if (tile_row_start >= m_total) {
                    double* zero_tile = new double[tile_m];
                    std::fill(zero_tile, zero_tile + tile_m, 0.0);
                    b_aug.tileInsert(i, 0, zero_tile, tile_m);
                }
            }
        }
        
        MPI_Barrier(comm);
        
        if (mpi_rank == 0) {
            std::cerr << "Solving with QR decomposition..." << std::endl;
        }
        
        // ALL ranks participate in the solve
        slate::gels(A_aug, b_aug);
        
        if (mpi_rank == 0) {
            std::cerr << "QR solve completed. Extracting solution..." << std::endl;
        }
        
        // Extract solution (first n elements of b_aug)
        std::vector<double> solution_local(n, 0.0);
        
        for (int i = 0; i < b_aug.nt() && i * mb < n; ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, n);
                int tile_m = tile_row_end - tile_row_start;
                
                auto T = b_aug(i, 0);
                double* tile_data = T.data();
                
                for (int ti = 0; ti < tile_m; ++ti) {
                    solution_local[tile_row_start + ti] = tile_data[ti];
                }
            }
        }
        
        // Reduce the solution to all processes
        MPI_Allreduce(solution_local.data(), solution, n, MPI_DOUBLE, MPI_SUM, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "Solution extracted successfully." << std::endl;
            std::cerr << "=====================================\n" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
