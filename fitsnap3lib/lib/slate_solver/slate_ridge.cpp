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
// This allows us to access the protected layout_ member and set it to RowMajor
template<typename scalar_t>
class RowMajorMatrix : public slate::Matrix<scalar_t> {
public:
    // Constructor that creates a row-major matrix
    RowMajorMatrix(int64_t m, int64_t n, int64_t mb, int64_t nb,
                   slate::GridOrder order, int p, int q, MPI_Comm mpi_comm)
        : slate::Matrix<scalar_t>(m, n, mb, nb, order, p, q, mpi_comm) {
        // Set the layout to RowMajor - this is the key!
        // layout_ is protected in BaseMatrix, so we can access it here
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
    
    // Debug output
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Solver ===" << std::endl;
        std::cerr << " Features (n): " << n;
        std::cerr << " Alpha: " << alpha;
        std::cerr << " MPI ranks: " << mpi_size;
        std::cerr << " Tile size: " << tile_size << std::endl;
        std::cerr.flush();
    }
    
    // Gather all local row counts
    std::vector<int> m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, m_locals.data(), 1, MPI_INT, comm);
    
    // Calculate total rows and global offsets for each rank
    // Simple approach: each rank owns a contiguous block of rows
    std::vector<int> rank_offsets(mpi_size + 1, 0);
    for (int r = 0; r < mpi_size; ++r) {
        rank_offsets[r + 1] = rank_offsets[r] + m_locals[r];
    }
    int m_total = rank_offsets[mpi_size];
    
    // This rank's global offset is simply the sum of rows from lower ranks
    int global_offset = rank_offsets[mpi_rank];
    
    if (mpi_rank == 0) {
        std::cerr << "Total rows (m_total): " << m_total << std::endl;
        
        // Debug: show row distribution
        std::cerr << "Row distribution by rank: ";
        for (int i = 0; i < std::min(mpi_size, 10); ++i) {
            std::cerr << m_locals[i] << " ";
        }
        if (mpi_size > 10) std::cerr << "...";
        std::cerr << std::endl;
        
        std::cerr << "Rank offsets: ";
        for (int i = 0; i <= std::min(mpi_size, 10); ++i) {
            std::cerr << rank_offsets[i] << " ";
        }
        if (mpi_size > 10) std::cerr << "...";
        std::cerr << std::endl;

    }
    
    // Set up process grid
    int p = mpi_size;
    int q = 1;  // 1D row distribution
    
    // Tile sizes - for column dimension, use min of tile_size and n
    // This prevents tiles from being wider than the matrix
    int mb = tile_size;  // Row tile size
    int nb = std::min(tile_size, n);  // Column tile size (can't exceed n)
    
    // Augmented system dimensions
    int m_aug_total = m_total + n;  // Add n regularization rows
    
    if (mpi_rank == 0) {
        std::cerr << "\nCreating augmented system:" << std::endl;
        std::cerr << " Original: " << m_total << " x " << n;
        std::cerr << " Augmented: " << m_aug_total << " x " << n;
        std::cerr << " Process grid: " << p << " x " << q;
        std::cerr << " Tile sizes: " << mb << " x " << nb << std::endl;
        std::cerr.flush();
    }
    
    try {
        // Create ROW-MAJOR SLATE matrices using our custom class
        RowMajorMatrix<double> A_aug(m_aug_total, n, mb, nb, slate::GridOrder::Col, p, q, comm);
        
        // b vector is still column-major (it's a single column)
        auto b_aug = slate::Matrix<double>(m_aug_total, 1, mb, 1, slate::GridOrder::Col, p, q, comm);
        
        // INSERT TILES - simpler approach: just copy data into SLATE format
        // This avoids all the complexity of trying to make zero-copy work
        // with different data distributions
        
        // Allocate workspace for all tiles we own
        std::vector<std::vector<double>> a_tiles;
        std::vector<std::vector<double>> b_tiles;
        
        // Process all tiles in the augmented matrix
        int mt_total = (m_aug_total + mb - 1) / mb;
        int nt_total = 1;  // b vector has only 1 column of tiles
        
        // Insert A matrix tiles
        for (int i = 0; i < m_aug_total; i += mb) {
            int tile_i = i / mb;
            int rows_in_tile = std::min(mb, m_aug_total - i);
            
            for (int j = 0; j < n; j += nb) {
                int tile_j = j / nb;
                int cols_in_tile = std::min(nb, n - j);
                
                if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                    // Allocate tile
                    a_tiles.emplace_back(rows_in_tile * cols_in_tile, 0.0);
                    double* tile_data = a_tiles.back().data();
                    
                    // Copy data if it's from our local portion
                    // Check if any rows in this tile belong to our rank
                    int my_global_start = global_offset;
                    int my_global_end = global_offset + m_local;
                    
                    if (i < m_total && i < my_global_end && (i + rows_in_tile) > my_global_start) {
                        // Calculate which rows of the tile we own
                        int tile_row_start = std::max(0, my_global_start - i);
                        int tile_row_end = std::min(rows_in_tile, my_global_end - i);
                        
                        for (int r = tile_row_start; r < tile_row_end; ++r) {
                            int global_row = i + r;
                            int local_row = global_row - my_global_start;
                            for (int c = 0; c < cols_in_tile; ++c) {
                                tile_data[r * cols_in_tile + c] = 
                                    local_a_data[local_row * n + j + c];
                            }
                        }
                    }
                    
                    // Fill regularization rows if needed
                    if (i + rows_in_tile > m_total) {
                        int reg_start_row = std::max(0, m_total - i);
                        for (int r = reg_start_row; r < rows_in_tile; ++r) {
                            int global_row = i + r;
                            if (global_row >= m_total && global_row < m_aug_total) {
                                int diag_idx = global_row - m_total;
                                if (diag_idx >= j && diag_idx < j + cols_in_tile) {
                                    int local_col = diag_idx - j;
                                    tile_data[r * cols_in_tile + local_col] = std::sqrt(alpha);
                                }
                            }
                        }
                    }
                    
                    A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_data, cols_in_tile);
                }
            }
            
            // Insert b vector tile for this row
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                b_tiles.emplace_back(rows_in_tile, 0.0);
                double* b_tile_data = b_tiles.back().data();
                
                // Copy data if it's from our local portion
                int my_global_start = global_offset;
                int my_global_end = global_offset + m_local;
                
                if (i < m_total && i < my_global_end && (i + rows_in_tile) > my_global_start) {
                    // Calculate which rows of the tile we own
                    int tile_row_start = std::max(0, my_global_start - i);
                    int tile_row_end = std::min(rows_in_tile, my_global_end - i);
                    
                    for (int r = tile_row_start; r < tile_row_end; ++r) {
                        int global_row = i + r;
                        int local_row = global_row - my_global_start;
                        b_tile_data[r] = local_b_data[local_row];
                    }
                }
                
                b_aug.tileInsert(tile_i, 0, slate::HostNum, b_tile_data, rows_in_tile);
            }
        }

        
        MPI_Barrier(comm);
        if (mpi_rank == 0) {
            std::cerr << "\nSolving augmented system with QR..." << std::endl;
            std::cerr.flush();
        }
        
        // SOLVE WITH QR DECOMPOSITION
        // The QR factorization will work correctly because:
        // 1. A_aug knows it's row-major (layout_ = RowMajor)
        // 2. All tiles inherit this layout when created
        // 3. BLAS operations will use the correct increment/stride
        slate::gels(A_aug, b_aug);
        
        if (mpi_rank == 0) {
            std::cerr << "QR solve completed." << std::endl;
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
