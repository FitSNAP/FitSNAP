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
        std::cerr << "Using RowMajorMatrix with ZERO-COPY of FitSNAP arrays" << std::endl;
        std::cerr.flush();
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
        std::cerr << "  Original: " << m_total << " x " << n << std::endl;
        std::cerr << "  Augmented: " << m_aug_total << " x " << n << std::endl;
        std::cerr << "  Process grid: " << p << " x " << q << std::endl;
        std::cerr << "  Tile sizes: " << mb << " x " << nb << std::endl;
        std::cerr.flush();
    }
    
    try {
        // Create ROW-MAJOR SLATE matrices using our custom class
        RowMajorMatrix<double> A_aug(m_aug_total, n, mb, nb, 
                                     slate::GridOrder::Col, p, q, comm);
        
        // b vector is still column-major (it's a single column)
        auto b_aug = slate::Matrix<double>(m_aug_total, 1, mb, 1,
                                           slate::GridOrder::Col, p, q, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "Matrix A_aug layout: " << 
                (A_aug.layout() == slate::Layout::RowMajor ? "RowMajor" : "ColMajor") << std::endl;
            std::cerr << "Inserting tiles with ZERO-COPY from row-major data..." << std::endl;
            std::cerr.flush();
        }
        
        // INSERT ALL TILES that this rank owns
        // SLATE expects all tiles to exist, even if we don't have data for them
        int mt_total = (m_aug_total + mb - 1) / mb;  // Total number of tile rows
        int nt_total = (n + nb - 1) / nb;  // Total number of tile columns
        
        // Storage for tiles we need to allocate
        std::vector<std::vector<double>> allocated_tiles;
        std::vector<std::vector<double>> allocated_b_tiles;
        
        // Insert all A matrix tiles this rank owns
        for (int tile_i = 0; tile_i < mt_total; ++tile_i) {
            for (int tile_j = 0; tile_j < nt_total; ++tile_j) {
                if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                    int tile_start_row = tile_i * mb;
                    int tile_end_row = std::min((tile_i + 1) * mb, m_aug_total);
                    int rows_in_tile = tile_end_row - tile_start_row;
                    
                    int tile_start_col = tile_j * nb;
                    int tile_end_col = std::min((tile_j + 1) * nb, n);
                    int cols_in_tile = tile_end_col - tile_start_col;
                    
                    // Check if this tile contains actual data from FitSNAP
                    if (tile_start_row < m_total && 
                        tile_start_row >= m_offsets[mpi_rank] && 
                        tile_start_row < m_offsets[mpi_rank + 1]) {
                        // This tile contains our local data - use zero-copy
                        int local_i = tile_start_row - m_offsets[mpi_rank];
                        double* tile_data = &local_a_data[local_i * n + tile_start_col];
                        A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_data, n);
                    }
                    else {
                        // This tile is either regularization or not our data - allocate
                        // We need to allocate exactly what this tile needs, not the full row
                        allocated_tiles.emplace_back(rows_in_tile * cols_in_tile, 0.0);
                        double* tile_data = allocated_tiles.back().data();
                        
                        // If this contains regularization rows, fill them
                        if (tile_end_row > m_total) {
                            int reg_start = std::max(0, m_total - tile_start_row);
                            for (int r = reg_start; r < rows_in_tile; ++r) {
                                int global_row = tile_start_row + r;
                                if (global_row >= m_total && global_row < m_aug_total) {
                                    int diag_idx = global_row - m_total;
                                    if (diag_idx >= tile_start_col && diag_idx < tile_end_col) {
                                        // In the tile's local coordinates
                                        int local_col = diag_idx - tile_start_col;
                                        tile_data[r * cols_in_tile + local_col] = std::sqrt(alpha);
                                    }
                                }
                            }
                        }
                        
                        // For allocated tiles, stride is cols_in_tile since we allocated exactly that
                        A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_data, cols_in_tile);
                    }
                }
            }
        }
        
        // Insert all b vector tiles this rank owns
        for (int tile_i = 0; tile_i < mt_total; ++tile_i) {
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                int tile_start_row = tile_i * mb;
                int tile_end_row = std::min((tile_i + 1) * mb, m_aug_total);
                int rows_in_tile = tile_end_row - tile_start_row;
                
                // Check if this tile contains actual data from FitSNAP
                if (tile_start_row < m_total && 
                    tile_start_row >= m_offsets[mpi_rank] && 
                    tile_start_row < m_offsets[mpi_rank + 1]) {
                    // This tile contains our local data
                    int local_i = tile_start_row - m_offsets[mpi_rank];
                    double* tile_data = &local_b_data[local_i];
                    b_aug.tileInsert(tile_i, 0, slate::HostNum, tile_data, rows_in_tile);
                }
                else {
                    // Allocate zero tile for b
                    allocated_b_tiles.emplace_back(rows_in_tile, 0.0);
                    b_aug.tileInsert(tile_i, 0, slate::HostNum, allocated_b_tiles.back().data(), rows_in_tile);
                }
            }
        }

        
        MPI_Barrier(comm);
        if (mpi_rank == 0) {
            std::cerr << "\nSolving augmented system with QR..." << std::endl;
            std::cerr << "All matrix operations use ZERO-COPY row-major data" << std::endl;
            std::cerr.flush();
        }
        
        // SOLVE WITH QR DECOMPOSITION
        // The QR factorization will work correctly because:
        // 1. A_aug knows it's row-major (layout_ = RowMajor)
        // 2. All tiles inherit this layout when created
        // 3. BLAS operations will use the correct increment/stride
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
            std::cerr << "\nRidge regression solved successfully with ZERO-COPY!" << std::endl;
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
