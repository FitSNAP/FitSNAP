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
        
        // INSERT TILES FOR ORIGINAL DATA (A matrix)
        // Each rank inserts tiles for its local rows
        // For row-major data:
        // - Data is stored row-by-row
        // - Stride (lda) is the number of columns (n) for row-major
        // - Each tile points directly into the FitSNAP shared array
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
                    // Calculate pointer to the start of this tile in row-major data
                    int local_i = i - local_start_row;
                    double* tile_data = &local_a_data[local_i * n + j];
                    
                    // For row-major: stride is the number of columns (n)
                    // This is ZERO-COPY - we're using the FitSNAP data directly!
                    int tile_stride = n;  // Always use full row stride for row-major
                    A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_data, tile_stride);
                }
            }
        }
        
        // INSERT TILES FOR b vector (column-major, single column)
        for (int i = local_start_row; i < local_end_row; i += mb) {
            int tile_i = i / mb;
            int rows_in_tile = std::min(mb, local_end_row - i);
            
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                int local_i = i - local_start_row;
                double* tile_data = &local_b_data[local_i];
                // For column vector, stride is the number of rows in the tile
                b_aug.tileInsert(tile_i, 0, slate::HostNum, tile_data, rows_in_tile);
            }
        }
        
        // ADD REGULARIZATION ROWS  
        // The regularization rows (m_total to m_total+n-1) need to be handled carefully
        // They are distributed in the same block-cyclic pattern as the data rows
        
        if (mpi_rank == 0) {
            std::cerr << "Adding regularization rows..." << std::endl;
            std::cerr << "Regularization rows: " << m_total << " to " << (m_total + n - 1) << std::endl;
            std::cerr.flush();
        }
        
        // Allocate memory for regularization tiles IN ROW-MAJOR FORMAT
        double sqrt_alpha = std::sqrt(alpha);
        
        // Storage for regularization data - must persist through solve
        std::vector<std::vector<double>> reg_A_storage;
        std::vector<std::vector<double>> reg_b_storage;
        
        // Process tiles that overlap with regularization rows
        // Tiles start at row m_total and go through row m_total + n - 1
        int first_reg_tile = m_total / mb;  // First tile that contains regularization rows
        int last_reg_tile = (m_total + n - 1) / mb;  // Last tile that contains regularization rows
        
        for (int tile_i = first_reg_tile; tile_i <= last_reg_tile; ++tile_i) {
            // Determine the actual rows in this tile
            int tile_start_row = tile_i * mb;
            int tile_end_row = std::min((tile_i + 1) * mb, m_aug_total);
            int tile_rows = tile_end_row - tile_start_row;
            
            // Determine which rows in this tile are regularization rows
            int reg_start_in_tile = std::max(0, m_total - tile_start_row);
            int reg_end_in_tile = std::min(tile_rows, m_total + n - tile_start_row);
            int num_reg_rows_in_tile = reg_end_in_tile - reg_start_in_tile;
            
            if (num_reg_rows_in_tile > 0) {
                // This tile contains regularization rows
                // Check if this rank owns any column tiles for this row tile
                bool owns_tiles = false;
                for (int j = 0; j < n; j += nb) {
                    int tile_j = j / nb;
                    if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                        owns_tiles = true;
                        break;
                    }
                }
                
                if (owns_tiles || b_aug.tileRank(tile_i, 0) == mpi_rank) {
                    // Allocate storage for the ENTIRE tile (not just regularization rows)
                    // because SLATE expects full tiles
                    std::vector<double> tile_data(tile_rows * n, 0.0);
                    std::vector<double> b_data(tile_rows, 0.0);
                    
                    // Fill in the regularization part (identity matrix scaled by sqrt_alpha)
                    for (int r = reg_start_in_tile; r < reg_end_in_tile; ++r) {
                        int global_reg_row = tile_start_row + r - m_total;  // 0-based index in regularization
                        if (global_reg_row >= 0 && global_reg_row < n) {
                            // Set diagonal element
                            tile_data[r * n + global_reg_row] = sqrt_alpha;
                        }
                    }
                    
                    // Store the data
                    reg_A_storage.push_back(std::move(tile_data));
                    reg_b_storage.push_back(std::move(b_data));
                    
                    // Get pointers to the stored data
                    double* reg_tile_data = reg_A_storage.back().data();
                    double* reg_b_data = reg_b_storage.back().data();
                    
                    // Insert column tiles for this row tile
                    for (int j = 0; j < n; j += nb) {
                        int tile_j = j / nb;
                        int cols_in_tile = std::min(nb, n - j);
                        
                        if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                            // Pointer to column j in the row-major data
                            double* tile_ptr = reg_tile_data + j;
                            int tile_stride = n;  // Row-major stride
                            A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_ptr, tile_stride);
                        }
                    }
                    
                    // Insert b vector tile
                    if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                        b_aug.tileInsert(tile_i, 0, slate::HostNum, reg_b_data, tile_rows);
                    }
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
