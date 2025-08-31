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
    
    // Override tileInsert to properly handle row-major tiles
    // When SLATE creates tiles internally, it needs to know they're row-major
    slate::Tile<scalar_t> tileInsert(int64_t i, int64_t j, int device,
                                     scalar_t* data, int64_t ld) {
        // For row-major matrices, we need to ensure the stride is correct
        // The parent class will create the tile with the right layout
        // because we set layout_ = RowMajor in the constructor
        return slate::Matrix<scalar_t>::tileInsert(i, j, device, data, ld);
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
        
        // Optional debug output for specific ranks
        // if (mpi_rank == 16) {
        //     std::cerr << "[Rank 16] local rows: " << local_start_row 
        //              << " to " << local_end_row << std::endl;
        // }
        
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
                    
                    // Optional debug for specific tiles
                    // if (mpi_rank == 16 && tile_i == 0) {
                    //     std::cerr << "[Rank 16] Tile dims: " << A_aug.tileMb(tile_i) 
                    //              << " x " << A_aug.tileNb(tile_j) << std::endl;
                    // }
                    
                    // For row-major: stride is the number of columns (n)
                    // This is ZERO-COPY - we're using the FitSNAP data directly!
                    // The tiles will know they are row-major because A_aug.layout_ is RowMajor
                    // Important: For row-major, stride must be >= actual tile width
                    int tile_stride = n;  // Always use full row stride for row-major
                    
                    // Verify stride is sufficient
                    if (A_aug.layout() == slate::Layout::RowMajor) {
                        int actual_nb = A_aug.tileNb(tile_j);
                        if (tile_stride < actual_nb) {
                            std::cerr << "[Rank " << mpi_rank << "] ERROR: stride " << tile_stride 
                                     << " < tile width " << actual_nb << " for tile (" 
                                     << tile_i << ", " << tile_j << ")" << std::endl;
                            MPI_Abort(comm, 1);
                        }
                    }
                    
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
        // Distribute regularization rows across ranks
        int reg_rows_per_rank = n / mpi_size;
        int reg_rows_remainder = n % mpi_size;
        int my_reg_rows = reg_rows_per_rank + (mpi_rank < reg_rows_remainder ? 1 : 0);
        int my_reg_start = mpi_rank * reg_rows_per_rank + std::min(mpi_rank, reg_rows_remainder);
        
        if (mpi_rank == 0) {
            std::cerr << "Adding regularization rows..." << std::endl;
            std::cerr.flush();
        }
        
        // Allocate memory for regularization tiles IN ROW-MAJOR FORMAT
        double sqrt_alpha = std::sqrt(alpha);
        std::vector<double> reg_tile_data;
        std::vector<double> reg_b_data;
        
        for (int i = 0; i < my_reg_rows; i += mb) {
            int global_row = m_total + my_reg_start + i;
            int tile_i = global_row / mb;
            int rows_in_tile = std::min(mb, my_reg_rows - i);
            
            // Allocate row-major regularization data for the entire row
            reg_tile_data.assign(rows_in_tile * n, 0.0);
            reg_b_data.assign(rows_in_tile, 0.0);
            
            // Set diagonal elements in row-major format
            for (int r = 0; r < rows_in_tile; ++r) {
                int diag_col = my_reg_start + i + r;
                if (diag_col < n) {
                    // Row-major: row r, column diag_col
                    reg_tile_data[r * n + diag_col] = sqrt_alpha;
                }
            }
            
            // Insert regularization tiles
            for (int j = 0; j < n; j += nb) {
                int tile_j = j / nb;
                int cols_in_tile = std::min(nb, n - j);
                
                if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                    // Pointer to the tile within the row-major data
                    // For row r, the data starts at r*n, and tile starts at column j
                    // So for tile at row block tile_i, starting at column j:
                    // We need pointer to reg_tile_data[0*n + j] for first row of tile
                    double* tile_ptr = &reg_tile_data[j];
                    // Stride is n (number of columns) for row-major
                    // Important: For edge tiles, SLATE needs stride >= tile width
                    int tile_stride = n;  // Always use full row stride for row-major
                    A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_ptr, tile_stride);
                }
            }
            
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                b_aug.tileInsert(tile_i, 0, slate::HostNum, reg_b_data.data(), rows_in_tile);
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
