#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Debug output - use cerr for immediate flushing
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Regression Solver (Augmented QR) ===" << std::endl;
        std::cerr << "Features (n): " << n << std::endl;
        std::cerr << "Alpha: " << alpha << std::endl;
        std::cerr << "MPI ranks: " << mpi_size << std::endl;
        std::cerr << "Tile size: " << tile_size << std::endl;
        std::cerr.flush();
    }
    
    // Gather all local row counts to understand the distribution
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
        std::cerr << "\nRow distribution across ranks:" << std::endl;
        for (int i = 0; i < mpi_size; ++i) {
            std::cerr << "  Rank " << i << ": " << m_locals[i] << " rows (offset: " << m_offsets[i] << ")" << std::endl;
        }
        std::cerr.flush();
    }
    
    // AUGMENTED LEAST SQUARES APPROACH
    // Solve: [ A     ]     [ b ]
    //        [ √α·I  ] x = [ 0 ]
    // Total augmented rows: m_total + n
    
    int m_aug_total = m_total + n;
    
    // Distribute the regularization rows (√α·I) across ranks
    // We'll add them to the last ranks to balance the load
    int reg_rows_per_rank = n / mpi_size;
    int reg_rows_remainder = n % mpi_size;
    
    // Calculate how many regularization rows this rank gets
    int my_reg_rows = reg_rows_per_rank;
    if (mpi_rank < reg_rows_remainder) {
        my_reg_rows++;
    }
    
    // Calculate which regularization columns this rank is responsible for
    int my_reg_start_col = 0;
    for (int i = 0; i < mpi_rank; ++i) {
        my_reg_start_col += reg_rows_per_rank;
        if (i < reg_rows_remainder) {
            my_reg_start_col++;
        }
    }
    
    int m_aug_local = m_local + my_reg_rows;
    
    if (mpi_rank == 0) {
        std::cerr << "\nAugmented system:" << std::endl;
        std::cerr << "  Total augmented rows: " << m_aug_total << std::endl;
        std::cerr << "  Each rank adds ~" << reg_rows_per_rank << " regularization rows" << std::endl;
        std::cerr.flush();
    }
    
    // Debug: Each rank reports its augmented row count
    std::stringstream ss;
    ss << "[Rank " << mpi_rank << "] m_local=" << m_local 
       << ", my_reg_rows=" << my_reg_rows
       << ", m_aug_local=" << m_aug_local 
       << ", reg_start_col=" << my_reg_start_col << std::endl;
    std::cerr << ss.str();
    std::cerr.flush();
    
    // CREATE AUGMENTED SYSTEM IN COLUMN-MAJOR FORMAT FOR SLATE
    // Allocate contiguous memory for the augmented system
    // Use aligned allocation for better performance
    double* aug_a_data = (double*)aligned_alloc(64, m_aug_local * n * sizeof(double));
    double* aug_b_data = (double*)aligned_alloc(64, m_aug_local * sizeof(double));
    
    // Initialize to zero
    std::memset(aug_a_data, 0, m_aug_local * n * sizeof(double));
    std::memset(aug_b_data, 0, m_aug_local * sizeof(double));
    
    if (mpi_rank == 0) {
        std::cerr << "Converting data to column-major format for SLATE..." << std::endl;
        std::cerr.flush();
    }
    
    // Copy original A matrix (convert from row-major to column-major)
    for (int i = 0; i < m_local; ++i) {
        for (int j = 0; j < n; ++j) {
            // Row-major source: local_a_data[i * n + j]
            // Column-major dest: aug_a_data[j * m_aug_local + i]
            aug_a_data[j * m_aug_local + i] = local_a_data[i * n + j];
        }
        // Copy b vector
        aug_b_data[i] = local_b_data[i];
    }
    
    // Add regularization rows (√α on diagonal)
    double sqrt_alpha = std::sqrt(alpha);
    for (int i = 0; i < my_reg_rows; ++i) {
        int local_row = m_local + i;          // Local row index in augmented system
        int global_col = my_reg_start_col + i; // Which column gets √α
        
        // Column-major: aug_a_data[col * m_aug_local + row]
        aug_a_data[global_col * m_aug_local + local_row] = sqrt_alpha;
        // b remains 0 for regularization rows (already initialized)
    }
    
    if (mpi_rank == 0) {
        std::cerr << "\nSetting up SLATE distributed matrices..." << std::endl;
        std::cerr.flush();
    }
    
    // SLATE PROCESS GRID SETUP
    // Use all available MPI ranks in a Px1 grid for row distribution
    int p = mpi_size;
    int q = 1;
    
    // Block sizes for SLATE tiling
    int mb = tile_size;  // Row block size
    int nb = tile_size;  // Column block size
    
    if (mpi_rank == 0) {
        std::cerr << "SLATE configuration:" << std::endl;
        std::cerr << "  Process grid: " << p << " x " << q << std::endl;
        std::cerr << "  Block sizes: mb=" << mb << ", nb=" << nb << std::endl;
        std::cerr << "  Creating distributed matrix (" << m_aug_total << " x " << n << ")" << std::endl;
        std::cerr.flush();
    }
    
    // Synchronize before creating matrices
    MPI_Barrier(comm);
    
    try {
        if (mpi_rank == 0) {
            std::cerr << "Creating SLATE Matrix objects..." << std::endl;
            std::cerr.flush();
        }
        
        // Create SLATE distributed matrices using fromScaLAPACK
        // This creates matrices that are distributed across all MPI ranks
        auto A_aug = slate::Matrix<double>::fromScaLAPACK(
            m_aug_total, n,           // Global dimensions
            aug_a_data,               // Local data pointer (column-major)
            m_aug_local,              // Local leading dimension (number of local rows)
            mb, nb,                   // Block sizes
            slate::GridOrder::Col,    // Column-major process grid
            p, q,                     // Process grid dimensions
            comm                      // MPI communicator
        );
        
        auto b_aug = slate::Matrix<double>::fromScaLAPACK(
            m_aug_total, 1,           // Global dimensions (column vector)
            aug_b_data,               // Local data pointer
            m_aug_local,              // Local leading dimension
            mb, 1,                    // Block sizes
            slate::GridOrder::Col,    // Column-major process grid
            p, q,                     // Process grid dimensions
            comm                      // MPI communicator
        );
        
        if (mpi_rank == 0) {
            std::cerr << "SLATE matrices created successfully" << std::endl;
            std::cerr << "Calling SLATE gels (QR-based least squares solver)..." << std::endl;
            std::cerr.flush();
        }
        
        // Synchronize before solve
        MPI_Barrier(comm);
        
        // SOLVE USING QR DECOMPOSITION
        // slate::gels uses QR decomposition for overdetermined systems
        // This is the most numerically stable approach
        slate::gels(A_aug, b_aug);
        
        if (mpi_rank == 0) {
            std::cerr << "SLATE gels completed successfully" << std::endl;
            std::cerr.flush();
        }
        
        // Extract solution from b_aug
        // After gels, the first n elements of b_aug contain the solution
        // The solution is distributed, so we need to gather it
        
        // First, zero out the solution array
        std::fill(solution, solution + n, 0.0);
        
        // Each rank extracts its portion of the solution
        // The solution is in the first n rows of b_aug
        for (int i = 0; i < std::min(n, m_aug_local); ++i) {
            int global_row = m_offsets[mpi_rank] + i;
            if (global_row < n) {
                solution[global_row] = aug_b_data[i];
            }
        }
        
        if (mpi_rank == 0) {
            std::cerr << "Gathering solution from all ranks..." << std::endl;
            std::cerr.flush();
        }
        
        // All-reduce to gather the complete solution on all ranks
        MPI_Allreduce(MPI_IN_PLACE, solution, n, MPI_DOUBLE, MPI_SUM, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "\nRidge regression (augmented QR) solved successfully!" << std::endl;
            std::cerr << "Solution vector has " << n << " coefficients" << std::endl;
            
            // Print first few coefficients for verification
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
    
    // Clean up aligned memory
    // Do this after SLATE matrices are destroyed (out of scope)
    std::free(aug_a_data);
    std::free(aug_b_data);
    
    if (mpi_rank == 0) {
        std::cerr << "Memory cleanup completed" << std::endl;
        std::cerr.flush();
    }
}

} // extern "C"
