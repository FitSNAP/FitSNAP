#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

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
        std::cerr << "\n=== SLATE Ridge Regression Solver ===" << std::endl;
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
    for (int i = 0; i < mpi_size; ++i) {
        m_total += m_locals[i];
    }
    
    if (mpi_rank == 0) {
        std::cerr << "Total rows (m_total): " << m_total << std::endl;
        std::cerr << "\nRow distribution across ranks:" << std::endl;
        for (int i = 0; i < mpi_size; ++i) {
            std::cerr << "  Rank " << i << ": " << m_locals[i] << " rows" << std::endl;
        }
        std::cerr.flush();
    }
    
    // Create the augmented system for ridge regression:
    // [ A     ]     [ b ]
    // [ √α·I  ] x = [ 0 ]
    //
    // This is more numerically stable than normal equations
    
    int m_aug_total = m_total + n;  // Augmented system has additional n rows
    int m_aug_local = m_local;      // This rank's portion of augmented rows
    
    // Determine which ranks will hold regularization rows
    // We'll distribute the n regularization rows across ranks
    int reg_rows_base = n / mpi_size;
    int reg_rows_extra = n % mpi_size;
    int reg_rows_this_rank = reg_rows_base + (mpi_rank < reg_rows_extra ? 1 : 0);
    
    m_aug_local += reg_rows_this_rank;
    
    // Calculate starting index for regularization rows on this rank
    int reg_start_idx;
    if (mpi_rank < reg_rows_extra) {
        reg_start_idx = mpi_rank * (reg_rows_base + 1);
    } else {
        reg_start_idx = reg_rows_extra * (reg_rows_base + 1) + 
                       (mpi_rank - reg_rows_extra) * reg_rows_base;
    }
    
    if (mpi_rank == 0) {
        std::cerr << "\nAugmented system setup:" << std::endl;
        std::cerr << "  Total augmented rows: " << m_aug_total << std::endl;
        std::cerr << "  Regularization rows per rank (approx): " << reg_rows_base << std::endl;
        std::cerr.flush();
    }
    
    // Debug: Each rank reports its augmented row count
    std::stringstream ss;
    ss << "[Rank " << mpi_rank << "] m_local=" << m_local 
       << ", reg_rows=" << reg_rows_this_rank 
       << ", m_aug_local=" << m_aug_local << std::endl;
    std::cerr << ss.str();
    std::cerr.flush();
    
    // Allocate local storage for augmented system in COLUMN-MAJOR format
    // SLATE expects column-major, so we need to transpose our row-major data
    std::vector<double> aug_a_col_major(m_aug_local * n, 0.0);
    std::vector<double> aug_b(m_aug_local, 0.0);
    
    // Copy and transpose the original A matrix data (row-major to column-major)
    for (int i = 0; i < m_local; ++i) {
        for (int j = 0; j < n; ++j) {
            // Row-major: local_a_data[i*n + j]
            // Column-major: aug_a_col_major[j*m_aug_local + i]
            aug_a_col_major[j * m_aug_local + i] = local_a_data[i * n + j];
        }
        aug_b[i] = local_b_data[i];
    }
    
    // Add regularization rows (√α on diagonal)
    double sqrt_alpha = std::sqrt(alpha);
    for (int i = 0; i < reg_rows_this_rank; ++i) {
        int global_reg_col = reg_start_idx + i;  // Which column gets √α
        int local_row = m_local + i;             // Local row index in augmented system
        
        // Set diagonal element to √α
        // Column-major: aug_a_col_major[col * m_aug_local + row]
        aug_a_col_major[global_reg_col * m_aug_local + local_row] = sqrt_alpha;
        
        // Corresponding b value is 0 (already initialized)
    }
    
    // Now create SLATE matrices using the column-major data
    // Use a 1D block-cyclic distribution
    int p = mpi_size;
    int q = 1;
    int mb = tile_size;
    int nb = tile_size;
    
    if (mpi_rank == 0) {
        std::cerr << "\nCreating SLATE matrices:" << std::endl;
        std::cerr << "  Process grid: " << p << " x " << q << std::endl;
        std::cerr << "  Block sizes: mb=" << mb << ", nb=" << nb << std::endl;
        std::cerr.flush();
    }
    
    // All ranks report their status before matrix creation
    MPI_Barrier(comm);
    for (int r = 0; r < mpi_size; ++r) {
        if (mpi_rank == r) {
            std::cerr << "[Rank " << r << "] Creating matrices with m_aug_local=" 
                     << m_aug_local << std::endl;
            std::cerr.flush();
        }
        MPI_Barrier(comm);
    }
    
    // Create augmented A matrix from column-major data
    slate::Matrix<double> A_aug;
    slate::Matrix<double> b_aug;
    
    if (m_aug_local > 0) {
        A_aug = slate::Matrix<double>::fromScaLAPACK(
            m_aug_total, n,                    // Global dimensions
            aug_a_col_major.data(),            // Local data (column-major)
            m_aug_local,                       // Local leading dimension
            mb, nb,                            // Block sizes
            slate::GridOrder::Col,             // Process grid ordering
            p, q,                              // Process grid
            comm                               // MPI communicator
        );
        
        b_aug = slate::Matrix<double>::fromScaLAPACK(
            m_aug_total, 1,                    // Global dimensions
            aug_b.data(),                      // Local data
            m_aug_local,                       // Local leading dimension
            mb, 1,                             // Block sizes
            slate::GridOrder::Col,             // Process grid ordering
            p, q,                              // Process grid
            comm                               // MPI communicator
        );
    } else {
        // Handle case where process has no rows
        A_aug = slate::Matrix<double>::fromScaLAPACK(
            m_aug_total, n,                    // Global dimensions
            nullptr,                           // No local data
            1,                                 // Dummy leading dimension
            mb, nb,                            // Block sizes
            slate::GridOrder::Col,             // Process grid ordering
            p, q,                              // Process grid
            comm                               // MPI communicator
        );
        
        b_aug = slate::Matrix<double>::fromScaLAPACK(
            m_aug_total, 1,                    // Global dimensions
            nullptr,                           // No local data
            1,                                 // Dummy leading dimension
            mb, 1,                             // Block sizes
            slate::GridOrder::Col,             // Process grid ordering
            p, q,                              // Process grid
            comm                               // MPI communicator
        );
    }
    
    if (mpi_rank == 0) {
        std::cerr << "  About to call SLATE least_squares_solve..." << std::endl;
        std::cerr.flush();
    }
    MPI_Barrier(comm);
    
    // Solve the augmented least squares problem using QR decomposition
    // This is more stable than normal equations
    try {
        slate::least_squares_solve(A_aug, b_aug);
        if (mpi_rank == 0) {
            std::cerr << "  SLATE least_squares_solve completed successfully" << std::endl;
            std::cerr.flush();
        }
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        std::cerr.flush();
        MPI_Abort(comm, 1);
    }
    
    // Extract solution from b_aug (first n elements after solve)
    // b_aug now contains the solution in its first n rows
    std::vector<double> local_solution(n, 0.0);
    
    // Each process extracts the parts of the solution it owns
    for (int i = 0; i < n; ++i) {
        // Determine which process owns row i of the solution
        int owner_rank = i % mpi_size;  // Simple cyclic distribution assumption
        
        if (mpi_rank == owner_rank && i / mpi_size < m_aug_local) {
            int local_idx = i / mpi_size;
            if (local_idx < m_aug_local) {
                local_solution[i] = aug_b[local_idx];
            }
        }
    }
    
    // All-reduce to get complete solution on all ranks
    MPI_Allreduce(local_solution.data(), solution, n, MPI_DOUBLE, MPI_SUM, comm);
    
    if (mpi_rank == 0) {
        std::cerr << "\nRidge regression solved successfully!" << std::endl;
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
}

} // extern "C"
