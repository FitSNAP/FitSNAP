#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

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
        std::cout << "\n=== SLATE Ridge Regression Solver ===" << std::endl;
        std::cout << "Features (n): " << n << std::endl;
        std::cout << "Alpha: " << alpha << std::endl;
        std::cout << "MPI ranks: " << mpi_size << std::endl;
        std::cout << "Tile size: " << tile_size << std::endl;
    }
    
    // Gather all local row counts to understand the distribution
    std::vector<int> m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, m_locals.data(), 1, MPI_INT, comm);
    
    int m_total = 0;
    for (int i = 0; i < mpi_size; ++i) {
        m_total += m_locals[i];
    }
    
    if (mpi_rank == 0) {
        std::cout << "Total rows (m_total): " << m_total << std::endl;
        std::cout << "\nRow distribution across ranks:" << std::endl;
        for (int i = 0; i < mpi_size; ++i) {
            std::cout << "  Rank " << i << ": " << m_locals[i] << " rows" << std::endl;
        }
    }
    
    // FitSNAP uses 1D row distribution:
    // - Each rank owns m_local complete rows
    // - All ranks have all n columns
    // - Data is stored row-major: row i, column j is at local_a_data[i*n + j]
    
    // For SLATE's fromScaLAPACK with 1D row distribution:
    // We use a Px1 process grid where P = mpi_size
    // This matches FitSNAP's row distribution exactly
    
    int p = mpi_size;  // P processes in rows
    int q = 1;         // 1 process in columns
    
    // Calculate the block size for row distribution
    // In ScaLAPACK terms, each process owns a block of rows
    // The block row size (mb) should be set to handle the distribution
    int mb = tile_size;  // Block size for rows
    int nb = n;          // Each process has all columns, so nb = n
    
    if (mpi_rank == 0) {
        std::cout << "\nSLATE configuration:" << std::endl;
        std::cout << "  Process grid: " << p << " x " << q << std::endl;
        std::cout << "  Block sizes: mb=" << mb << ", nb=" << nb << std::endl;
    }
    
    // Create SLATE matrix A from FitSNAP's row-distributed data
    // Key insight: With Px1 grid and row distribution:
    // - Process grid is column-major (GridOrder::Col)
    // - Each process's data starts at its first row
    // - Local leading dimension is m_local (number of local rows)
    
    auto A = slate::Matrix<double>::fromScaLAPACK(
        m_total, n,                       // Global dimensions
        local_a_data,                     // Local data pointer
        m_local,                          // Local leading dimension (rows this process owns)
        mb, nb,                           // Block sizes
        slate::GridOrder::Col,            // Column-major process grid ordering
        p, q,                             // Process grid dimensions (Px1)
        comm                              // MPI communicator
    );
    
    // Create SLATE vector b from local data
    auto b_vec = slate::Matrix<double>::fromScaLAPACK(
        m_total, 1,                       // Global dimensions (column vector)
        local_b_data,                     // Local data pointer
        m_local,                          // Local leading dimension
        mb, 1,                            // Block sizes
        slate::GridOrder::Col,            // Process grid ordering
        p, q,                             // Process grid (Px1)
        comm                              // MPI communicator
    );
    
    if (mpi_rank == 0) {
        std::cout << "\nComputing normal equations: (A^T A + alpha*I) x = A^T b" << std::endl;
    }
    
    // Allocate distributed matrices for normal equations
    // A^T A is n x n, will be distributed across all processes
    auto AtA = slate::Matrix<double>(n, n, tile_size, tile_size, slate::GridOrder::Col, p, q, comm);
    
    // A^T b is n x 1 vector
    auto Atb = slate::Matrix<double>(n, 1, tile_size, 1, slate::GridOrder::Col, p, q, comm);
    
    // Compute A^T A using distributed GEMM
    // AtA = 1.0 * A^T * A + 0.0 * AtA
    if (mpi_rank == 0) {
        std::cout << "  Computing A^T A..." << std::endl;
    }
    slate::gemm(1.0, A.conj_transpose(), A, 0.0, AtA);
    
    // Compute A^T b
    // Atb = 1.0 * A^T * b + 0.0 * Atb
    if (mpi_rank == 0) {
        std::cout << "  Computing A^T b..." << std::endl;
    }
    slate::gemm(1.0, A.conj_transpose(), b_vec, 0.0, Atb);
    
    // Add ridge regularization to diagonal of AtA
    if (mpi_rank == 0) {
        std::cout << "  Adding regularization (alpha = " << alpha << ") to diagonal..." << std::endl;
    }
    
    // Each process updates the diagonal elements it owns
    for (int64_t i = 0; i < n; ++i) {
        int64_t tile_i = i / tile_size;
        if (AtA.tileIsLocal(tile_i, tile_i)) {
            auto T = AtA(tile_i, tile_i);
            int local_i = i % tile_size;
            if (local_i < T.mb() && local_i < T.nb()) {
                T.at(local_i, local_i) += alpha;
            }
        }
    }
    
    // Solve the system using Cholesky factorization
    // AtA is symmetric positive definite after regularization
    if (mpi_rank == 0) {
        std::cout << "  Solving system using Cholesky factorization..." << std::endl;
    }
    
    // Use SLATE's posv (positive definite solver)
    slate::posv(AtA, Atb);
    
    // Extract the distributed solution from Atb
    // Each process owns some elements of the solution
    std::fill(solution, solution + n, 0.0);
    
    for (int64_t i = 0; i < n; ++i) {
        int64_t tile_i = i / tile_size;
        if (Atb.tileIsLocal(tile_i, 0)) {
            auto T = Atb(tile_i, 0);
            int local_i = i % tile_size;
            if (local_i < T.mb()) {
                solution[i] = T.at(local_i, 0);
            }
        }
    }
    
    // All-reduce to ensure all processes have the complete solution
    MPI_Allreduce(MPI_IN_PLACE, solution, n, MPI_DOUBLE, MPI_SUM, comm);
    
    if (mpi_rank == 0) {
        std::cout << "\nRidge regression solved successfully!" << std::endl;
        std::cout << "Solution vector has " << n << " coefficients" << std::endl;
        
        // Print first few coefficients for verification
        std::cout << "First 5 coefficients: ";
        for (int i = 0; i < std::min(5, n); ++i) {
            std::cout << std::fixed << std::setprecision(6) << solution[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "=====================================\n" << std::endl;
    }
}

} // extern "C"
