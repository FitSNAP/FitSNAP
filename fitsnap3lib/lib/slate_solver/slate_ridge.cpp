#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Set up 2D process grid (as square as possible)
    int p = static_cast<int>(std::sqrt(mpi_size));
    int q = mpi_size / p;
    while (p * q != mpi_size && p > 1) {
        --p;
        q = mpi_size / p;
    }
    
    // Get total number of rows across all processes
    int m_total;
    MPI_Allreduce(&m_local, &m_total, 1, MPI_INT, MPI_SUM, comm);
    
    // For augmented least squares, we need an augmented system
    // But since each process has its local A and b directly, we can use them
    // For ridge regression with QR, we'll create augmented matrices
    
    // Create augmented matrix locally: [A; sqrt(alpha)*I]
    int m_augmented = m_local + n;  // local rows plus regularization rows
    int m_augmented_total = m_total + n;
    double sqrt_alpha = std::sqrt(alpha);
    
    // Allocate augmented local matrices
    std::vector<double> a_augmented(m_augmented * n);
    std::vector<double> b_augmented(m_augmented);
    
    // Copy local A into augmented matrix
    for (int i = 0; i < m_local; ++i) {
        for (int j = 0; j < n; ++j) {
            a_augmented[i * n + j] = local_a_data[i * n + j];
        }
        b_augmented[i] = local_b_data[i];
    }
    
    // Add regularization part (only some processes get regularization rows)
    // Distribute the n regularization rows across processes
    int row_offset;
    MPI_Scan(&m_local, &row_offset, 1, MPI_INT, MPI_SUM, comm);
    row_offset -= m_local; // Exclusive scan
    
    // Calculate which regularization rows this process owns
    int reg_rows_per_proc = n / mpi_size;
    int extra_reg_rows = n % mpi_size;
    
    int my_reg_start, my_reg_count;
    if (mpi_rank < extra_reg_rows) {
        my_reg_start = mpi_rank * (reg_rows_per_proc + 1);
        my_reg_count = reg_rows_per_proc + 1;
    } else {
        my_reg_start = extra_reg_rows * (reg_rows_per_proc + 1) + 
                       (mpi_rank - extra_reg_rows) * reg_rows_per_proc;
        my_reg_count = reg_rows_per_proc;
    }
    
    // Fill in regularization rows for this process
    for (int i = 0; i < my_reg_count; ++i) {
        int local_row = m_local + i;
        int global_reg_row = my_reg_start + i;
        
        // Zero out the row
        for (int j = 0; j < n; ++j) {
            a_augmented[local_row * n + j] = 0.0;
        }
        // Set diagonal element
        a_augmented[local_row * n + global_reg_row] = sqrt_alpha;
        
        // RHS is zero for regularization
        b_augmented[local_row] = 0.0;
    }
    
    // Update local row count to include regularization rows
    int m_local_aug = m_local + my_reg_count;
    
    // Create SLATE matrices from the augmented data using fromScaLAPACK
    // This creates matrices that point to our data without copying
    int lld_a = m_local_aug;  // local leading dimension for A
    int lld_b = m_local_aug;  // local leading dimension for B
    
    auto A = slate::Matrix<double>::fromScaLAPACK(
        m_augmented_total, n,           // global dimensions
        a_augmented.data(),              // local data pointer
        lld_a,                           // local leading dimension
        tile_size, tile_size,            // tile sizes
        slate::GridOrder::Col,           // column-major process grid
        p, q,                            // process grid dimensions
        comm                             // MPI communicator
    );
    
    // For least squares, BX needs to be max(m, n) x 1
    int max_mn = std::max(m_augmented_total, n);
    
    // Need to allocate properly sized local BX
    int bx_local_rows = (max_mn + mpi_size - 1) / mpi_size;  // ceiling division
    if (mpi_rank == mpi_size - 1) {
        // Last process might have fewer rows
        bx_local_rows = max_mn - mpi_rank * ((max_mn + mpi_size - 1) / mpi_size);
    }
    
    std::vector<double> bx_local(bx_local_rows, 0.0);
    
    // Copy b_augmented into bx_local
    for (int i = 0; i < std::min(m_local_aug, bx_local_rows); ++i) {
        bx_local[i] = b_augmented[i];
    }
    
    auto BX = slate::Matrix<double>::fromScaLAPACK(
        max_mn, 1,                       // global dimensions
        bx_local.data(),                 // local data pointer
        bx_local_rows,                   // local leading dimension
        tile_size, tile_size,            // tile sizes
        slate::GridOrder::Col,           // column-major process grid
        p, q,                            // process grid dimensions
        comm                             // MPI communicator
    );
    
    // Set options for SLATE
    slate::Options opts;
    opts[slate::Option::Target] = slate::Target::HostTask;
    
    // Solve the least squares problem using QR factorization
    slate::least_squares_solve(A, BX, opts);
    
    // Extract solution from BX (first n elements on appropriate processes)
    // The solution is distributed in BX, need to gather it
    std::vector<double> local_solution(n, 0.0);
    
    // Each process extracts its portion of the solution
    int sol_rows_per_proc = n / mpi_size;
    int extra_sol_rows = n % mpi_size;
    
    int my_sol_start, my_sol_count;
    if (mpi_rank < extra_sol_rows) {
        my_sol_start = mpi_rank * (sol_rows_per_proc + 1);
        my_sol_count = sol_rows_per_proc + 1;
    } else {
        my_sol_start = extra_sol_rows * (sol_rows_per_proc + 1) + 
                       (mpi_rank - extra_sol_rows) * sol_rows_per_proc;
        my_sol_count = sol_rows_per_proc;
    }
    
    // Copy from bx_local to local_solution
    for (int i = 0; i < my_sol_count; ++i) {
        if (my_sol_start + i < n && i < bx_local_rows) {
            local_solution[my_sol_start + i] = bx_local[i];
        }
    }
    
    // All-reduce to get complete solution on all processes
    MPI_Allreduce(local_solution.data(), solution, n, MPI_DOUBLE, MPI_SUM, comm);
}

} // extern "C"
