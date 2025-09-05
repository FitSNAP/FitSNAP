#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data,
                          int m, int n, void* comm_ptr, int tile_size) {
    
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Tile size configuration
    // For augmented matrix with regularization rows, ensure tiles are appropriate
    int mb = tile_size;
    int nb = std::min(tile_size, n);  // Column tiles shouldn't exceed n
    
    // Ensure mb divides well into the problem
    mb = std::min(mb, std::max(1, m / mpi_size));
    
    // Create process grid - prefer column distribution for QR
    int p = mpi_size, q = 1;
    // Try to create a more square grid if possible
    for (int i = (int)std::sqrt(mpi_size); i >= 1; i--) {
        if (mpi_size % i == 0) {
            p = i;
            q = mpi_size / i;
            if (p >= q) break;  // Prefer p >= q for QR
        }
    }
    
    p = mpi_size;
    q = 1;
    mb = m / mpi_size;
    nb = n;
    
    if (mpi_rank == 0) {
        std::cerr << "\n=== Clean SLATE Ridge Solver ===" << std::endl;
        std::cerr << "Augmented size: " << m << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
    }
    
    try {
        // Create SLATE matrices 
        slate::Matrix<double> A(m, n, mb, nb, p, q, comm);
        slate::Matrix<double> b(m, 1, mb, 1, p, q, comm);
        
        // Insert A matrix
        for ( int j = 0; j < A.nt (); ++j)
          for ( int i = 0; i < A.mt (); ++i)
            if (A.tileIsLocal( i, j ))
              A.tileInsert( i, j, local_a_data + i*mb, m );
            
        // Insert b vector
        for ( int i = 0; i < b.mt(); ++i)
          if (b.tileIsLocal( i, 0 ))
            b.tileInsert( i, 0, local_b_data + i*mb, m );
            
        // Make sure every node/rank done building global matrix
        MPI_Barrier(MPI_COMM_WORLD);
        
        slate::Options opts = {
          { slate::Option::PrintVerbose, 4 },
          { slate::Option::PrintPrecision, 4 },
          { slate::Option::PrintWidth, 8 }
        };
    
        slate::print("A", A, opts);
        slate::print("b", b, opts);
        
        if (mpi_rank == 0) std::cerr << "Solving with SLATE QR decomposition..." << std::endl;
        // Solve using QR decomposition
        slate::least_squares_solve(A, b);
        if (mpi_rank == 0) std::cerr << "SLATE solve completed successfully" << std::endl;
        slate::print("b", b, opts);

        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
