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
        
    int p = mpi_size;
    int q = 1;
    int mb = m / mpi_size;
    int nb = n;
    
    // CRITICAL: For QR decomposition, tile rows (mb) must be >= number of columns (n)
    // This is because the QR process needs to handle all n Householder reflectors
    // If mb < n, the tpqrt function will fail with assertion A1.mb() >= k
    if (mb < n) {
        // Adjust tile size to be at least n
        mb = std::max(n, m / mpi_size);
        // If we can't fit n rows in a tile with current process count,
        // we need to reduce the process grid or increase tile size
        if (mpi_rank == 0) {
            std::cerr << "WARNING: Adjusting tile size from " << m/mpi_size
                      << " to " << mb << " to satisfy QR requirements (mb >= n)" << std::endl;
        }
    }
    
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Solver ===" << std::endl;
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
        // Doesnt seem to be needed only the barrier after the QR is needed
        // 	slate::least_squares_solve(A, b) is collective and internally synchronized. SLATE’s QR path triggers plenty of MPI collectives (broadcasts, reductions, etc.). Even if one rank reaches the call earlier, it will block at the first collective until everyone is in. That implicitly synchronizes construction + entry to the solve. Hence a barrier before the call is unnecessary for correctness. [chatgpt 5]

        // MPI_Barrier(MPI_COMM_WORLD);
        
        slate::Options opts = {
          { slate::Option::PrintVerbose, 4 },
          { slate::Option::PrintPrecision, 3 },
          { slate::Option::PrintWidth, 7 }
        };
    
        slate::print("A", A, opts);
        slate::print("b", b, opts);
        
        // Solve using QR decomposition
        slate::least_squares_solve(A, b);
        MPI_Barrier(MPI_COMM_WORLD);

        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
