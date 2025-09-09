#include <slate/slate.hh>
#include <mpi.h>
#include <iostream>
#include <cstdio>

extern "C" {

void slate_augmented_qr(double* local_a, double* local_b, int m, int n, int lld, int debug) {
    
    // -------- MPI --------

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int mpi_number_of_nodes = m / lld;  // m always integer multiple of lld
    int mpi_sub_size = mpi_size / mpi_number_of_nodes;
    
    // -------- PROCESS GRID AND TILE SIZE --------
    
    auto ceil_div = [](int64_t a, int64_t b) -> int64_t { return (a + b - 1) / b; };

    int num_nodes = m / lld;          // exact by your design
    int p = num_nodes;                // rows split by node
    int q = mpi_size / num_nodes;     // ranks per node (column dimension)
    int mb = lld;                     // one tile-row per node
    int64_t mt = num_nodes;           // known without computing

    // Heuristic near-square starting point for nb
    int64_t tile_area = 256 * 256;
    double nb_ideal = std::sqrt(double(tile_area) / double(mb));
    int nb = std::max(1, std::min<int>(n, int(std::llround(nb_ideal))));

    // Enforce QR constraint mt >= nt  <=>  nb >= ceil(n/mt)
    nb = std::max(nb, int(std::max<int64_t>(1, ceil_div(n, mt))));

    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Solver ===" << std::endl;
        std::cerr << "Global A size: " << m << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
    }
    
    try {
        // -------- CREATE SLATE MATRICES --------
        slate::Matrix<double> A(m, n, mb, nb, slate::GridOrder::Row, p, q, MPI_COMM_WORLD);
        slate::Matrix<double> b(m, 1, mb, 1,  slate::GridOrder::Row, p, q, MPI_COMM_WORLD);
        
        // TODO: only cpu tiles pointing directly to fitsnap shared array for now
        // for gpu support use insertLocalTiles( slate::Target::Devices ) instead
        // and sync fitsnap shared array to slate gpu tile memory
        
        // -------- INSERT A MATRIX TILES --------
        for ( int j = 0; j < A.nt (); ++j)
          for ( int i = 0; i < A.mt (); ++i)
            if (A.tileIsLocal( i, j )) {
                int64_t offset = int64_t(j) * int64_t(nb) * int64_t(lld);
              
                //std::fprintf(stderr, "rank %d i %d j %d offset %d\n", mpi_rank, i, j, offset);
              
                A.tileInsert( i, j, local_a + offset, lld );
              
            }
            
        // -------- INSERT B VECTOR TILES --------
        for ( int i = 0; i < b.mt(); ++i)
          if (b.tileIsLocal( i, 0 )) {
          
            const int offset = (i % mpi_sub_size)*mb;

            //std::fprintf(stderr, "rank %d i %d offset %d\n", mpi_rank, i, offset);

            b.tileInsert( i, 0, local_b + offset, lld );
          
          }
            
        // Make sure every node/rank done building global matrix
        // Doesnt seem to be needed only the barrier after the QR is needed
        // 	slate::least_squares_solve(A, b) is collective and internally synchronized. SLATE’s QR path triggers plenty of MPI collectives (broadcasts, reductions, etc.). Even if one rank reaches the call earlier, it will block at the first collective until everyone is in. That implicitly synchronizes construction + entry to the solve. Hence a barrier before the call is unnecessary for correctness. [chatgpt 5]
        // MPI_Barrier(MPI_COMM_WORLD);
        
        // -------- DEBUG --------

        if (debug) {
            slate::Options opts = {
              { slate::Option::PrintVerbose, 4 },
              { slate::Option::PrintPrecision, 3 },
              { slate::Option::PrintWidth, 7 }
            };
        
            slate::print("A", A, opts);
            slate::print("b", b, opts);
        }
        
        // -------- LEAST SQUARES AUGMENTED QR --------
        slate::least_squares_solve(A, b);
        MPI_Barrier(MPI_COMM_WORLD);
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

} // extern "C"
