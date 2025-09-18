#include <slate/slate.hh>
#include <mpi.h>

#include <cstdint>
#include <cmath>
#include <utility>
#include <limits>

#include <iostream>
#include <cstdio>



extern "C" {

constexpr int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

void slate_augmented_qr(double* local_a, double* local_b, int64_t m, int64_t n, int64_t lld, int debug) {
    
    // -------- MPI --------

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int mpi_number_of_nodes = m / lld;  // m always integer multiple of lld
    int mpi_sub_size = mpi_size / mpi_number_of_nodes;
    
    // -------- PROCESS GRID AND TILE SIZE --------

    // orig.mt() <= 1 || orig.nt() <= 1 || orig.tileMb(0) == orig.tileNb(0)
    // one of three must hold:
    // 1.	only one tile row (mt <= 1),
    // 2.	or only one tile column (nt <= 1),
    // 3.	or square tiles (mb == nb).
    
    int64_t mb, nb, p, q = 1;
    
    // Find the largest nt such that nt*nt <= mpi_sub_size
    int64_t nt_start = 1;
    while ((nt_start + 1) * (nt_start + 1) <= mpi_sub_size) ++nt_start;

    for (int64_t nt = nt_start; nt >= 1; --nt) {
        if (mpi_sub_size % nt) continue;
        int64_t size = ceil_div64(n, nt);
        mb = nb = size;
        q = nt;
        p = mpi_size / q;

        if (mpi_rank == 0)
            std::fprintf(stderr, "*** nt %lld p %lld q %lld tile %lld x %lld (%lld bytes)\n", nt, p, q, nb, nb, nb*nb*8);

        if( p*size > m ) break; // make sure to cover matrix at least once
        if( size*size*sizeof(double) > 16*1024*1024 ) break; // pick biggest tile <= 16MB

    }
    
    // Enforce QR constraint mt >= nt  <=>  nb >= ceil(n/mt)
    //nb = std::max(nb, int(std::max<int64_t>(1, ceil_div(n, mt))));

    if (mpi_rank == 0) {
        std::cerr << "\n---------------- SLATE Ridge Solver ----------------" << std::endl;
        std::cerr << "MPI: " << mpi_size << " ranks, ";
        std::cerr            << mpi_number_of_nodes << " node(s), ";
        std::cerr            << mpi_sub_size << " ranks/node" << std::endl;
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
        std::cerr << "Matrix size: " << m << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "----------------------------------------------------" << std::endl;
    }
    
    try {
        // -------- CREATE SLATE MATRICES --------
        slate::Matrix<double> A(m, n, mb, nb, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        slate::Matrix<double> b(m, 1, mb, 1,  slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        
        // TODO: only cpu tiles pointing directly to fitsnap shared array for now
        // for gpu support use insertLocalTiles( slate::Target::Devices ) instead
        // and sync fitsnap shared array to slate gpu tile memory
        
        // -------- INSERT A MATRIX TILES --------
        for ( int64_t j = 0; j < A.nt (); ++j)
          for ( int64_t i = 0; i < A.mt (); ++i)
            if (A.tileIsLocal( i, j )) {
                const int64_t offset = i * mb + j * nb * lld;
              
                // std::fprintf(stderr, "rank %d i %lld j %lld offset %lld\n", mpi_rank, i, j, offset);
              
                A.tileInsert( i, j, local_a + offset, lld );
              
            }
            
        // -------- INSERT B VECTOR TILES --------
        for ( int64_t i = 0; i < b.mt(); ++i)
          if (b.tileIsLocal( i, 0 )) {
          
            const int64_t offset = i * mb;

            // std::fprintf(stderr, "rank %d i %lld offset %lld\n", mpi_rank, i, offset);

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
