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

void slate_ard_regression(double* local_a, double* local_b, double* local_diag, int64_t m, int64_t n, int64_t lld, int debug) {
    
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
        
        // -------- COMPUTE DIAGONAL OF POSTERIOR COVARIANCE --------
        // After QR, A contains R in its upper n x n block
        // We need diag((R^T R)^(-1))
        
        // Extract the upper n x n block containing R
        // A is m x n, we want rows 0:n-1, cols 0:n-1
        auto A_slice = A.slice(0, n-1, 0, n-1);
        
        // Now create triangular matrix from the square slice
        auto R = slate::TriangularMatrix<double>(slate::Uplo::Upper, slate::Diag::NonUnit, A_slice);
        
        // For each diagonal element i, we need to compute:
        // Σ_ii = sum_k (R^(-1))_ik^2
        // This is equivalent to ||R^(-1) e_i||^2 where e_i is the i-th unit vector
        
        // Create identity matrix for solving
        slate::Matrix<double> I(n, n, nb, nb, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        I.insertLocalTiles();
        slate::set(0.0, I);
        
        // Set diagonal to 1
        for (int64_t j = 0; j < I.nt(); ++j) {
            for (int64_t i = 0; i < I.mt(); ++i) {
                if (I.tileIsLocal(i, j) && i == j) {
                    auto T = I(i, j);
                    for (int64_t jj = 0; jj < T.nb(); ++jj) {
                        for (int64_t ii = 0; ii < T.mb(); ++ii) {
                            if (i * mb + ii == j * nb + jj && i * mb + ii < n) {
                                T.at(ii, jj) = 1.0;
                            }
                        }
                    }
                }
            }
        }
        
        // Solve R X = I, so X = R^(-1)
        // Since R is upper triangular, use triangular solve
        slate::triangular_solve(1.0, R, I);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Compute diagonal: diag_i = sum_j X_ij^2 (row-wise sum of squares)
        // Each rank computes its local contribution
        std::vector<double> local_contrib(n, 0.0);
        std::vector<double> global_diag(n, 0.0);
        
        for (int64_t j = 0; j < I.nt(); ++j) {
            for (int64_t i = 0; i < I.mt(); ++i) {
                if (I.tileIsLocal(i, j)) {
                    auto T = I(i, j);
                    for (int64_t ii = 0; ii < T.mb(); ++ii) {
                        int64_t global_row = i * mb + ii;
                        if (global_row < n) {
                            for (int64_t jj = 0; jj < T.nb(); ++jj) {
                                int64_t global_col = j * nb + jj;
                                if (global_col < n) {
                                    double val = T.at(ii, jj);
                                    local_contrib[global_row] += val * val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Reduce across all ranks to get global diagonal
        MPI_Allreduce(local_contrib.data(), global_diag.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Copy to output array
        for (int64_t i = 0; i < n; ++i) {
            local_diag[i] = global_diag[i];
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void slate_ridge_augmented_qr(double* local_a, double* local_b, int64_t m, int64_t n, int64_t lld, int debug) {
    
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
