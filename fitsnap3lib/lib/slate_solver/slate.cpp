#include <slate/slate.hh>
#include <mpi.h>

#include <cstdint>
#include <cmath>
#include <utility>
#include <limits>
#include <vector>

#include <iostream>
#include <cstdio>



extern "C" {

constexpr int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

void slate_ard_update_sigma(double* local_aw, double* local_sigma, int64_t m, int64_t n, int64_t lld,
                           double alpha, double* lambda, unsigned char* keep_lambda, int64_t n_active, int debug) {
    // Compute sigma = inv(alpha * X.T @ X + diag(lambda)) for active features
    // Uses SLATE distributed operations: herk for X.T @ X, potrf/potri for inversion
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update_sigma ===\n");
        std::fprintf(stderr, "  m=%lld, n=%lld, n_active=%lld, alpha=%.6e\n", m, n, n_active, alpha);
    }
    
    if (n_active == 0) {
        return;  // No active features
    }
    
    // Build index map for active features
    std::vector<int64_t> active_indices;
    active_indices.reserve(n_active);
    for (int64_t i = 0; i < n; ++i) {
        if (keep_lambda[i]) {
            active_indices.push_back(i);
        }
    }
    
    int mpi_number_of_nodes = m / lld;
    int mpi_sub_size = mpi_size / mpi_number_of_nodes;
    
    // Process grid setup (following ridge solver pattern)
    int64_t mb, nb, p, q = 1;
    int64_t nt_start = 1;
    while ((nt_start + 1) * (nt_start + 1) <= mpi_sub_size) ++nt_start;

    for (int64_t nt = nt_start; nt >= 1; --nt) {
        if (mpi_sub_size % nt) continue;
        int64_t size = ceil_div64(n_active, nt);
        mb = nb = size;
        q = nt;
        p = mpi_size / q;
        
        if (p * size > m) break;
        if (size * size * sizeof(double) > 16 * 1024 * 1024) break;
    }
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "  Process grid: %lld x %lld, tile size: %lld x %lld\n", p, q, mb, nb);
    }
    
    try {
        // Create X_active matrix (m x n_active) with only active columns
        slate::Matrix<double> X_active(m, n_active, mb, nb, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        
        // Insert tiles for X_active - extract only active columns from local_a
        for (int64_t j = 0; j < X_active.nt(); ++j) {
            for (int64_t i = 0; i < X_active.mt(); ++i) {
                if (X_active.tileIsLocal(i, j)) {
                    int64_t tile_row = i * mb;
                    int64_t tile_col = j * nb;
                    
                    // Map tile column index to original active column
                    int64_t first_col = (tile_col < n_active) ? active_indices[tile_col] : 0;
                    int64_t offset = tile_row + first_col * lld;
                    
                    X_active.tileInsert(i, j, local_aw + offset, lld);
                }
            }
        }
        
        // Create Hermitian matrix for result: C = alpha * X.T @ X + diag(lambda)
        slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, nb, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        
        // Insert local tiles (SLATE allocates memory for them)
        C.insertLocalTiles();
        
        // Initialize C to zero, then set diagonal to lambda values
        slate::set(0.0, C);
        
        // Set diagonal to lambda[active_indices]
        for (int64_t idx = 0; idx < n_active; ++idx) {
            int64_t tile_i = idx / nb;
            int64_t local_i = idx % nb;
            
            if (C.tileIsLocal(tile_i, tile_i)) {
                auto tile = C(tile_i, tile_i);
                tile.at(local_i, local_i) = lambda[active_indices[idx]];
            }
        }

        // -------- DEBUG --------

        if (debug) {
            slate::Options opts = {
              { slate::Option::PrintVerbose, 4 },
              { slate::Option::PrintPrecision, 3 },
              { slate::Option::PrintWidth, 7 }
            };
        
            slate::print("X_active", X_active, opts);
            slate::print("C", C, opts);
        }

        // Compute C = alpha * X.T @ X + 1.0 * C using SLATE's herk
        // herk computes C = alpha * A * A^H + beta * C
        // We want C = alpha * X^T * X, so pass transpose(X_active) which is n_active x m
        // Then herk gives C = alpha * X^T * (X^T)^H = alpha * X^T * X
        auto X_active_T = transpose(X_active);
        slate::herk(alpha, X_active_T, 1.0, C);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  After herk + diagonal, before inversion\n");
        }
        
        // Compute Cholesky factorization: C = L * L^T
        slate::potrf(C);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  After potrf (Cholesky factorization)\n");
        }
        
        // Compute inverse using Cholesky factorization: C = inv(C)
        slate::potri(C);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  After potri (inversion)\n");
        }
        
        // Gather sigma from distributed tiles to local_sigma on rank 0
        if (mpi_rank == 0) {
            // Initialize to zero
            for (int64_t i = 0; i < n_active * n_active; ++i) {
                local_sigma[i] = 0.0;
            }
            
            // Extract from tiles
            for (int64_t j = 0; j < n_active; ++j) {
                for (int64_t i = j; i < n_active; ++i) {  // Lower triangular
                    int64_t tile_i = i / nb;
                    int64_t tile_j = j / nb;
                    int64_t local_i = i % nb;
                    int64_t local_j = j % nb;
                    
                    if (C.tileIsLocal(tile_i, tile_j)) {
                        auto tile = C(tile_i, tile_j);
                        double val = tile.at(local_i, local_j);
                        // Column-major storage
                        local_sigma[i + j * n_active] = val;
                        // Symmetric: also set upper triangle
                        if (i != j) {
                            local_sigma[j + i * n_active] = val;
                        }
                    }
                }
            }
            
            if (debug) {
                std::fprintf(stderr, "  Sigma extracted (first 5x5):\n");
                for (int64_t i = 0; i < std::min(n_active, int64_t(5)); ++i) {
                    std::fprintf(stderr, "    ");
                    for (int64_t j = 0; j < std::min(n_active, int64_t(5)); ++j) {
                        std::fprintf(stderr, "%.4e ", local_sigma[i + j * n_active]);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
        
        // Broadcast sigma to all ranks
        MPI_Bcast(local_sigma, n_active * n_active, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE ARD sigma error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void slate_ard_update_coeff(double* local_aw, double* local_bw, double* local_coef,
                           int64_t m, int64_t n, int64_t lld,
                           double alpha, unsigned char* keep_lambda, int64_t n_active, 
                           double* sigma, int debug) {
    // Compute coef[keep_lambda] = alpha * sigma @ X[:, keep_lambda].T @ y
    // Uses SLATE gemm for X.T @ y, then local matrix-vector multiply with sigma
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update_coeff ===\n");
        std::fprintf(stderr, "  m=%lld, n=%lld, n_active=%lld, alpha=%.6e\n", m, n, n_active, alpha);
    }
    
    if (n_active == 0) {
        // Zero out all coefficients
        for (int64_t i = 0; i < n; ++i) {
            local_coef[i] = 0.0;
        }
        return;
    }
    
    // Build index map for active features
    std::vector<int64_t> active_indices;
    active_indices.reserve(n_active);
    for (int64_t i = 0; i < n; ++i) {
        if (keep_lambda[i]) {
            active_indices.push_back(i);
        }
    }
    
    int mpi_number_of_nodes = m / lld;
    int mpi_sub_size = mpi_size / mpi_number_of_nodes;
    
    // Process grid setup
    int64_t mb, nb, p, q = 1;
    int64_t nt_start = 1;
    while ((nt_start + 1) * (nt_start + 1) <= mpi_sub_size) ++nt_start;

    for (int64_t nt = nt_start; nt >= 1; --nt) {
        if (mpi_sub_size % nt) continue;
        int64_t size = ceil_div64(n_active, nt);
        mb = nb = size;
        q = nt;
        p = mpi_size / q;
        
        if (p * size > m) break;
        if (size * size * sizeof(double) > 16 * 1024 * 1024) break;
    }
    
    try {
        // Create X_active matrix (m x n_active)
        slate::Matrix<double> X_active(m, n_active, mb, nb, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        
        // Insert tiles for X_active
        for (int64_t j = 0; j < X_active.nt(); ++j) {
            for (int64_t i = 0; i < X_active.mt(); ++i) {
                if (X_active.tileIsLocal(i, j)) {
                    int64_t tile_row = i * mb;
                    int64_t tile_col = j * nb;
                    int64_t first_col = (tile_col < n_active) ? active_indices[tile_col] : 0;
                    int64_t offset = tile_row + first_col * lld;
                    X_active.tileInsert(i, j, local_aw + offset, lld);
                }
            }
        }
        
        // Create y vector (m x 1)
        slate::Matrix<double> y(m, 1, mb, 1, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        
        for (int64_t i = 0; i < y.mt(); ++i) {
            if (y.tileIsLocal(i, 0)) {
                int64_t offset = i * mb;
                y.tileInsert(i, 0, local_bw + offset, lld);
            }
        }
        
        // Create result vector X.T @ y (n_active x 1)
        slate::Matrix<double> XTy(n_active, 1, nb, 1, slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
        XTy.insertLocalTiles();
        slate::set(0.0, XTy);
        
        // Compute XTy = X.T @ y using SLATE gemm
        // gemm: C = alpha * op(A) * op(B) + beta * C
        // We want: XTy = 1.0 * X_active^T * y + 0.0 * XTy
        auto X_active_T = transpose(X_active);
        slate::gemm(1.0, X_active_T, y, 0.0, XTy);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Gather XTy to rank 0
        std::vector<double> xty_vec(n_active, 0.0);
        if (mpi_rank == 0) {
            for (int64_t i = 0; i < n_active; ++i) {
                int64_t tile_i = i / nb;
                int64_t local_i = i % nb;
                
                if (XTy.tileIsLocal(tile_i, 0)) {
                    auto tile = XTy(tile_i, 0);
                    xty_vec[i] = tile.at(local_i, 0);
                }
            }
            
            if (debug) {
                std::fprintf(stderr, "  X.T @ y (first 10): ");
                for (int64_t i = 0; i < std::min(n_active, int64_t(10)); ++i) {
                    std::fprintf(stderr, "%.4e ", xty_vec[i]);
                }
                std::fprintf(stderr, "\n");
            }
        }
        
        // Broadcast xty_vec to all ranks for local computation
        MPI_Bcast(xty_vec.data(), n_active, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Compute sigma @ (X.T @ y) locally on all ranks
        // sigma is n_active x n_active (column-major), xty is n_active x 1
        std::vector<double> result(n_active, 0.0);
        
        for (int64_t i = 0; i < n_active; ++i) {
            for (int64_t j = 0; j < n_active; ++j) {
                result[i] += sigma[i + j * n_active] * xty_vec[j];
            }
        }
        
        // Scale by alpha and assign to full coefficient vector
        for (int64_t i = 0; i < n; ++i) {
            local_coef[i] = 0.0;
        }
        
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t idx = active_indices[i];
            local_coef[idx] = alpha * result[i];
        }
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  Updated coef (first 10): ");
            for (int64_t i = 0; i < std::min(n, int64_t(10)); ++i) {
                std::fprintf(stderr, "%.4e ", local_coef[i]);
            }
            std::fprintf(stderr, "\n");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE ARD coeff error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}



void slate_ridge_augmented_qr(double* local_aw, double* local_bw, int64_t m, int64_t n, int64_t lld, int debug) {
    
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
              
                A.tileInsert( i, j, local_aw + offset, lld );
              
            }
            
        // -------- INSERT B VECTOR TILES --------
        for ( int64_t i = 0; i < b.mt(); ++i)
          if (b.tileIsLocal( i, 0 )) {
          
            const int64_t offset = i * mb;

            // std::fprintf(stderr, "rank %d i %lld offset %lld\n", mpi_rank, i, offset);

            b.tileInsert( i, 0, local_bw + offset, lld );
          
          }
            
        // Make sure every node/rank done building global matrix
        // Doesnt seem to be needed only the barrier after the QR is needed
        // 	slate::least_squares_solve(A, b) is collective and internally synchronized. SLATE's QR path triggers plenty of MPI collectives (broadcasts, reductions, etc.). Even if one rank reaches the call earlier, it will block at the first collective until everyone is in. That implicitly synchronizes construction + entry to the solve. Hence a barrier before the call is unnecessary for correctness. [chatgpt 5]
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
