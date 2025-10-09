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

using slate::func::ij_tuple;

constexpr int64_t ceil_div64(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Combined ARD update function - computes both sigma and coefficients
// Expects pre-filtered matrices with only active features
void slate_ard_update(double* local_aw_active, double* local_bw, double* local_sigma, double* local_coef_active,
                     int64_t m, int64_t n_active, int64_t lld,
                     double alpha, double* lambda_active, int debug) {
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "\n=== slate_ard_update ===\n");
        std::fprintf(stderr, "  m=%lld, n_active=%lld, alpha=%.6e\n", m, n_active, alpha);
    }
    
    if (n_active == 0) {
        return;
    }
    
    int mpi_number_of_nodes = m / lld;
    int mpi_sub_size = mpi_size / mpi_number_of_nodes;
    
    // Find optimal tile size
    int64_t mb, nb, nt;
    int64_t nt_start = 1;
    while ((nt_start + 1) * (nt_start + 1) <= mpi_sub_size) ++nt_start;

    for (nt = nt_start; nt >= 1; --nt) {
        if (mpi_sub_size % nt) continue;
        int64_t size = ceil_div64(n_active, nt);
        mb = nb = size;
        
        if (mpi_rank == 0 && debug)
            std::fprintf(stderr, "*** nt %lld tile %lld x %lld (%lld bytes)\n", 
                        nt, nb, nb, nb*nb*8);
        
        if (size*size*sizeof(double) > 16*1024*1024) break; // pick biggest tile <= 16MB
    }
    
    if (nt == 0) nt = 1;
    int64_t mt = ceil_div64(m, mb);
    int64_t mt_node = mt / mpi_number_of_nodes;
    
    // Define tile lambdas
    std::function<int64_t (int64_t)> tileNb = [nb](int64_t) { return nb; };
    std::function<int64_t (int64_t)> tile1 = [](int64_t) { return 1; };
    
    std::function<int64_t (int64_t)> tileMb = [lld, mt_node, mb](int64_t i) {
        if (i % mt_node == mt_node - 1)
            return lld - (mt_node-1)*mb; // remainder tile
        else
            return mb;
    };
    
    std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
    
    std::function<int (slate::func::ij_tuple)> tileRank = [mt_node, nt, mpi_sub_size](slate::func::ij_tuple ij) {
        int64_t i = std::get<0>(ij);
        int64_t j = std::get<1>(ij);
        return i / mt_node * mpi_sub_size + int((i % mt_node)*nt + (j%nt)) % mpi_sub_size;
    };
    
    if (mpi_rank == 0 && debug) {
        std::fprintf(stderr, "*** MPI: %d ranks, %d nodes, %d ranks/node\n",
                    mpi_size, mpi_number_of_nodes, mpi_sub_size);
        std::fprintf(stderr, "*** Matrix %lld x %lld Tile %lld x %lld Grid %lld x %lld\n",
                    m, n_active, mb, nb, mt, nt);
        std::fflush(stderr);
    }
    
    std::fprintf(stderr, "*** [Rank %d] Matrix %lld x %lld Tile %lld x %lld Grid %lld x %lld\n",
                    mpi_rank, m, n_active, mb, nb, mt, nt);
        std::fflush(stderr);
        
    MPI_Barrier(MPI_COMM_WORLD);
    
    try {
        // Create matrices - now working with pre-filtered active features only
        
        slate::Matrix<double> X_active(m, n_active, tileMb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD);
        slate::Matrix<double> y(m, 1, tileMb, tile1, tileRank, tileDevice, MPI_COMM_WORLD);
        
        // Insert X_active and y tiles - data already filtered to active columns
        
        for (int64_t i = 0; i < mt; ++i) {
            for (int64_t j = 0; j < nt; ++j) {
                if (X_active.tileIsLocal(i, j)) {
                    const int64_t offset = (i % mt_node) * mb + j * nb * lld;
                    X_active.tileInsert(i, j, local_aw_active + offset, lld);
                }
            }
            if (y.tileIsLocal(i, 0)) {
                const int64_t offset = (i % mt_node) * mb;
                y.tileInsert(i, 0, local_bw + offset, lld);
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Create C = alpha * X.T @ X + diag(lambda)
        // CRITICAL: Must use same distribution scheme as X_active to avoid herk deadlock
        
        // Use same tileRank function as X_active for consistent distribution
        std::function<int (ij_tuple)> tileRankC = [mt_node, nt, mpi_sub_size](ij_tuple ij) {
            int64_t i = std::get<0>(ij);
            int64_t j = std::get<1>(ij);
            return i / mt_node * mpi_sub_size + int((i % mt_node)*nt + (j%nt)) % mpi_sub_size;
        };
        
        slate::HermitianMatrix<double> C(slate::Uplo::Lower, n_active, tileNb, tileRankC, tileDevice, MPI_COMM_WORLD);
        C.insertLocalTiles();
        
        // Initialize C with diagonal lambda values
        slate::set(0.0, C);
        
        for (int64_t idx = 0; idx < n_active; ++idx) {
            int64_t tile_i = idx / nb;
            int64_t local_i = idx % nb;
            
            if (C.tileIsLocal(tile_i, tile_i)) {
                auto tile = C(tile_i, tile_i);
                tile.at(local_i, local_i) = lambda_active[idx];
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-HERK] About to compute C = alpha * X.T @ X + diag(lambda)\n");
            std::fflush(stderr);
        }
        
        // C = alpha * X.T @ X + C
        auto X_active_T = transpose(X_active);
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Calling slate::herk\n", mpi_rank);
            std::fflush(stderr);
        }
        
        slate::herk(alpha, X_active_T, 1.0, C);
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Completed HERK\n", mpi_rank);
            std::fflush(stderr);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Compute Cholesky factorization and inverse

        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Calling slate::potrf\n", mpi_rank);
            std::fflush(stderr);
        }
        
        slate::potrf(C);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-POTRF] Completed POTRF\n");
            std::fflush(stderr);
        }
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-POTRI] About to compute Cholesky inverse\n");
            std::fflush(stderr);
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Calling slate::potri\n", mpi_rank);
            std::fflush(stderr);
        }
        
        slate::potri(C);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-POTRI] Completed POTRI\n");
            std::fflush(stderr);
        }
        
        // Compute X.T @ y
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-GEMM] About to compute X.T @ y\n");
            std::fflush(stderr);
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Creating XTy matrix\n", mpi_rank);
            std::fflush(stderr);
        }
        
        // Use consistent distribution scheme
        slate::Matrix<double> XTy(n_active, 1, tileNb, tile1, tileRankC, tileDevice, MPI_COMM_WORLD);
        XTy.insertLocalTiles();
        slate::set(0.0, XTy);
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Calling slate::gemm\n", mpi_rank);
            std::fflush(stderr);
        }
        
        slate::gemm(1.0, X_active_T, y, 0.0, XTy);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-GEMM] Completed GEMM\n");
            std::fflush(stderr);
        }
        
        // Compute coef = alpha * C @ XTy
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-HEMM] About to compute coef = alpha * C @ XTy\n");
            std::fflush(stderr);
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Creating coef_active matrix\n", mpi_rank);
            std::fflush(stderr);
        }
        
        // Use consistent distribution scheme
        slate::Matrix<double> coef_active(n_active, 1, tileNb, tile1, tileRankC, tileDevice, MPI_COMM_WORLD);
        coef_active.insertLocalTiles();
        slate::set(0.0, coef_active);
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Calling slate::hemm\n", mpi_rank);
            std::fflush(stderr);
        }
        
        slate::hemm(slate::Side::Left, alpha, C, XTy, 0.0, coef_active);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-HEMM] Completed HEMM\n");
            std::fflush(stderr);
        }
        
        // Gather sigma to rank 0 and broadcast
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-GATHER-SIGMA] About to gather sigma matrix\n");
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Zeroing sigma buffer\n", mpi_rank);
            std::fflush(stderr);
        }
        
        for (int64_t i = 0; i < n_active * n_active; ++i) {
            local_sigma[i] = 0.0;
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Starting tile loop for sigma (n_active=%lld, nt=%lld)\n", mpi_rank, n_active, nt);
            std::fflush(stderr);
        }
        
        int64_t local_tile_count = 0;
        for (int64_t j = 0; j < n_active; ++j) {
            for (int64_t i = j; i < n_active; ++i) {
                int64_t tile_i = i / nb;
                int64_t tile_j = j / nb;
                int64_t local_i = i % nb;
                int64_t local_j = j % nb;
                
                if (C.tileIsLocal(tile_i, tile_j)) {
                    auto tile = C(tile_i, tile_j);
                    double val = tile.at(local_i, local_j);
                    local_sigma[i + j * n_active] = val;
                    if (i != j) {
                        local_sigma[j + i * n_active] = val;
                    }
                    local_tile_count++;
                }
            }
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Finished tile loop (%lld local tiles), entering MPI_Reduce\n", mpi_rank, local_tile_count);
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Reduce sigma to rank 0
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-REDUCE-SIGMA] About to reduce sigma to rank 0\n");
            std::fflush(stderr);
        }
        
        if (mpi_rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, local_sigma, n_active * n_active, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(local_sigma, nullptr, n_active * n_active, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-REDUCE-SIGMA] Completed reduction, about to broadcast\n");
            std::fflush(stderr);
        }
        
        MPI_Bcast(local_sigma, n_active * n_active, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-BCAST-SIGMA] Completed sigma broadcast\n");
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Gather coefficients
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-GATHER-COEF] About to gather coefficients\n");
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Zeroing coef buffer\n", mpi_rank);
            std::fflush(stderr);
        }
        
        for (int64_t i = 0; i < n_active; ++i) {
            local_coef_active[i] = 0.0;
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Starting tile loop for coef\n", mpi_rank);
            std::fflush(stderr);
        }
        
        int64_t local_coef_count = 0;
        for (int64_t i = 0; i < n_active; ++i) {
            int64_t tile_i = i / nb;
            int64_t local_i = i % nb;
            
            if (coef_active.tileIsLocal(tile_i, 0)) {
                auto tile = coef_active(tile_i, 0);
                local_coef_active[i] = tile.at(local_i, 0);
                local_coef_count++;
            }
        }
        
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Finished tile loop (%lld local values), entering MPI_Allreduce\n", mpi_rank, local_coef_count);
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Reduce coefficients to all ranks
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [PRE-ALLREDUCE-COEF] About to allreduce coefficients\n");
            std::fflush(stderr);
        }
        
        MPI_Allreduce(MPI_IN_PLACE, local_coef_active, n_active, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [POST-ALLREDUCE-COEF] Completed allreduce\n");
            std::fflush(stderr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (mpi_rank == 0 && debug) {
            std::fprintf(stderr, "  [FINAL] Updated coef_active (first 10): ");
            for (int64_t i = 0; i < std::min(n_active, int64_t(10)); ++i) {
                std::fprintf(stderr, "%.4e ", local_coef_active[i]);
            }
            std::fprintf(stderr, "\n");
            std::fprintf(stderr, "=== slate_ard_update COMPLETE ===\n\n");
            std::fflush(stderr);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (debug) {
            std::fprintf(stderr, "  [Rank %d] Exiting slate_ard_update normally\n", mpi_rank);
            std::fflush(stderr);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE ARD error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void slate_ridge_augmented_qr(double* local_aw, double* local_bw, int64_t m, int64_t n, int64_t lld, int debug) {
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    int mpi_number_of_nodes = m / lld;
    int mpi_sub_size = mpi_size / mpi_number_of_nodes;
    
    // Find optimal tile size
    int64_t mb, nb, nt;
    int64_t nt_start = 1;
    while ((nt_start + 1) * (nt_start + 1) <= mpi_sub_size) ++nt_start;

    for (nt = nt_start; nt >= 1; --nt) {
        if (mpi_sub_size % nt) continue;
        int64_t size = ceil_div64(n, nt);
        mb = nb = size;
        
        if (mpi_rank == 0 && debug)
            std::fprintf(stderr, "*** nt %lld tile %lld x %lld (%lld bytes)\n", nt, nb, nb, nb*nb*8);
        
        if (size*size*sizeof(double) > 16*1024*1024) break; // pick biggest tile <= 16MB
    }
    
    if (nt == 0) nt = 1;
    int64_t mt = ceil_div64(m, mb);
    int64_t mt_node = mt / mpi_number_of_nodes;
    
    if (mpi_rank == 0) {
        std::cerr << "\n---------------- SLATE Ridge Solver ----------------" << std::endl;
        std::cerr << "MPI: " << mpi_size << " ranks, ";
        std::cerr            << mpi_number_of_nodes << " node(s), ";
        std::cerr            << mpi_sub_size << " ranks/node" << std::endl;
        std::cerr << "Matrix size: " << m << " x " << n << std::endl;
        std::cerr << "Tile size: " << mb << " x " << nb << std::endl;
        std::cerr << "Grid: " << mt << " x " << nt << std::endl;
        std::cerr << "----------------------------------------------------" << std::endl;
    }
    
    try {
        // Define tile lambdas inline (same as before)
        std::function<int64_t (int64_t)> tileNb = [nb](int64_t) { return nb; };
        std::function<int64_t (int64_t)> tile1 = [](int64_t) { return 1; };
        
        std::function<int64_t (int64_t)> tileMb = [lld, mt_node, mb](int64_t i) {
            if (i % mt_node == mt_node - 1)
                return lld - (mt_node-1)*mb; // remainder tile
            else
                return mb;
        };
        
        std::function<int (slate::func::ij_tuple)> tileDevice = [](slate::func::ij_tuple) { return 0; };
        
        std::function<int (slate::func::ij_tuple)> tileRank = [mt_node, nt, mpi_sub_size](slate::func::ij_tuple ij) {
            int64_t i = std::get<0>(ij);
            int64_t j = std::get<1>(ij);
            return i / mt_node * mpi_sub_size + int((i % mt_node)*nt + (j%nt)) % mpi_sub_size;
        };
        
        // Create SLATE matrices with tile lambdas
        slate::Matrix<double> A(m, n, tileMb, tileNb, 
                                tileRank, tileDevice, MPI_COMM_WORLD);
        slate::Matrix<double> b(m, 1, tileMb, tile1,  
                                tileRank, tileDevice, MPI_COMM_WORLD);
        
        // Insert A matrix tiles
        for (int64_t i = 0; i < mt; ++i) {
            for (int64_t j = 0; j < nt; ++j) {
                if (A.tileIsLocal(i, j)) {
                    const int64_t offset = (i % mt_node) * mb + j * nb * lld;
                    A.tileInsert(i, j, local_aw + offset, lld);
                }
            }
        }
            
        // Insert b vector tiles
        for (int64_t i = 0; i < mt; ++i) {
            if (b.tileIsLocal(i, 0)) {
                const int64_t offset = (i % mt_node) * mb;
                b.tileInsert(i, 0, local_bw + offset, lld);
            }
        }
        
        // Debug output
        if (debug) {
            slate::Options opts = {
              {slate::Option::PrintVerbose, 4},
              {slate::Option::PrintPrecision, 3},
              {slate::Option::PrintWidth, 7}
            };
        
            slate::print("A", A, opts);
            slate::print("b", b, opts);
        }
        
        // Least squares solve
        slate::least_squares_solve(A, b);
        MPI_Barrier(MPI_COMM_WORLD);

        if (debug) {
            slate::Options opts = {
              {slate::Option::PrintVerbose, 4},
              {slate::Option::PrintPrecision, 3},
              {slate::Option::PrintWidth, 7}
            };
        
            slate::print("b (solution)", b, opts);
        }

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

} // extern "C"
