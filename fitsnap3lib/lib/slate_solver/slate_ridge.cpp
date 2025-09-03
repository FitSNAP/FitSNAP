#include <slate/slate.hh>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <cstring>
#include <map>
#include <utility>

// Create a RowMajorMatrix class that inherits from slate::Matrix
template<typename scalar_t>
class RowMajorMatrix : public slate::Matrix<scalar_t> {
public:
    RowMajorMatrix(int64_t m, int64_t n, int64_t mb, int64_t nb,
                   slate::GridOrder order, int p, int q, MPI_Comm mpi_comm)
        : slate::Matrix<scalar_t>(m, n, mb, nb, order, p, q, mpi_comm) {
        // Set the layout to RowMajor
        this->layout_ = slate::Layout::RowMajor;
    }
};

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int m, int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Gather all local sizes to compute offsets
    // Only rank 0 within each node has non-zero m_local
    std::vector<int> all_m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, all_m_locals.data(), 1, MPI_INT, comm);
    
    // Verify that m matches the sum of all local sizes
    int m_computed = 0;
    std::vector<int> row_offsets(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; i++) {
        m_computed += all_m_locals[i];
        row_offsets[i + 1] = row_offsets[i] + all_m_locals[i];
    }
    
    if (m_computed != m && mpi_rank == 0) {
        std::cerr << "WARNING: m_computed (" << m_computed << ") != m (" << m << ")" << std::endl;
    }
    
    // Use the passed-in m as the authoritative global size
    int m_total = m;
    
    // Debug output
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Solver (Row-Major, Zero-Copy) ===" << std::endl;
        std::cerr << "Global dimensions: " << m_total << " x " << n << std::endl;
        std::cerr << "Ridge parameter (alpha): " << alpha << std::endl;
        std::cerr << "MPI ranks: " << mpi_size << std::endl;
        std::cerr << "Tile size: " << tile_size << std::endl;
        std::cerr << "Row distribution:" << std::endl;
        for (int i = 0; i < mpi_size; i++) {
            if (all_m_locals[i] > 0) {
                std::cerr << "  Rank " << i << ": rows " << row_offsets[i] 
                         << "-" << (row_offsets[i+1]-1) 
                         << " (" << all_m_locals[i] << " rows)" << std::endl;
            }
        }
    }
    
    // For ridge regression, solve the augmented system:
    // [  A  ]     [  b  ]
    // [√α*I ] x = [ 0  ]
    int m_aug = m_total + n;
    
    // Use appropriate tile size - needs to be large enough for QR
    int mb, nb;
    
    // For small problems, use larger tiles to avoid QR issues
    if (m_total <= tile_size) {
        // Use the augmented size directly for small problems
        mb = m_total + n;  // Full augmented height
        nb = n;            // Full width
    } else {
        // Use the requested tile size for larger problems
        mb = tile_size;
        nb = std::min(tile_size, n);
    }
    
    // Ensure minimum tile size for numerical stability in QR
    // The tile must be at least as tall as it is wide for QR to work
    if (mb < nb) {
        mb = nb;
    }
    
    // Create process grid for SLATE's 2D block-cyclic distribution
    // For multi-node shared memory setup, we want a column-major grid
    // to align with how data is distributed (contiguous rows per node)
    int p = mpi_size, q = 1;  // Column vector process grid
    
    // Only use 2D grid if all processes have data
    bool all_have_data = true;
    for (int i = 0; i < mpi_size; i++) {
        if (all_m_locals[i] == 0 && i != 0) {
            all_have_data = false;
            break;
        }
    }
    
    if (all_have_data) {
        // Standard 2D grid when all processes have data
        for (int i = (int)std::sqrt(mpi_size); i >= 1; i--) {
            if (mpi_size % i == 0) {
                p = i;
                q = mpi_size / i;
                break;
            }
        }
    }
    
    if (mpi_rank == 0) {
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
        std::cerr << "Augmented system: " << m_aug << " x " << n << std::endl;
        std::cerr << "Tile sizes: " << mb << " x " << nb << std::endl;
    }
    
    try {
        // Create SLATE matrices with ROW-MAJOR layout
        RowMajorMatrix<double> A_aug(m_aug, n, mb, nb, slate::GridOrder::Col, p, q, comm);
        slate::Matrix<double> b_aug(m_aug, 1, mb, 1, slate::GridOrder::Col, p, q, comm);
        
        // Each process with data inserts tiles for its portion
        // In shared memory architecture, only rank 0 within each node has m_local > 0
        int my_row_start = row_offsets[mpi_rank];
        int my_row_end = row_offsets[mpi_rank + 1];
          
            
        // Insert tiles for matrix A using point-to-point communication
        // This approach scales to production size without collective operations
        
        // Structure to track receive operations and where data goes
        struct RecvInfo {
            double* buffer;
            int tile_i, tile_j;
            int row_in_tile;
            int tile_n;
        };
        std::vector<MPI_Request> recv_requests;
        std::vector<RecvInfo> recv_infos;
        std::map<std::pair<int,int>, double*> tile_data_map;
        
        // Phase 1: Tile owners allocate tiles and post receives for rows they need
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                if (A_aug.tileIsLocal(i, j)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_col_start = j * nb;
                    int tile_col_end = std::min((j + 1) * nb, n);
                    
                    int tile_m = tile_row_end - tile_row_start;
                    int tile_n = tile_col_end - tile_col_start;
                    
                    // Allocate tile memory
                    double* tile_data = new double[tile_m * tile_n];
                    std::fill(tile_data, tile_data + tile_m * tile_n, 0.0);
                    tile_data_map[{i,j}] = tile_data;
                    
                    // For data tiles (not regularization)
                    if (tile_row_start < m_total) {
                        for (int ti = 0; ti < tile_m; ++ti) {
                            int global_row = tile_row_start + ti;
                            if (global_row >= m_total) break;
                            
                            // Find which rank owns this row
                            int owner_rank = -1;
                            for (int r = 0; r < mpi_size; r++) {
                                if (all_m_locals[r] > 0 && 
                                    global_row >= row_offsets[r] && 
                                    global_row < row_offsets[r+1]) {
                                    owner_rank = r;
                                    break;
                                }
                            }
                            
                            if (owner_rank == -1) continue;
                            
                            if (owner_rank == mpi_rank) {
                                // I own this row, copy directly
                                if (m_local > 0) {
                                    int local_row = global_row - my_row_start;
                                    for (int tj = 0; tj < tile_n; ++tj) {
                                        int global_col = tile_col_start + tj;
                                        if (global_col < n) {
                                            tile_data[ti * tile_n + tj] = 
                                                local_a_data[local_row * n + global_col];
                                        }
                                    }
                                }
                            } else {
                                // Post receive for this row from owner
                                int tag = global_row * 1000 + tile_col_start;
                                double* row_buffer = new double[tile_n];
                                MPI_Request req;
                                MPI_Irecv(row_buffer, tile_n, MPI_DOUBLE, owner_rank, tag, comm, &req);
                                recv_requests.push_back(req);
                                recv_infos.push_back({row_buffer, i, j, ti, tile_n});
                            }
                        }
                    }
                }
            }
        }
        
        // Phase 2: Row owners send their data to tile owners
        if (m_local > 0) {
            for (int j = 0; j < A_aug.nt(); ++j) {
                for (int i = 0; i < A_aug.mt(); ++i) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_col_start = j * nb;
                    int tile_col_end = std::min((j + 1) * nb, n);
                    
                    if (tile_row_start >= m_total) continue;
                    
                    int tile_n = tile_col_end - tile_col_start;
                    
                    for (int ti = 0; ti < (tile_row_end - tile_row_start); ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row >= m_total) break;
                        
                        if (global_row >= my_row_start && global_row < my_row_end) {
                            // I own this row, check if any other rank needs it for this tile
                            if (!A_aug.tileIsLocal(i, j)) {
                                // Someone else owns this tile, send them the row
                                int local_row = global_row - my_row_start;
                                double* row_segment = new double[tile_n];
                                for (int tj = 0; tj < tile_n; ++tj) {
                                    int global_col = tile_col_start + tj;
                                    if (global_col < n) {
                                        row_segment[tj] = local_a_data[local_row * n + global_col];
                                    } else {
                                        row_segment[tj] = 0.0;
                                    }
                                }
                                
                                int tag = global_row * 1000 + tile_col_start;
                                // Send to the tile owner (need to determine who owns tile i,j)
                                int tile_owner = A_aug.tileRank(i, j);
                                MPI_Send(row_segment, tile_n, MPI_DOUBLE, tile_owner, tag, comm);
                                delete[] row_segment;
                            }
                        }
                    }
                }
            }
        }
        
        // Phase 3: Wait for all receives and copy data into tiles
        if (!recv_requests.empty()) {
            MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
            
            // Copy received data into tiles at the correct positions
            for (const auto& info : recv_infos) {
                double* tile_data = tile_data_map[{info.tile_i, info.tile_j}];
                for (int tj = 0; tj < info.tile_n; ++tj) {
                    tile_data[info.row_in_tile * info.tile_n + tj] = info.buffer[tj];
                }
                delete[] info.buffer;
            }
        }
        
        // Insert all tiles
        for (auto& [indices, tile_data] : tile_data_map) {
            int i = indices.first;
            int j = indices.second;
            int tile_row_start = i * mb;
            int tile_row_end = std::min((i + 1) * mb, m_aug);
            int tile_col_start = j * nb;
            int tile_col_end = std::min((j + 1) * nb, n);
            int tile_n = tile_col_end - tile_col_start;
            
            A_aug.tileInsert(i, j, tile_data, tile_n);
        }
            
        // Insert tiles for vector b using point-to-point communication
        std::vector<MPI_Request> b_recv_requests;
        std::vector<RecvInfo> b_recv_infos;
        std::map<int, double*> b_tile_data_map;
        
        for (int i = 0; i < b_aug.mt(); ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                int tile_m = tile_row_end - tile_row_start;
                
                double* tile_data = new double[tile_m];
                std::fill(tile_data, tile_data + tile_m, 0.0);
                b_tile_data_map[i] = tile_data;
                
                // For data tiles (not regularization)
                if (tile_row_start < m_total) {
                    for (int ti = 0; ti < tile_m; ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row >= m_total) break;
                        
                        // Find which rank owns this row
                        int owner_rank = -1;
                        for (int r = 0; r < mpi_size; r++) {
                            if (all_m_locals[r] > 0 && 
                                global_row >= row_offsets[r] && 
                                global_row < row_offsets[r+1]) {
                                owner_rank = r;
                                break;
                            }
                        }
                        
                        if (owner_rank == -1) continue;
                        
                        if (owner_rank == mpi_rank) {
                            // I own this row, copy directly
                            if (m_local > 0) {
                                int local_row = global_row - my_row_start;
                                tile_data[ti] = local_b_data[local_row];
                            }
                        } else {
                            // Post receive for this element from owner
                            int tag = global_row + 100000;  // Different tag range for b
                            double* elem_buffer = new double[1];
                            MPI_Request req;
                            MPI_Irecv(elem_buffer, 1, MPI_DOUBLE, owner_rank, tag, comm, &req);
                            b_recv_requests.push_back(req);
                            b_recv_infos.push_back({elem_buffer, i, 0, ti, 1});
                        }
                    }
                }
            }
        }
        
        // Send b vector elements to tile owners
        if (m_local > 0) {
            for (int i = 0; i < b_aug.mt(); ++i) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                
                if (tile_row_start >= m_total) continue;
                
                for (int ti = 0; ti < (tile_row_end - tile_row_start); ++ti) {
                    int global_row = tile_row_start + ti;
                    if (global_row >= m_total) break;
                    
                    if (global_row >= my_row_start && global_row < my_row_end) {
                        if (!b_aug.tileIsLocal(i, 0)) {
                            // Someone else owns this tile, send them the element
                            int local_row = global_row - my_row_start;
                            int tag = global_row + 100000;
                            int tile_owner = b_aug.tileRank(i, 0);
                            MPI_Send(&local_b_data[local_row], 1, MPI_DOUBLE, tile_owner, tag, comm);
                        }
                    }
                }
            }
        }
        
        // Wait for b receives and copy data
        if (!b_recv_requests.empty()) {
            MPI_Waitall(b_recv_requests.size(), b_recv_requests.data(), MPI_STATUSES_IGNORE);
            
            for (const auto& info : b_recv_infos) {
                double* tile_data = b_tile_data_map[info.tile_i];
                tile_data[info.row_in_tile] = info.buffer[0];
                delete[] info.buffer;
            }
        }
        
        // Insert b tiles
        for (auto& [i, tile_data] : b_tile_data_map) {
            int tile_row_start = i * mb;
            int tile_row_end = std::min((i + 1) * mb, m_aug);
            int tile_m = tile_row_end - tile_row_start;
            b_aug.tileInsert(i, 0, tile_data, tile_m);
        }
        
        // All ranks (including those without data) handle regularization rows
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                if (A_aug.tileIsLocal(i, j)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_col_start = j * nb;
                    int tile_col_end = std::min((j + 1) * nb, n);
                    
                    int tile_m = tile_row_end - tile_row_start;
                    int tile_n = tile_col_end - tile_col_start;
                    
                    // Handle regularization rows (√α * I)
                    if (tile_row_start >= m_total) {
                        double* reg_tile = new double[tile_m * tile_n];
                        std::fill(reg_tile, reg_tile + tile_m * tile_n, 0.0);
                        
                        double sqrt_alpha = std::sqrt(alpha);
                        for (int ti = 0; ti < tile_m; ++ti) {
                            for (int tj = 0; tj < tile_n; ++tj) {
                                int global_row = tile_row_start + ti;
                                int global_col = tile_col_start + tj;
                                if (global_row >= m_total && global_row - m_total == global_col) {
                                    reg_tile[ti * tile_n + tj] = sqrt_alpha;
                                }
                            }
                        }
                        A_aug.tileInsert(i, j, reg_tile, tile_n);
                    }
                }
            }
        }
        
        // Regularization part of b (zeros)
        for (int i = 0; i < b_aug.mt(); ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                int tile_m = tile_row_end - tile_row_start;
                
                if (tile_row_start >= m_total) {
                    double* zero_tile = new double[tile_m];
                    std::fill(zero_tile, zero_tile + tile_m, 0.0);
                    b_aug.tileInsert(i, 0, zero_tile, tile_m);
                }
            }
        }
        
        MPI_Barrier(comm);
        
        if (mpi_rank == 0) {
            std::cerr << "Solving with QR decomposition..." << std::endl;
        }
        
        // ALL ranks participate in the solve
        slate::gels(A_aug, b_aug);
        
        if (mpi_rank == 0) {
            std::cerr << "QR solve completed. Extracting solution..." << std::endl;
        }
        
        // Extract solution (first n elements of b_aug)
        std::vector<double> solution_local(n, 0.0);
        
        for (int i = 0; i < b_aug.nt() && i * mb < n; ++i) {
            if (b_aug.tileIsLocal(i, 0)) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, n);
                int tile_m = tile_row_end - tile_row_start;
                
                auto T = b_aug(i, 0);
                double* tile_data = T.data();
                
                for (int ti = 0; ti < tile_m; ++ti) {
                    solution_local[tile_row_start + ti] = tile_data[ti];
                }
            }
        }
        
        // Reduce the solution to all processes
        MPI_Allreduce(solution_local.data(), solution, n, MPI_DOUBLE, MPI_SUM, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "Solution extracted successfully." << std::endl;
            std::cerr << "=====================================\n" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
