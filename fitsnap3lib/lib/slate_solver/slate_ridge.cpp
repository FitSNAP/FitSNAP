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

// Structure to hold receive information for b vector
struct RecvInfo {
    double* buffer;
    int tile_i;
    int tile_j;
    int row_in_tile;
    int count;
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
    
    // Build a map of which ranks actually have data and their true offsets
    // This is critical for shared memory architecture where only some ranks have data
    int m_computed = 0;
    std::vector<int> row_offsets(mpi_size + 1, 0);
    std::vector<int> data_rank_map;  // Maps data ranks to their order
    std::map<int, int> rank_to_data_idx;  // Maps rank to its data index
    
    for (int i = 0; i < mpi_size; i++) {
        row_offsets[i + 1] = row_offsets[i] + all_m_locals[i];
        m_computed += all_m_locals[i];
        if (all_m_locals[i] > 0) {
            rank_to_data_idx[i] = data_rank_map.size();
            data_rank_map.push_back(i);
        }
    }
    
    if (m_computed != m && mpi_rank == 0) {
        std::cerr << "WARNING: m_computed (" << m_computed << ") != m (" << m << ")" << std::endl;
    }
    
    // Use the passed-in m as the authoritative global size
    int m_total = m;
    
    // For ridge regression, solve the augmented system:
    // [  A  ]     [  b  ]
    // [√α*I ] x = [ 0  ]
    int m_aug = m_total + n;
    
    // Use appropriate tile size for distribution
    // We want multiple tiles to distribute across processes
    int mb, nb;
    
    // For the augmented system, ensure tiles are large enough for QR
    // QR needs tiles at least as tall as they are wide, and preferably taller
    if (m_aug <= 32) {
        // Very small problem - use a single tile per process if possible
        // But ensure stability for QR
        mb = std::max(8, (m_aug + mpi_size - 1) / mpi_size);
        nb = n;
    } else {
        // Normal case - use requested tile size
        mb = std::min(tile_size, (m_aug + mpi_size - 1) / mpi_size);
        nb = std::min(tile_size, n);
    }
    
    // Ensure tiles are tall enough for QR stability
    // QR with tiles needs mb >= 2*nb for numerical stability
    mb = std::max(mb, 2 * nb);
    
    // Create process grid for SLATE's 2D block-cyclic distribution
    // With shared memory, only some ranks have data (ranks 0 and 2 in 2-node case)
    // We need a grid that works with this distribution
    int p = 1, q = 1;
    
    // Count how many ranks actually have data
    int ranks_with_data = 0;
    for (int i = 0; i < mpi_size; i++) {
        if (all_m_locals[i] > 0) {
            ranks_with_data++;
        }
    }
    
    // For shared memory setup where only some ranks have data,
    // use a 1D distribution along rows (px1 grid)
    if (ranks_with_data < mpi_size) {
        p = mpi_size;
        q = 1;
    } else {
        // Standard 2D grid when all processes have data
        for (int i = (int)std::sqrt(mpi_size); i >= 1; i--) {
            if (mpi_size % i == 0) {
                p = i;
                q = mpi_size / i;
                break;
            }
        }
    }
    
    // Debug output
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Solver ===" << std::endl;
        std::cerr << "Problem size: " << m_total << " x " << n << std::endl;
        std::cerr << "Tile configuration: " << mb << " x " << nb << std::endl;
        std::cerr << "Process grid: " << p << " x " << q << std::endl;
        std::cerr << "Data distribution: " << ranks_with_data << " ranks have data" << std::endl;
        std::cerr << "Data ranks: ";
        for (int r : data_rank_map) {
            std::cerr << r << "(rows " << row_offsets[r] << "-" << row_offsets[r+1] << ") ";
        }
        std::cerr << std::endl;
    }
    
    try {
        // Create SLATE matrices with ROW-MAJOR layout
        RowMajorMatrix<double> A_aug(m_aug, n, mb, nb, slate::GridOrder::Col, p, q, comm);
        slate::Matrix<double> b_aug(m_aug, 1, mb, 1, slate::GridOrder::Col, p, q, comm);
        
        // Each process with data inserts tiles for its portion
        // In shared memory architecture, only rank 0 within each node has m_local > 0
        // CRITICAL: row_offsets gives us the GLOBAL row range, but our local_a_data is indexed from 0
        int my_row_start = row_offsets[mpi_rank];
        int my_row_end = row_offsets[mpi_rank + 1];
        
        // Debug: Print what each rank thinks it owns
        std::cerr << "[Rank " << mpi_rank << "] m_local=" << m_local 
                  << ", owns global rows [" << my_row_start << ", " << my_row_end << ")" << std::endl;
          
            
        // Redistribute matrix A from row-block to 2D block-cyclic using MPI_Alltoallv
        // This is much cleaner and potentially more efficient than point-to-point
        
        // Calculate send counts and displacements for each process
        std::vector<int> send_counts(mpi_size, 0);
        std::vector<int> send_displs(mpi_size, 0);
        std::vector<int> recv_counts(mpi_size, 0);
        std::vector<int> recv_displs(mpi_size, 0);
        
        // First pass: count how much data to send/receive to/from each process
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                int tile_col_start = j * nb;
                int tile_col_end = std::min((j + 1) * nb, n);
                int tile_m = tile_row_end - tile_row_start;
                int tile_n = tile_col_end - tile_col_start;
                
                if (tile_row_start >= m_total) continue;
                
                int tile_owner = A_aug.tileRank(i, j);
                
                // Count data this process needs to send
                if (m_local > 0) {
                    for (int ti = 0; ti < tile_m; ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row >= m_total) break;
                        // Check if this global row belongs to us
                        if (global_row >= my_row_start && global_row < my_row_end) {
                            send_counts[tile_owner] += tile_n;
                        }
                    }
                }
                
                // Count data this process needs to receive
                if (A_aug.tileIsLocal(i, j)) {
                    for (int ti = 0; ti < tile_m; ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row >= m_total) break;
                        
                        // Find owner of this row
                        for (int r = 0; r < mpi_size; r++) {
                            if (all_m_locals[r] > 0 && 
                                global_row >= row_offsets[r] && 
                                global_row < row_offsets[r+1]) {
                                recv_counts[r] += tile_n;
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate displacements
        for (int i = 1; i < mpi_size; i++) {
            send_displs[i] = send_displs[i-1] + send_counts[i-1];
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
        }
        
        int total_send = send_displs[mpi_size-1] + send_counts[mpi_size-1];
        int total_recv = recv_displs[mpi_size-1] + recv_counts[mpi_size-1];
        
        // Allocate send and receive buffers
        // CRITICAL: Even ranks with no data need properly sized buffers for MPI_Alltoallv
        std::vector<double> send_buffer(std::max(1, total_send), 0.0);
        std::vector<double> recv_buffer(std::max(1, total_recv), 0.0);
        
        // Pack send buffer
        std::vector<int> send_offsets(mpi_size, 0);
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                int tile_col_start = j * nb;
                int tile_col_end = std::min((j + 1) * nb, n);
                int tile_m = tile_row_end - tile_row_start;
                int tile_n = tile_col_end - tile_col_start;
                
                if (tile_row_start >= m_total) continue;
                
                int tile_owner = A_aug.tileRank(i, j);
                
                if (m_local > 0) {
                    for (int ti = 0; ti < tile_m; ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row >= m_total) break;
                        if (global_row >= my_row_start && global_row < my_row_end) {
                            int local_row = global_row - my_row_start;
                            // CRITICAL: Ensure local_row is valid for our local array
                            if (local_row >= m_local) {
                                std::cerr << "[Rank " << mpi_rank << "] ERROR: local_row=" << local_row 
                                          << " >= m_local=" << m_local << " for global_row=" << global_row << std::endl;
                                continue;
                            }
                            int offset = send_displs[tile_owner] + send_offsets[tile_owner];
                            for (int tj = 0; tj < tile_n; ++tj) {
                                int global_col = tile_col_start + tj;
                                if (global_col < n) {
                                    send_buffer[offset + tj] = local_a_data[local_row * n + global_col];
                                } else {
                                    send_buffer[offset + tj] = 0.0;
                                }
                            }
                            send_offsets[tile_owner] += tile_n;
                        }
                    }
                }
            }
        }
        
        // Perform the all-to-all redistribution
        MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                      recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                      comm);
        
        // Unpack receive buffer into tiles
        std::map<std::pair<int,int>, double*> tile_data_map;
        std::vector<int> recv_offsets(mpi_size, 0);
        
        for (int j = 0; j < A_aug.nt(); ++j) {
            for (int i = 0; i < A_aug.mt(); ++i) {
                if (A_aug.tileIsLocal(i, j)) {
                    int tile_row_start = i * mb;
                    int tile_row_end = std::min((i + 1) * mb, m_aug);
                    int tile_col_start = j * nb;
                    int tile_col_end = std::min((j + 1) * nb, n);
                    int tile_m = tile_row_end - tile_row_start;
                    int tile_n = tile_col_end - tile_col_start;
                    
                    // Allocate and initialize tile to zero
                    // CRITICAL: Must initialize ALL elements to prevent garbage values
                    double* tile_data = new double[tile_m * tile_n]();
                    std::fill(tile_data, tile_data + tile_m * tile_n, 0.0);
                    
                    if (tile_row_start < m_total) {
                        for (int ti = 0; ti < tile_m; ++ti) {
                            int global_row = tile_row_start + ti;
                            if (global_row >= m_total) break;
                            
                            // Find owner of this row
                            int owner_rank = -1;
                            for (int r = 0; r < mpi_size; r++) {
                                if (all_m_locals[r] > 0 && 
                                    global_row >= row_offsets[r] && 
                                    global_row < row_offsets[r+1]) {
                                    owner_rank = r;
                                    break;
                                }
                            }
                            
                            if (owner_rank >= 0) {
                                if (owner_rank == mpi_rank && m_local > 0) {
                                    // Local copy - only if we actually have data
                                    int local_row = global_row - my_row_start;
                                    if (local_row >= 0 && local_row < m_local) {
                                        for (int tj = 0; tj < tile_n; ++tj) {
                                            int global_col = tile_col_start + tj;
                                            if (global_col < n) {
                                                tile_data[ti * tile_n + tj] = 
                                                    local_a_data[local_row * n + global_col];
                                            } else {
                                                tile_data[ti * tile_n + tj] = 0.0;
                                            }
                                        }
                                    } else {
                                        // Out of bounds - shouldn't happen
                                        std::cerr << "[Rank " << mpi_rank << "] ERROR: local_row=" << local_row 
                                                  << " out of bounds [0, " << m_local << ")" << std::endl;
                                        for (int tj = 0; tj < tile_n; ++tj) {
                                            tile_data[ti * tile_n + tj] = 0.0;
                                        }
                                    }
                                } else if (owner_rank >= 0) {
                                    // Copy from receive buffer
                                    int offset = recv_displs[owner_rank] + recv_offsets[owner_rank];
                                    for (int tj = 0; tj < tile_n; ++tj) {
                                        tile_data[ti * tile_n + tj] = recv_buffer[offset + tj];
                                    }
                                    recv_offsets[owner_rank] += tile_n;
                                } else {
                                    // No owner found - fill with zeros
                                    for (int tj = 0; tj < tile_n; ++tj) {
                                        tile_data[ti * tile_n + tj] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                    
                    // Insert tile
                    A_aug.tileInsert(i, j, tile_data, tile_n);
                }
            }
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
        std::vector<MPI_Request> b_send_requests;
        std::vector<double*> b_send_buffers;
        
        if (m_local > 0) {
            for (int i = 0; i < b_aug.mt(); ++i) {
                int tile_row_start = i * mb;
                int tile_row_end = std::min((i + 1) * mb, m_aug);
                
                if (tile_row_start >= m_total) continue;
                
                int tile_owner = b_aug.tileRank(i, 0);
                
                // Only send if someone else owns this tile
                if (tile_owner != mpi_rank) {
                    for (int ti = 0; ti < (tile_row_end - tile_row_start); ++ti) {
                        int global_row = tile_row_start + ti;
                        if (global_row >= m_total) break;
                        
                        if (global_row >= my_row_start && global_row < my_row_end) {
                            // I own this element, send it to tile owner
                            int local_row = global_row - my_row_start;
                            double* elem_buffer = new double[1];
                            elem_buffer[0] = local_b_data[local_row];
                            b_send_buffers.push_back(elem_buffer);
                            
                            int tag = global_row + 100000;
                            MPI_Request req;
                            MPI_Isend(elem_buffer, 1, MPI_DOUBLE, tile_owner, tag, comm, &req);
                            b_send_requests.push_back(req);
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
        
        // Wait for b sends to complete
        if (!b_send_requests.empty()) {
            MPI_Waitall(b_send_requests.size(), b_send_requests.data(), MPI_STATUSES_IGNORE);
            for (auto* buf : b_send_buffers) {
                delete[] buf;
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
        
        // Solve using QR decomposition
        slate::gels(A_aug, b_aug);
        
        // Extract solution (first n elements of b_aug)
        std::vector<double> solution_local(n, 0.0);
        
        // CRITICAL BUG FIX: Use b_aug.mt() for row tiles, not b_aug.nt() which is column tiles!
        for (int i = 0; i < b_aug.mt() && i * mb < n; ++i) {
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
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
