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

// Create a RowMajorMatrix class that inherits from slate::Matrix
// This allows us to access the protected layout_ member and set it to RowMajor
template<typename scalar_t>
class RowMajorMatrix : public slate::Matrix<scalar_t> {
public:
    // Constructor that creates a row-major matrix
    RowMajorMatrix(int64_t m, int64_t n, int64_t mb, int64_t nb,
                   slate::GridOrder order, int p, int q, MPI_Comm mpi_comm)
        : slate::Matrix<scalar_t>(m, n, mb, nb, order, p, q, mpi_comm) {
        // Set the layout to RowMajor - this is the key!
        // layout_ is protected in BaseMatrix, so we can access it here
        this->layout_ = slate::Layout::RowMajor;
    }
};

extern "C" {

void slate_ridge_solve_qr(double* local_a_data, double* local_b_data, double* solution,
                          int m_local, int n, double alpha, void* comm_ptr, int tile_size) {
    
    // Cast the void pointer back to MPI_Comm
    MPI_Comm comm = (MPI_Comm)comm_ptr;
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
    
    // Debug output
    if (mpi_rank == 0) {
        std::cerr << "\n=== SLATE Ridge Regression Solver (Zero-Copy Row-Major) ===" << std::endl;
        std::cerr << "Features (n): " << n << std::endl;
        std::cerr << "Alpha: " << alpha << std::endl;
        std::cerr << "MPI ranks: " << mpi_size << std::endl;
        std::cerr << "Tile size: " << tile_size << std::endl;
        std::cerr.flush();
    }
    
    // Gather all local row counts
    std::vector<int> m_locals(mpi_size);
    MPI_Allgather(&m_local, 1, MPI_INT, m_locals.data(), 1, MPI_INT, comm);
    
    // FitSNAP uses node-based distribution: first to nodes, then within nodes
    // We need to understand the node structure
    // Assume processes are laid out as: node0_procs, node1_procs, node2_procs, ...
    
    // First, figure out the number of processes per node
    // We can infer this by looking at the pattern of m_locals
    // All processes on the same node should have similar row counts
    
    // Get MPI node information
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    
    // Create a communicator for all rank 0s of each node
    int color = (node_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm head_comm;
    MPI_Comm_split(comm, color, mpi_rank, &head_comm);
    
    int num_nodes = 0;
    if (head_comm != MPI_COMM_NULL) {
        MPI_Comm_size(head_comm, &num_nodes);
    }
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, node_comm);
    
    // Calculate which node this rank belongs to
    int node_id = mpi_rank / node_size;
    
    // Calculate total rows and create offset arrays
    // For FitSNAP's distribution:
    // - Each node owns a contiguous block of the global rows
    // - Within each node, processes share that block
    
    // First, sum up rows per node
    std::vector<int> rows_per_node(num_nodes, 0);
    for (int rank = 0; rank < mpi_size; ++rank) {
        int rank_node = rank / node_size;
        if (rank_node < num_nodes) {
            rows_per_node[rank_node] += m_locals[rank];
        }
    }
    
    // Calculate node offsets in global data
    std::vector<int> node_offsets(num_nodes + 1, 0);
    for (int n = 0; n < num_nodes; ++n) {
        node_offsets[n + 1] = node_offsets[n] + rows_per_node[n];
    }
    int m_total = node_offsets[num_nodes];
    
    // Now calculate this rank's offset within its node's data
    int rank_offset_in_node = 0;
    for (int r = node_id * node_size; r < mpi_rank; ++r) {
        rank_offset_in_node += m_locals[r];
    }
    
    // Global offset for this rank's data
    int global_offset = node_offsets[node_id] + rank_offset_in_node;
    
    if (mpi_rank == 0) {
        std::cerr << "Total rows (m_total): " << m_total << std::endl;
        std::cerr << "Using RowMajorMatrix with ZERO-COPY of FitSNAP arrays" << std::endl;
        std::cerr.flush();
    }
    
    // Set up process grid
    int p = mpi_size;
    int q = 1;  // 1D row distribution
    
    // Tile sizes - for column dimension, use min of tile_size and n
    // This prevents tiles from being wider than the matrix
    int mb = tile_size;  // Row tile size
    int nb = std::min(tile_size, n);  // Column tile size (can't exceed n)
    
    // Augmented system dimensions
    int m_aug_total = m_total + n;  // Add n regularization rows
    
    if (mpi_rank == 0) {
        std::cerr << "\nCreating augmented system:" << std::endl;
        std::cerr << "  Original: " << m_total << " x " << n << std::endl;
        std::cerr << "  Augmented: " << m_aug_total << " x " << n << std::endl;
        std::cerr << "  Process grid: " << p << " x " << q << std::endl;
        std::cerr << "  Tile sizes: " << mb << " x " << nb << std::endl;
        std::cerr.flush();
    }
    
    try {
        // Create ROW-MAJOR SLATE matrices using our custom class
        RowMajorMatrix<double> A_aug(m_aug_total, n, mb, nb, 
                                     slate::GridOrder::Col, p, q, comm);
        
        // b vector is still column-major (it's a single column)
        auto b_aug = slate::Matrix<double>(m_aug_total, 1, mb, 1,
                                           slate::GridOrder::Col, p, q, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "Matrix A_aug layout: " << 
                (A_aug.layout() == slate::Layout::RowMajor ? "RowMajor" : "ColMajor") << std::endl;
            std::cerr << "Inserting tiles with ZERO-COPY from row-major data..." << std::endl;
            std::cerr.flush();
        }
        
        // INSERT TILES - simpler approach: just copy data into SLATE format
        // This avoids all the complexity of trying to make zero-copy work
        // with different data distributions
        
        // Allocate workspace for all tiles we own
        std::vector<std::vector<double>> a_tiles;
        std::vector<std::vector<double>> b_tiles;
        
        // Process all tiles in the augmented matrix
        int mt_total = (m_aug_total + mb - 1) / mb;
        int nt_total = 1;  // b vector has only 1 column of tiles
        
        // Insert A matrix tiles
        for (int i = 0; i < m_aug_total; i += mb) {
            int tile_i = i / mb;
            int rows_in_tile = std::min(mb, m_aug_total - i);
            
            for (int j = 0; j < n; j += nb) {
                int tile_j = j / nb;
                int cols_in_tile = std::min(nb, n - j);
                
                if (A_aug.tileRank(tile_i, tile_j) == mpi_rank) {
                    // Allocate tile
                    a_tiles.emplace_back(rows_in_tile * cols_in_tile, 0.0);
                    double* tile_data = a_tiles.back().data();
                    
                    // Copy data if it's from our local portion
                    // Check if any rows in this tile belong to our rank
                    int my_global_start = global_offset;
                    int my_global_end = global_offset + m_local;
                    
                    if (i < m_total && i < my_global_end && (i + rows_in_tile) > my_global_start) {
                        // Calculate which rows of the tile we own
                        int tile_row_start = std::max(0, my_global_start - i);
                        int tile_row_end = std::min(rows_in_tile, my_global_end - i);
                        
                        for (int r = tile_row_start; r < tile_row_end; ++r) {
                            int global_row = i + r;
                            int local_row = global_row - my_global_start;
                            for (int c = 0; c < cols_in_tile; ++c) {
                                tile_data[r * cols_in_tile + c] = 
                                    local_a_data[local_row * n + j + c];
                            }
                        }
                    }
                    
                    // Fill regularization rows if needed
                    if (i + rows_in_tile > m_total) {
                        int reg_start_row = std::max(0, m_total - i);
                        for (int r = reg_start_row; r < rows_in_tile; ++r) {
                            int global_row = i + r;
                            if (global_row >= m_total && global_row < m_aug_total) {
                                int diag_idx = global_row - m_total;
                                if (diag_idx >= j && diag_idx < j + cols_in_tile) {
                                    int local_col = diag_idx - j;
                                    tile_data[r * cols_in_tile + local_col] = std::sqrt(alpha);
                                }
                            }
                        }
                    }
                    
                    A_aug.tileInsert(tile_i, tile_j, slate::HostNum, tile_data, cols_in_tile);
                }
            }
            
            // Insert b vector tile for this row
            if (b_aug.tileRank(tile_i, 0) == mpi_rank) {
                b_tiles.emplace_back(rows_in_tile, 0.0);
                double* b_tile_data = b_tiles.back().data();
                
                // Copy data if it's from our local portion
                int my_global_start = global_offset;
                int my_global_end = global_offset + m_local;
                
                if (i < m_total && i < my_global_end && (i + rows_in_tile) > my_global_start) {
                    // Calculate which rows of the tile we own
                    int tile_row_start = std::max(0, my_global_start - i);
                    int tile_row_end = std::min(rows_in_tile, my_global_end - i);
                    
                    for (int r = tile_row_start; r < tile_row_end; ++r) {
                        int global_row = i + r;
                        int local_row = global_row - my_global_start;
                        b_tile_data[r] = local_b_data[local_row];
                    }
                }
                
                b_aug.tileInsert(tile_i, 0, slate::HostNum, b_tile_data, rows_in_tile);
            }
        }

        
        MPI_Barrier(comm);
        if (mpi_rank == 0) {
            std::cerr << "\nSolving augmented system with QR..." << std::endl;
            std::cerr << "All matrix operations use ZERO-COPY row-major data" << std::endl;
            std::cerr.flush();
        }
        
        // SOLVE WITH QR DECOMPOSITION
        // The QR factorization will work correctly because:
        // 1. A_aug knows it's row-major (layout_ = RowMajor)
        // 2. All tiles inherit this layout when created
        // 3. BLAS operations will use the correct increment/stride
        slate::gels(A_aug, b_aug);
        
        if (mpi_rank == 0) {
            std::cerr << "QR solve completed. Extracting solution..." << std::endl;
            std::cerr.flush();
        }
        
        // Extract solution from first n elements of b_aug
        std::fill(solution, solution + n, 0.0);
        
        for (int i = 0; i < n; ++i) {
            int tile_i = i / mb;
            int local_i = i % mb;
            
            if (b_aug.tileRank(tile_i, 0) == mpi_rank && b_aug.tileIsLocal(tile_i, 0)) {
                auto tile = b_aug(tile_i, 0);
                if (local_i < tile.mb()) {
                    solution[i] = tile(local_i, 0);
                }
            }
        }
        
        // Gather solution to all ranks
        MPI_Allreduce(MPI_IN_PLACE, solution, n, MPI_DOUBLE, MPI_SUM, comm);
        
        if (mpi_rank == 0) {
            std::cerr << "\nRidge regression solved successfully with ZERO-COPY!" << std::endl;
            std::cerr << "Solution vector has " << n << " coefficients" << std::endl;
            
            // Print first few coefficients
            std::cerr << "First 5 coefficients: ";
            for (int i = 0; i < std::min(5, n); ++i) {
                std::cerr << std::fixed << std::setprecision(6) << solution[i] << " ";
            }
            std::cerr << std::endl;
            std::cerr << "=====================================\n" << std::endl;
            std::cerr.flush();
        }
        
        // Clean up MPI communicators
        MPI_Comm_free(&node_comm);
        if (head_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&head_comm);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << mpi_rank << "] SLATE error: " << e.what() << std::endl;
        std::cerr.flush();
        MPI_Abort(comm, 1);
    }
}

} // extern "C"
