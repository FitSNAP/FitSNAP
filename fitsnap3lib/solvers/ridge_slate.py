from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import DistributedList
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Import the SLATE module
SLATE_AVAILABLE = False
ridge_solve_qr = None

try:
    # Primary import method (after pip install -e .)
    from slate_wrapper import ridge_solve_qr
    SLATE_AVAILABLE = True
except ImportError as e:
    # Fallback: try direct path import for in-place builds
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        import slate_wrapper
        ridge_solve_qr = slate_wrapper.ridge_solve_qr
        SLATE_AVAILABLE = True
    except ImportError:
        print(f"Warning: SLATE module import failed: {e}")
        print("To install: cd fitsnap3lib/lib/slate_solver && pip install -e .")
        ridge_solve_qr = None
        SLATE_AVAILABLE = False

class RidgeSlate(Solver):
    """
    Multi-node Ridge regression solver using SLATE (Software for Linear Algebra Targeting Exascale).
    
    This solver leverages SLATE's distributed matrix operations to solve ridge regression
    problems across multiple nodes efficiently.
    
    Solves: (A^T A + alpha * I) x = A^T b
    """

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        
        # Get regularization parameter and tile size from RIDGE section
        if 'RIDGE' in self.config.sections:
            self.alpha = self.config.sections['RIDGE'].alpha
            self.tile_size = self.config.sections['RIDGE'].tile_size if hasattr(self.config.sections['RIDGE'], 'tile_size') else 256
        else:
            self.alpha = 1e-6
            self.tile_size = 256  # Default tile size for blocking

    def perform_fit(self):
        """
        Perform ridge regression fit on the linear system using SLATE for distributed computation.
        The fit is stored as a member `self.fit`.
        """
        
        pt = self.pt
        
        # Get the data for this node - use only the scraped length, not the full padded array
        scraped_length = pt.shared_arrays['a'].get_scraped_length()
        w_node = pt.shared_arrays['w'].array[:scraped_length]
        a_node = pt.shared_arrays['a'].array[:scraped_length]
        b_node = pt.shared_arrays['b'].array[:scraped_length]
        
        if pt._sub_rank == 0:
            pt.sub_print(f"Node {pt._node_index}: scraped_length={scraped_length}")
            pt.sub_print(f"Node {pt._node_index}: First 5 b values: {b_node[:5]}")
            pt.sub_print(f"Node {pt._node_index}: Sum of b values: {np.sum(b_node):.6f}")
            pt.sub_print(f"Node {pt._node_index}: Mean of b values: {np.mean(b_node):.6f}")
        
        # Handle testing/training split using the Testing mask from fitsnap_dict
        # The Testing mask is gathered with allgather within each node, creating a list of lists
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_gathered = pt.fitsnap_dict['Testing']
            
            # Debug: understand the structure
            if pt._sub_rank == 0:
                pt.sub_print(f"Node {pt._node_index}: Testing structure type={type(testing_gathered)}, len={len(testing_gathered) if isinstance(testing_gathered, list) else 'N/A'}")
                if isinstance(testing_gathered, list) and len(testing_gathered) > 0:
                    pt.sub_print(f"Node {pt._node_index}: First element type={type(testing_gathered[0])}, len={len(testing_gathered[0]) if isinstance(testing_gathered[0], list) else 'N/A'}")
            
            # The Testing mask was gathered with allgather across procs in this node
            # Each proc contributed its portion, so we have node_size sublists
            # We need to concatenate them in the right order
            
            if isinstance(testing_gathered, list) and len(testing_gathered) == pt._sub_size:
                # Each element is a sublist from one proc on this node
                # Concatenate all sublists to get the full Testing mask for this node
                testing_node = []
                for proc_list in testing_gathered:
                    if proc_list is not None:  # Some procs might have no data
                        testing_node.extend(proc_list)
                testing_node = np.array(testing_node, dtype=bool)
                
                if pt._sub_rank == 0:
                    pt.sub_print(f"Node {pt._node_index}: Reconstructed Testing mask length={len(testing_node)}")
            else:
                # Unexpected format - try to use as-is
                if pt._sub_rank == 0:
                    pt.sub_print(f"WARNING: Node {pt._node_index} unexpected Testing format, attempting to flatten")
                # Try to flatten whatever structure we have
                testing_flat = []
                if isinstance(testing_gathered, list):
                    for item in testing_gathered:
                        if isinstance(item, list):
                            testing_flat.extend(item)
                        elif item is not None:
                            testing_flat.append(item)
                    testing_node = np.array(testing_flat, dtype=bool) if testing_flat else np.array([], dtype=bool)
                else:
                    testing_node = np.array(testing_gathered, dtype=bool)
            
            # Verify the Testing mask matches our data size
            if len(testing_node) != scraped_length:
                if pt._sub_rank == 0:
                    pt.sub_print(f"ERROR on Node {pt._node_index}: Testing mask size {len(testing_node)} != scraped_length {scraped_length}")
                raise ValueError(f"Testing mask size mismatch on node {pt._node_index}: {len(testing_node)} != {scraped_length}")
            
            # Create training mask (inverse of testing)
            training_node = ~testing_node
            
            # Count samples
            train_count = np.sum(training_node)
            test_count = np.sum(testing_node)
            
            if pt._sub_rank == 0:
                pt.sub_print(f"Node {pt._node_index}: Using {train_count} training, {test_count} testing samples")
            
            # Apply the training filter to get only training data
            w_node = w_node[training_node]
            a_node = a_node[training_node]
            b_node = b_node[training_node]
        else:
            if pt._sub_rank == 0:
                pt.sub_print(f"Node {pt._node_index}: No Testing mask found, using all {scraped_length} samples for training")
        
        # Debug: Report total training data across all nodes
        if pt._sub_rank == 0:
            # Gather statistics from all nodes to check for data duplication
            local_train = len(w_node) if 'w_node' in locals() else scraped_length
            local_b_sum = np.sum(b_node) if 'b_node' in locals() else 0.0
            local_b_mean = np.mean(b_node) if 'b_node' in locals() else 0.0
            
            if pt._head_group_comm:
                # Head nodes communicate
                all_train_counts = pt._head_group_comm.gather(local_train, root=0)
                all_b_sums = pt._head_group_comm.gather(local_b_sum, root=0) 
                all_b_means = pt._head_group_comm.gather(local_b_mean, root=0)
                
                if pt._node_index == 0 and all_train_counts:
                    total_training = sum(all_train_counts)
                    pt.single_print(f"\n=== DATA DISTRIBUTION DEBUG ===")
                    pt.single_print(f"Total training rows: {total_training}")
                    pt.single_print(f"Per-node counts: {all_train_counts}")
                    pt.single_print(f"Per-node b sums: {[f'{s:.2f}' for s in all_b_sums]}")
                    pt.single_print(f"Per-node b means: {[f'{m:.6f}' for m in all_b_means]}")
                    # Check if all nodes have same mean - would indicate duplication
                    if len(set([f'{m:.6f}' for m in all_b_means])) == 1:
                        pt.single_print(f"WARNING: All nodes have identical b means - possible data duplication!")
                    pt.single_print(f"================================\n")
        node_rows = len(w_node)
        rows_per_proc = node_rows // pt._sub_size
        extra_rows = node_rows % pt._sub_size
        
        # Calculate CONTIGUOUS block for this process
        if pt._sub_rank < extra_rows:
            # First 'extra_rows' processes get one extra row
            start_idx = pt._sub_rank * (rows_per_proc + 1)
            end_idx = start_idx + rows_per_proc + 1
        else:
            # Remaining processes get standard number of rows
            start_idx = extra_rows * (rows_per_proc + 1) + (pt._sub_rank - extra_rows) * rows_per_proc
            end_idx = start_idx + rows_per_proc
        
        # Get this process's portion of the data
        # Make sure we don't go out of bounds
        actual_end_idx = min(end_idx, len(w_node))
        w = w_node[start_idx:actual_end_idx]
        a_local = a_node[start_idx:actual_end_idx]
        b_local = b_node[start_idx:actual_end_idx]
        
        # Apply weights
        if len(w) > 0:
            aw = w[:, np.newaxis] * a_local
            bw = w * b_local
        else:
            # Process has no data
            aw = np.zeros((0, a_node.shape[1] if len(a_node) > 0 else 0), dtype=np.float64)
            bw = np.zeros(0, dtype=np.float64)
        
        # Get dimensions - IMPORTANT: m_local is the actual number of rows after filtering
        m_local = aw.shape[0]  # local number of rows for this process
        n_features = aw.shape[1] if len(aw) > 0 else (a_node.shape[1] if len(a_node) > 0 else 0)
        
        # Make sure all processes agree on n_features
        n_features_array = np.array([n_features], dtype=np.int32)
        pt._comm.Allreduce(MPI.IN_PLACE, n_features_array, op=MPI.MAX)
        n_features = n_features_array[0]
        
        # Ensure correct shape even for empty arrays
        if m_local == 0:
            aw = np.zeros((0, n_features), dtype=np.float64)
            bw = np.zeros(0, dtype=np.float64)
        
        # ALL processes participate in the SLATE solve
        self.fit = self._slate_ridge_solve_qr(aw, bw, m_local, n_features, pt)
        
        # Solution is already available on all processes after SLATE solve
    
    def _slate_ridge_solve_qr(self, aw, bw, m_local, n_features, pt):
        """
        Solve ridge regression using SLATE with augmented least squares and QR.
        
        Args:
            aw: Local weighted matrix A (m_local x n_features)
            bw: Local weighted vector b (m_local,)
            m_local: Number of local rows
            n_features: Number of features (columns in A)
            pt: Parallel tools instance
        
        Returns:
            Solution vector (coefficients)
        """
        if not SLATE_AVAILABLE:
            error_msg = f"[Rank {pt._rank}, Node {pt._node_index}] SLATE module not available. Please compile it first."
            pt.single_print(error_msg)
            raise RuntimeError(error_msg)
        
        # Call the SLATE ridge solver with QR directly
        # Use the full communicator (pt._comm) to use ALL MPI ranks
        solution = ridge_solve_qr(
            aw, 
            bw, 
            self.alpha, 
            pt._comm,  # Use full communicator with ALL processes
            self.tile_size
        )
        
        return solution
            
    def _dump_a(self):
        """Save the A matrix to file."""
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        """Save the solution vector to file."""
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        """Save the predicted values to file."""
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
    
    def evaluate_errors(self):
        """Evaluate training and testing errors using distributed computation."""
        pt = self.pt
        
        # Each node computes predictions for its portion of data - use scraped length
        scraped_length = pt.shared_arrays['a'].get_scraped_length()
        a_node = pt.shared_arrays['a'].array[:scraped_length]
        b_node = pt.shared_arrays['b'].array[:scraped_length]
        w_node = pt.shared_arrays['w'].array[:scraped_length]
        
        # Local matrix multiply - each node does its portion
        predictions_node = a_node @ self.fit
        
        # Calculate weighted errors locally
        errors_node = (predictions_node - b_node) * w_node
        
        # Get testing mask for this node's data
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing']:
            # Extract the Testing mask
            testing_full = pt.fitsnap_dict['Testing']
            
            # Handle list of lists from gather_fitsnap
            if testing_full and isinstance(testing_full[0], list):
                testing_flat = []
                for sublist in testing_full:
                    testing_flat.extend(sublist)
                testing_full = testing_flat
            
            # Convert to numpy boolean array
            testing_node = np.array(testing_full, dtype=bool)
            
            if len(testing_node) == len(errors_node):
                training_node = ~testing_node
                
                # Calculate local error statistics
                train_errors_local = errors_node[training_node]
                test_errors_local = errors_node[testing_node]
                
                # Calculate local sums and counts for distributed RMSE
                train_sum_sq = np.sum(train_errors_local**2) if len(train_errors_local) > 0 else 0.0
                train_count = len(train_errors_local)
                test_sum_sq = np.sum(test_errors_local**2) if len(test_errors_local) > 0 else 0.0
                test_count = len(test_errors_local)
                
                # Reduce across all nodes to get global statistics
                train_stats = np.array([train_sum_sq, train_count], dtype=np.float64)
                test_stats = np.array([test_sum_sq, test_count], dtype=np.float64)
                
                if pt._sub_rank == 0:
                    # Head node of each node participates in reduction
                    pt._head_group_comm.Allreduce(MPI.IN_PLACE, train_stats, op=MPI.SUM)
                    pt._head_group_comm.Allreduce(MPI.IN_PLACE, test_stats, op=MPI.SUM)
                
                # Broadcast from head to all procs on node
                pt._sub_comm.Bcast(train_stats, root=0)
                pt._sub_comm.Bcast(test_stats, root=0)
                
                # Calculate global RMSE
                train_rmse = np.sqrt(train_stats[0] / train_stats[1]) if train_stats[1] > 0 else 0.0
                test_rmse = np.sqrt(test_stats[0] / test_stats[1]) if test_stats[1] > 0 else 0.0
                
                if pt._rank == 0:
                    print(f"\nDistributed Evaluation Results:")
                    print(f"Training RMSE: {train_rmse:.6f}")
                    print(f"Testing RMSE: {test_rmse:.6f}")
                    print(f"Training samples: {int(train_stats[1])}")
                    print(f"Testing samples: {int(test_stats[1])}")
                    print(f"Data distributed across {pt._number_of_nodes} nodes, {pt._size} total processes")
            else:
                # Size mismatch - calculate overall RMSE without split
                sum_sq_local = np.sum(errors_node**2)
                count_local = len(errors_node)
                
                stats = np.array([sum_sq_local, count_local], dtype=np.float64)
                
                if pt._sub_rank == 0:
                    pt._head_group_comm.Allreduce(MPI.IN_PLACE, stats, op=MPI.SUM)
                
                pt._sub_comm.Bcast(stats, root=0)
                
                rmse = np.sqrt(stats[0] / stats[1]) if stats[1] > 0 else 0.0
                
                if pt._rank == 0:
                    print(f"\nDistributed Evaluation Results:")
                    print(f"Overall RMSE: {rmse:.6f}")
                    print(f"Total samples: {int(stats[1])}")
                    print(f"Data distributed across {pt._number_of_nodes} nodes")
        else:
            # No test/train split - calculate overall RMSE
            sum_sq_local = np.sum(errors_node**2)
            count_local = len(errors_node)
            
            stats = np.array([sum_sq_local, count_local], dtype=np.float64)
            
            if pt._sub_rank == 0:
                pt._head_group_comm.Allreduce(MPI.IN_PLACE, stats, op=MPI.SUM)
            
            pt._sub_comm.Bcast(stats, root=0)
            
            rmse = np.sqrt(stats[0] / stats[1]) if stats[1] > 0 else 0.0
            
            if pt._rank == 0:
                print(f"\nDistributed Evaluation Results:")
                print(f"Overall RMSE: {rmse:.6f}")
                print(f"Total samples: {int(stats[1])}")
                print(f"Data distributed across {pt._number_of_nodes} nodes")
        
        return errors_node  # Return local errors only
