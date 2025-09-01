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
            self.tile_size = self.config.sections['RIDGE'].tile_size if hasattr(self.config.sections['RIDGE'], 'tile_size') else 16
        else:
            self.alpha = 1e-6
            self.tile_size = 256  # Default tile size for blocking

    def perform_fit(self):
        """
        Perform ridge regression fit on the linear system using SLATE for distributed computation.
        The fit is stored as a member `self.fit`.
        """
        
        pt = self.pt
        
        # Get the full arrays (may include padding)
        a_full = pt.shared_arrays['a'].array
        b_full = pt.shared_arrays['b'].array
        w_full = pt.shared_arrays['w'].array
                
        np.set_printoptions(precision=5, suppress=True, linewidth=np.inf)
        
        if pt._sub_rank == 0:
            pt.sub_print(f"Rank {pt._rank}: actual_length={actual_length} (excluding padding)")
            pt.sub_print(f"Rank {pt._rank}: ACTUAL a matrix shape: {a_node.shape}")
            pt.sub_print(f"Rank {pt._rank}: ACTUAL a matrix (no padding):\n{a_node}")
            pt.sub_print(f"Rank {pt._rank}: ACTUAL b vector: {b_node}")
            pt.sub_print(f"Rank {pt._rank}: ACTUAL w vector: {w_node}")
        
        # Handle testing/training split using the Testing mask from fitsnap_dict
        # The Testing mask is gathered with allgather within each node, creating a list of lists
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            
            if pt._sub_rank == 0:
                pt.sub_print(f"Node {pt._node_index}: Using {train_count} training, {test_count} testing samples")
                pt.sub_print(f"Node {pt._node_index}: training_node mask: {training_node}")
                # Show what we're actually filtering
                pt.sub_print(f"\nNode {pt._node_index}: AFTER filtering (training only):")
                pt.sub_print(f"Node {pt._node_index}: a_node.shape={a_node[training_node].shape}")
                pt.sub_print(f"Node {pt._node_index}: Filtered a matrix:\n{a_node[training_node]}")
                pt.sub_print(f"Node {pt._node_index}: Filtered b vector: {b_node[training_node]}")
                pt.sub_print(f"Node {pt._node_index}: Filtered w vector: {w_node[training_node]}")
            
        
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
        
        # Get the full arrays (may include padding)
        a_full = pt.shared_arrays['a'].array
        b_full = pt.shared_arrays['b'].array
        w_full = pt.shared_arrays['w'].array
        
        # Detect actual data length by finding non-zero weights
        # Padded rows have w=0
        non_zero_mask = w_full != 0
        
        if np.any(non_zero_mask):
            # Extract only the actual data (non-padded rows)
            a_node = a_full[non_zero_mask]
            b_node = b_full[non_zero_mask]
            w_node = w_full[non_zero_mask]
        else:
            # No data on this node
            a_node = np.zeros((0, a_full.shape[1] if len(a_full.shape) > 1 else 0), dtype=np.float64)
            b_node = np.zeros(0, dtype=np.float64)
            w_node = np.zeros(0, dtype=np.float64)
        
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
