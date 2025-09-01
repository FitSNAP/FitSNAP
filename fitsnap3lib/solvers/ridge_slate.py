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
        
        # Debug output - print all in one statement to avoid tangled output
        pt.all_print(f"--------\n"
                     f"Ridge solver BEFORE filtering:\n"
                     f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                     f"pt.shared_arrays['a'].array\n{pt.shared_arrays['a'].array}\n"
                     f"pt.shared_arrays['b'].array\n{pt.shared_arrays['b'].array}\n"
                     f"--------\n")
        
        # Get raw arrays from shared memory (may include padding)
        a_full = pt.shared_arrays['a'].array
        b_full = pt.shared_arrays['b'].array  
        w_full = pt.shared_arrays['w'].array
        
        # Find actual data by checking for non-zero weights (padded rows have w=0)
        non_zero_mask = w_full != 0
        actual_data_indices = np.where(non_zero_mask)[0]
        
        if len(actual_data_indices) > 0:
            a_node = a_full[non_zero_mask]
            b_node = b_full[non_zero_mask]
            w_node = w_full[non_zero_mask]
        else:
            # No data on this node
            a_node = np.zeros((0, a_full.shape[1] if len(a_full.shape) > 1 else 0), dtype=np.float64)
            b_node = np.zeros(0, dtype=np.float64)
            w_node = np.zeros(0, dtype=np.float64)
        
        # Handle train/test split
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_gathered = pt.fitsnap_dict['Testing']
            
            # The Testing list was gathered with _sub_comm.allgather (within node)
            # It contains contributions from all ranks on THIS node
            # Filter out padding (' ' or other non-boolean values)
            testing_node = []
            if isinstance(testing_gathered, list):
                for item in testing_gathered:
                    if isinstance(item, list):
                        for val in item:
                            if isinstance(val, bool):
                                testing_node.append(val)
                    elif isinstance(item, bool):
                        testing_node.append(item)
            else:
                testing_node = testing_gathered
            
            testing_node = np.array(testing_node, dtype=bool)
            
            # Verify size match
            if len(testing_node) != len(a_node):
                pt.all_print(f"ERROR: Testing mask size {len(testing_node)} != actual data size {len(a_node)}")
                raise ValueError(f"Testing mask mismatch on node {pt._node_index}")
            
            # Create training mask
            training_node = ~testing_node
            
            # Filter for training data only
            w_train = w_node[training_node]
            a_train = a_node[training_node]
            b_train = b_node[training_node]
        else:
            # No test/train split
            w_train = w_node
            a_train = a_node
            b_train = b_node
        
        # Distribute training data across processes within this node
        node_rows = len(w_train)
        rows_per_proc = node_rows // pt._sub_size
        extra_rows = node_rows % pt._sub_size
        
        if pt._sub_rank < extra_rows:
            start_idx = pt._sub_rank * (rows_per_proc + 1)
            end_idx = start_idx + rows_per_proc + 1
        else:
            start_idx = extra_rows * (rows_per_proc + 1) + (pt._sub_rank - extra_rows) * rows_per_proc
            end_idx = start_idx + rows_per_proc
        
        # Get this process's portion
        actual_end_idx = min(end_idx, len(w_train))
        w_local = w_train[start_idx:actual_end_idx]
        a_local = a_train[start_idx:actual_end_idx]
        b_local = b_train[start_idx:actual_end_idx]
        
        # Apply weights
        if len(w_local) > 0:
            aw = w_local[:, np.newaxis] * a_local
            bw = w_local * b_local
        else:
            aw = np.zeros((0, a_train.shape[1] if len(a_train) > 0 else 0), dtype=np.float64)
            bw = np.zeros(0, dtype=np.float64)
        
        # Get dimensions
        m_local = aw.shape[0]
        n_features = aw.shape[1] if len(aw) > 0 else (a_train.shape[1] if len(a_train) > 0 else 0)
        
        # Ensure all processes agree on n_features
        n_features_array = np.array([n_features], dtype=np.int32)
        pt._comm.Allreduce(MPI.IN_PLACE, n_features_array, op=MPI.MAX)
        n_features = n_features_array[0]
        
        # Ensure correct shape
        if m_local == 0:
            aw = np.zeros((0, n_features), dtype=np.float64)
            bw = np.zeros(0, dtype=np.float64)
        
        # Debug output on rank 0
        if pt._rank == 0:
            pt.single_print(f"\nRank 0 sending to SLATE:")
            pt.single_print(f"aw shape: {aw.shape}")
            pt.single_print(f"bw shape: {bw.shape}")
            pt.single_print(f"alpha: {self.alpha}")
            if m_local > 0:
                pt.single_print(f"aw matrix:\n{aw}")
                pt.single_print(f"bw vector: {bw}")
        
        # ALL processes participate in the SLATE solve
        self.fit = self._slate_ridge_solve_qr(aw, bw, m_local, n_features, pt)
        
        # Solution is available on all processes
        if pt._rank == 0:
            pt.single_print(f"First 5 coefficients: {self.fit[:min(5, len(self.fit))]}")
    
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
        
        # Get raw arrays
        a_full = pt.shared_arrays['a'].array
        b_full = pt.shared_arrays['b'].array
        w_full = pt.shared_arrays['w'].array
        
        # Find actual data vs padding
        non_zero_mask = w_full != 0
        
        if np.any(non_zero_mask):
            a_node = a_full[non_zero_mask]
            b_node = b_full[non_zero_mask]
            w_node = w_full[non_zero_mask]
        else:
            a_node = np.zeros((0, a_full.shape[1] if len(a_full.shape) > 1 else 0), dtype=np.float64)
            b_node = np.zeros(0, dtype=np.float64)
            w_node = np.zeros(0, dtype=np.float64)
        
        # Local predictions
        predictions_node = a_node @ self.fit
        
        # Calculate weighted errors
        errors_node = (predictions_node - b_node) * w_node
        
        # Get testing mask for this node's data
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing']:
            testing_gathered = pt.fitsnap_dict['Testing']
            
            # Filter out padding to get actual Testing mask for this node
            testing_node = []
            if isinstance(testing_gathered, list):
                for item in testing_gathered:
                    if isinstance(item, list):
                        for val in item:
                            if isinstance(val, bool):
                                testing_node.append(val)
                    elif isinstance(item, bool):
                        testing_node.append(item)
            else:
                testing_node = testing_gathered
            
            testing_node = np.array(testing_node, dtype=bool)
            
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
