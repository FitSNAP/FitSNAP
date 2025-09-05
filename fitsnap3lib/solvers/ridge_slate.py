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
        
        # Check that SLATE is available
        if not SLATE_AVAILABLE:
            error_msg = f"[Rank {self.pt._rank}, Node {self.pt._node_index}] SLATE module not available. Please compile it first."
            pt.single_print(error_msg)
            raise RuntimeError(error_msg)
        
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
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        # Debug output - print all in one statement to avoid tangled output
        # *** DO NOT REMOVE !!! ***
        np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)
        pt.single_print(f"------------------------\nSLATE solver BEFORE filtering:\n"
                     f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                     f"a = {a}\n"
                     #f"b = {b}\n"
                     f"--------------------------------\n")
        
        pt.sub_barrier()

        # Each rank works on its portion of the shared array
        # Get this rank's portion indices
        start_idx, end_idx = pt.fitsnap_dict["sub_a_indices"]
        reg_idx = pt.fitsnap_dict["reg_idx"]
        #pt.all_print(f"pt.fitsnap_dict {pt.fitsnap_dict}")
        pt.all_print(f"start_idx {start_idx} end_idx {end_idx} reg_idx {reg_idx}")
        
        # Apply weights to my portion
        #my_aw = my_w[:, np.newaxis] * my_a
        #my_bw = my_w * my_b
        
        # Handle train/test split: test rows should have weight=0
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            
            # The Testing list has markers for each row
            # We need to map it to our local rows
            testing_mask = pt.fitsnap_dict['Testing']
            
            
        # -------- REGULARIZATION_ROWS --------

        sqrt_alpha = np.sqrt(self.alpha)
        n = a.shape[1]
        a[reg_idx:end_idx+1,:] = 99
    
        # Set diagonal elements for the last n_reg_rows in my block
        for i in range(n):
          local_row = reg_idx + i
          diag_idx = pt._rank * int(np.ceil(n/pt._size)) + i
          if local_row <= end_idx :
            a[local_row, diag_idx] = sqrt_alpha
            b[local_row] = 0.0

        
        # Synchronize all ranks on this node
        pt.sub_barrier()
                               
        # Debug output on all ranks
        # *** DO NOT REMOVE !!! ***
        #pt.all_print(f"\nsending to SLATE:\na\n{a}\nb{b}")
                
        # Call the SLATE augmented Q ridge solver with all node/ranks
        m = a.shape[0] * self.pt._number_of_nodes
        ridge_solve_qr(a, b, m, self.pt._comm, self.tile_size)
        self.fit = b[:n]
        
        # Solution is available on all processes
        # *** DO NOT REMOVE !!! ***
        pt.all_print(f"------------------------\nself.fit\n{self.fit}\n--------------------------------\n")
    
    
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
            
            # Filter out padding markers (' ') to get actual Testing mask for this node
            testing_bools = []
            if isinstance(testing_gathered, list):
                for item in testing_gathered:
                    if isinstance(item, bool):
                        testing_bools.append(item)
                    elif isinstance(item, list):
                        # Handle nested lists if they exist
                        for val in item:
                            if isinstance(val, bool):
                                testing_bools.append(val)
                    # Skip padding markers (' ') - they correspond to padded rows already filtered out
            else:
                # If not a list, use as is
                testing_bools = testing_gathered
            
            testing_node = np.array(testing_bools, dtype=bool)
            
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
