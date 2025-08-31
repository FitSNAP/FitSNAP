from fitsnap3lib.solvers.solver import Solver
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Try multiple import methods for the SLATE module
SLATE_AVAILABLE = False
ridge_solve = None
ridge_solve_qr = None

# First try the direct import (after pip install -e .)
try:
    from slate_wrapper import ridge_solve_qr
    SLATE_AVAILABLE = True
except ImportError:
    # Try the in-place build import path
    try:
        from fitsnap3lib.lib.slate_solver.slate_wrapper import ridge_solve_qr
        SLATE_AVAILABLE = True
    except ImportError:
        # Try relative import for in-place builds
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
            print("SLATE module not compiled. Run: cd fitsnap3lib/lib/slate_solver && python setup.py build_ext --inplace")
            ridge_solve = None
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
        
        # Split arrays by node for distributed computation
        pt.split_by_node(pt.shared_arrays['w'])
        pt.split_by_node(pt.shared_arrays['a'])
        pt.split_by_node(pt.shared_arrays['b'])
        
        # Get array dimensions for distributed computation
        total_length = pt.shared_arrays['a'].get_total_length()
        node_length = pt.shared_arrays['a'].get_node_length()
        scraped_length = pt.shared_arrays['a'].get_scraped_length()
        lengths = [total_length, node_length, scraped_length]
        
        # Handle testing/training split on all processes
        training = None
        if any(pt.fitsnap_dict.get('Testing', [])):
            # Extract training indices
            training = [not elem for elem in pt.fitsnap_dict['Testing']]
        
        # Get the shared data for this node
        w_node = pt.shared_arrays['w'].array[:]
        a_node = pt.shared_arrays['a'].array[:]
        b_node = pt.shared_arrays['b'].array[:]
        
        # Now distribute the node's data across all processes on this node
        # Each process gets a portion based on its subrank
        node_rows = len(w_node)
        rows_per_proc = node_rows // pt._sub_size
        extra_rows = node_rows % pt._sub_size
        
        # Calculate start and end indices for this process
        if pt._sub_rank < extra_rows:
            start_idx = pt._sub_rank * (rows_per_proc + 1)
            end_idx = start_idx + rows_per_proc + 1
        else:
            start_idx = extra_rows * (rows_per_proc + 1) + (pt._sub_rank - extra_rows) * rows_per_proc
            end_idx = start_idx + rows_per_proc
        
        # Get this process's portion of the data
        w = w_node[start_idx:end_idx]
        a_local = a_node[start_idx:end_idx]
        b_local = b_node[start_idx:end_idx]
        
        if training is not None and len(w) > 0:
            # Apply training mask to this process's portion
            training_mask = np.array(training[start_idx:end_idx])
            w = w[training_mask]
            a_local = a_local[training_mask]
            b_local = b_local[training_mask]
        
        # Apply weights
        if len(w) > 0:
            aw = w[:, np.newaxis] * a_local
            bw = w * b_local
        else:
            # Process has no data
            aw = np.zeros((0, a_node.shape[1] if len(a_node) > 0 else 0), dtype=np.float64)
            bw = np.zeros(0, dtype=np.float64)
        
        # Get dimensions
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
        self.fit = self._slate_ridge_solve_qr(aw, bw, m_local, n_features, lengths, pt)
        
        # Solution is already available on all processes after SLATE solve
    
    def _slate_ridge_solve_qr(self, aw, bw, m_local, n_features, lengths, pt):
        """
        Solve ridge regression using SLATE with augmented least squares and QR.
        
        Args:
            aw: Local weighted matrix A (m_local x n_features)
            bw: Local weighted vector b (m_local,)
            m_local: Number of local rows
            n_features: Number of features (columns in A)
            lengths: Array dimensions [total_length, node_length, scraped_length]
            pt: Parallel tools instance
        
        Returns:
            Solution vector (coefficients)
        """
        if not SLATE_AVAILABLE:
            error_msg = f"[Rank {pt._rank}, Node {pt._node}] SLATE module not available. Please compile it first."
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
