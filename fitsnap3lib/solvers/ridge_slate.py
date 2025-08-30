from fitsnap3lib.solvers.solver import Solver
import numpy as np

try:
    from fitsnap3lib.lib.slate_solver.slate_wrapper import ridge_solve
    SLATE_AVAILABLE = True
except ImportError:
    print("SLATE module not compiled. Run: cd fitsnap3lib/lib/slate_solver && python setup.py build_ext --inplace")
    ridge_solve = None
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
        
        # Get regularization parameter
        self.alpha = self.config.sections['RIDGE'].alpha if 'RIDGE' in self.config.sections else 1e-6
        
        # Set SLATE parameters
        self.tile_size = 256  # Default tile size for blocking
        if 'SLATE' in self.config.sections:
            if hasattr(self.config.sections['SLATE'], 'tile_size'):
                self.tile_size = self.config.sections['SLATE'].tile_size

    def perform_fit(self):
        """
        Perform ridge regression fit on the linear system using SLATE for distributed computation.
        The fit is stored as a member `self.fit`.
        """
        pt = self.pt
        
        # Handle testing/training split similar to ScaLAPACK
        if pt.get_subrank() == 0:
            if any(pt.fitsnap_dict.get('Testing', [])):
                # Extract training indices
                training = [not elem for elem in pt.fitsnap_dict['Testing']]
            else:
                training = None  # Use all data
        
        # Split arrays by node for distributed computation
        pt.split_by_node(pt.shared_arrays['w'])
        pt.split_by_node(pt.shared_arrays['a'])
        pt.split_by_node(pt.shared_arrays['b'])
        
        # Get array dimensions for distributed computation
        total_length = pt.shared_arrays['a'].get_total_length()
        node_length = pt.shared_arrays['a'].get_node_length()
        scraped_length = pt.shared_arrays['a'].get_scraped_length()
        lengths = [total_length, node_length, scraped_length]
        
        if pt.get_subrank() == 0:
            # Apply weights and handle training indices
            w = pt.shared_arrays['w'].array[:]
            a_local = pt.shared_arrays['a'].array[:]
            b_local = pt.shared_arrays['b'].array[:]
            
            if training is not None:
                # Apply training mask
                training_mask = np.array(training[:len(w)])  # Ensure same length as local data
                w = w[training_mask]
                a_local = a_local[training_mask]
                b_local = b_local[training_mask]
            
            # Apply weights
            aw = w[:, np.newaxis] * a_local
            bw = w * b_local
            
            # Apply transpose method if configured
            if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
                bw = aw.T @ bw
                aw = aw.T @ aw
                # In this case, aw is already A^T A and bw is A^T b
                local_ata = aw
                local_atb = bw
            else:
                # Compute local portions of normal equations
                local_ata = aw.T @ aw
                local_atb = aw.T @ bw
            
            # Get feature dimension
            n_features = local_ata.shape[0]
            
            # Use SLATE for distributed ridge regression
            self.fit = self._slate_ridge_solve(local_ata, local_atb, n_features, lengths, pt)
            
            if pt.get_subrank() == 0:
                self.fit = pt.gather_to_head_node(self.fit)[0]
        else:
            self.fit = self._dummy_slate_solve()
    
    def _slate_ridge_solve(self, local_ata, local_atb, n_features, lengths, pt):
        """
        Solve ridge regression using SLATE.
        
        Args:
            local_ata: Local portion of A^T A matrix
            local_atb: Local portion of A^T b vector
            n_features: Number of features (columns in A)
            lengths: Array dimensions [total_length, node_length, scraped_length]
            pt: Parallel tools instance
        
        Returns:
            Solution vector (coefficients)
        """
        if not SLATE_AVAILABLE:
            raise RuntimeError("SLATE module not available. Please compile it first.")
        
        # Call the SLATE ridge solver directly
        solution = ridge_solve(
            local_ata, 
            local_atb, 
            self.alpha, 
            pt._head_group_comm, 
            self.tile_size
        )
        
        return solution
    
    def _dummy_slate_solve(self):
        """Dummy solver for non-head ranks."""
        # Non-head ranks don't participate in solving
        return None
            

        
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
