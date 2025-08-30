from fitsnap3lib.solvers.solver import Solver
import numpy as np
import sys

try:
    import pyslate
    import pyslate.slate as slate
    SLATE_AVAILABLE = True
except ImportError:
    SLATE_AVAILABLE = False
    slate = None

class RidgeSlate(Solver):
    """
    Multi-node Ridge regression solver using SLATE (Software for Linear Algebra Targeting Exascale).
    
    This solver leverages SLATE's distributed matrix operations to solve ridge regression problems
    across multiple nodes efficiently. It solves the regularized least squares problem:
    
    min ||Ax - b||^2 + alpha * ||x||^2
    
    which has the closed-form solution:
    x = (A^T A + alpha * I)^(-1) A^T b
    """

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        
        if not SLATE_AVAILABLE:
            raise ImportError("SLATE library (pyslate) is not installed. "
                            "Please install SLATE to use this solver.")
        
        # Initialize SLATE with MPI communicator if available
        if pt._comm is not None:
            self.slate_comm = pt._comm
            self.use_mpi = True
        else:
            self.slate_comm = None
            self.use_mpi = False
            
        # Get regularization parameter
        self.alpha = self.config.sections['RIDGE'].alpha if 'RIDGE' in self.config.sections else 1e-6
        
        # Set SLATE parameters
        self.tile_size = 256  # Default tile size for SLATE blocking
        if 'SLATE' in self.config.sections:
            if hasattr(self.config.sections['SLATE'], 'tile_size'):
                self.tile_size = self.config.sections['SLATE'].tile_size

    def perform_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
        """
        Perform fit on a linear system using SLATE distributed operations.
        If no args are supplied, will use fitting data in `pt.shared_arrays`.

        Args:
            a (np.array): Optional "A" matrix.
            b (np.array): Optional Truth array.
            w (np.array): Optional Weight array.
            fs_dict (dict): Optional dictionary containing a `Testing` key of which A matrix rows should not be trained.
            trainall (bool): Optional boolean declaring whether to train on all samples in the A matrix.

        The fit is stored as a member `fs.solver.fit`.
        """
        pt = self.pt
        
        # Multi-node execution path
        if self.config.sections["SOLVER"].true_multinode:
            self._perform_multinode_fit(a, b, w, fs_dict, trainall)
        else:
            # Single node execution (rank 0 only)
            if pt._rank == 0:
                self._perform_single_node_fit(a, b, w, fs_dict, trainall)
                
    def _perform_single_node_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
        """Single node fit implementation (fallback to standard ridge behavior)."""
        pt = self.pt
        
        # Determine training indices
        if fs_dict is not None:
            training = [not elem for elem in fs_dict['Testing']]
        elif trainall:
            training = [True]*np.shape(a)[0] if a is not None else [True]*np.shape(pt.shared_arrays['a'].array)[0]
        else:
            training = [not elem for elem in pt.fitsnap_dict['Testing']]

        # Get data arrays
        if a is None and b is None and w is None:
            w = pt.shared_arrays['w'].array[training]
            aw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training]
            bw = w * pt.shared_arrays['b'].array[training]
        else:
            aw = w[:, np.newaxis] * a[training]
            bw = w * b[training]

        # Apply transpose if configured
        if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
            bw = aw.T @ bw
            aw = aw.T @ aw
            
        # Solve ridge regression: x = (A^T A + alpha * I)^(-1) A^T b
        ata = aw.T @ aw
        atb = aw.T @ bw
        
        # Add regularization
        n_features = ata.shape[0]
        ata += self.alpha * np.eye(n_features)
        
        # Solve using numpy (single node fallback)
        try:
            self.fit = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            # Fallback to least squares if solve fails
            self.fit = np.linalg.lstsq(ata, atb, rcond=None)[0]
            
    def _perform_multinode_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
        """Multi-node fit implementation using SLATE."""
        pt = self.pt
        
        # Check for testing data
        if pt.get_subrank() == 0:
            if any(pt.fitsnap_dict['Testing']):
                raise NotImplementedError("Testing with the SLATE solver is not yet implemented!")
        
        # Split arrays by node
        pt.split_by_node(pt.shared_arrays['w'])
        pt.split_by_node(pt.shared_arrays['a'])
        pt.split_by_node(pt.shared_arrays['b'])
        
        # Get array dimensions
        total_length = pt.shared_arrays['a'].get_total_length()
        node_length = pt.shared_arrays['a'].get_node_length()
        scraped_length = pt.shared_arrays['a'].get_scraped_length()
        n_features = pt.shared_arrays['a'].array.shape[1] if len(pt.shared_arrays['a'].array.shape) > 1 else 1
        
        lengths = [total_length, node_length, scraped_length]
        
        if pt.get_subrank() == 0:
            # Get local data
            w = pt.shared_arrays['w'].array[:]
            aw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:]
            bw = w * pt.shared_arrays['b'].array[:]
            
            # Initialize SLATE if available
            if SLATE_AVAILABLE and slate is not None:
                try:
                    # Initialize SLATE with MPI communicator
                    slate.initialize(pt._head_group_comm if pt._head_group_comm else None)
                    
                    # Set up SLATE grid dimensions
                    grid_rows, grid_cols = self._setup_slate_grid(pt._number_of_nodes)
                    
                    # Create SLATE matrix grid
                    grid = slate.Grid(pt._head_group_comm, grid_rows, grid_cols)
                    
                    # Compute local portions of A^T A and A^T b
                    local_ata = aw.T @ aw
                    local_atb = aw.T @ bw
                    
                    # Create SLATE distributed matrices
                    # Note: SLATE uses 2D block-cyclic distribution
                    slate_ata = slate.Matrix.from_numpy(local_ata, grid, self.tile_size)
                    slate_atb = slate.Matrix.from_numpy(local_atb.reshape(-1, 1), grid, self.tile_size)
                    
                    # Perform distributed reduction for A^T A and A^T b
                    slate.allreduce_sum(slate_ata, pt._head_group_comm)
                    slate.allreduce_sum(slate_atb, pt._head_group_comm)
                    
                    # Add ridge regularization: A^T A + alpha * I
                    slate_identity = slate.Matrix.identity(n_features, grid, self.tile_size)
                    slate.scale(self.alpha, slate_identity)
                    slate.add(slate_identity, slate_ata)
                    
                    # Solve the system using SLATE's distributed solver
                    # This performs: x = (A^T A + alpha * I)^(-1) A^T b
                    slate_x = slate.Matrix.zeros(n_features, 1, grid, self.tile_size)
                    slate.posv(slate_ata, slate_atb, slate_x)  # Positive definite solve
                    
                    # Convert back to numpy array
                    self.fit = slate_x.to_numpy().flatten()
                    
                    # Finalize SLATE
                    slate.finalize()
                    
                except Exception as e:
                    # If SLATE fails, fall back to MPI-based approach
                    pt.single_print(f"SLATE operation failed: {e}. Falling back to MPI-based approach.")
                    self._fallback_multinode_fit(aw, bw, n_features, pt)
            else:
                # SLATE not available, use MPI-based fallback
                self._fallback_multinode_fit(aw, bw, n_features, pt)
            
            # Gather solution to head node
            if pt.get_subrank() == 0 and self.fit is not None:
                self.fit = pt.gather_to_head_node(self.fit)[0]
        else:
            # Non-head ranks wait
            self.fit = None
            
    def _fallback_multinode_fit(self, aw, bw, n_features, pt):
        """Fallback multinode implementation using MPI reductions."""
        # Local computation first (each node computes its portion)
        local_ata = aw.T @ aw
        local_atb = aw.T @ bw
        
        # Reduce across nodes using MPI
        if pt._head_group_comm is not None:
            global_ata = np.zeros_like(local_ata)
            global_atb = np.zeros_like(local_atb)
            
            pt._head_group_comm.Allreduce(local_ata, global_ata)
            pt._head_group_comm.Allreduce(local_atb, global_atb)
        else:
            global_ata = local_ata
            global_atb = local_atb
        
        # Add ridge regularization
        global_ata += self.alpha * np.eye(n_features)
        
        # Solve on head node
        if pt._node_index == 0:
            try:
                self.fit = np.linalg.solve(global_ata, global_atb)
            except np.linalg.LinAlgError:
                self.fit = np.linalg.lstsq(global_ata, global_atb, rcond=None)[0]
        else:
            self.fit = None
        
        # Broadcast solution to all head nodes
        if pt._head_group_comm is not None:
            self.fit = pt._head_group_comm.bcast(self.fit, root=0)
            
    def _setup_slate_grid(self, n_nodes):
        """Set up optimal 2D process grid for SLATE."""
        # Find optimal grid dimensions (as square as possible)
        grid_rows = int(np.sqrt(n_nodes))
        grid_cols = n_nodes // grid_rows
        
        while grid_rows * grid_cols != n_nodes:
            grid_rows -= 1
            if grid_rows == 0:
                grid_rows = 1
                grid_cols = n_nodes
                break
            grid_cols = n_nodes // grid_rows
            
        return grid_rows, grid_cols
    
    def _distribute_matrix_slate(self, matrix, tile_size=256):
        """Distribute a matrix using SLATE's tile distribution."""
        # This would implement SLATE's 2D block-cyclic distribution
        # For now, this is a placeholder
        pass
        
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
