from fitsnap3lib.solvers.solver import Solver
import numpy as np

try:
    from mpi4py import MPI
    # Import from the module's __init__.py which handles the wrapper imports
    from fitsnap3lib.lib.scalapack_solver import lstsq, dummy_lstsq

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)

        def perform_fit(self):
            # For ScaLAPACK with true_multinode, we use all data for fitting
            # The testing/validation split is handled during error analysis
            # This is because the data is already distributed across nodes and filtering
            # would break the distribution pattern required by ScaLAPACK
            
            if self.config.sections["SOLVER"].true_multinode:
                # Use original arrays but need to redistribute them across nodes
                self.pt.shared_arrays['a_train'] = self.pt.shared_arrays['a']
                self.pt.shared_arrays['b_train'] = self.pt.shared_arrays['b']
                self.pt.shared_arrays['w_train'] = self.pt.shared_arrays['w']
                
                # Call split_by_node to redistribute the data across nodes
                # This is required for arrays created with tm=True
                self.pt.split_by_node(self.pt.shared_arrays['w_train'])
                self.pt.split_by_node(self.pt.shared_arrays['a_train'])
                self.pt.split_by_node(self.pt.shared_arrays['b_train'])
                
                # Now get the dimensions after redistribution
                # total_length should be the sum across all nodes
                # node_length is the data on this specific node
                # scraped_length is the actual usable data on this node
                total_length = self.pt.shared_arrays['a_train'].get_total_length()
                node_length = self.pt.shared_arrays['a_train'].get_node_length()
                scraped_length = self.pt.shared_arrays['a_train'].get_scraped_length()
                
                # Validate that total_length is reasonable
                if total_length <= 0 or total_length > 1e9:
                    raise ValueError(f"[Node {self.pt.get_node()}] Invalid total_length: {total_length}")
                
                # For ScaLAPACK, we need the actual global size, not per-node
                # The total_length should already be the global size if properly computed
                # But let's make sure by using MPI allreduce if needed
                local_size = len(self.pt.shared_arrays['a_train'].array)
                global_size = self.pt._comm.allreduce(local_size, op=MPI.SUM)
                
                if self.pt.get_subrank() == 0:
                    print(f"[Node {self.pt.get_node()}] Size validation:", flush=True)
                    print(f"  local_size = {local_size}", flush=True)
                    print(f"  global_size (via allreduce) = {global_size}", flush=True)
                    print(f"  total_length (from get_total_length) = {total_length}", flush=True)
                
                # Use the correct global size
                if abs(global_size - total_length) > 1:
                    if self.pt.get_subrank() == 0:
                        print(f"[Node {self.pt.get_node()}] WARNING: total_length mismatch, using global_size", flush=True)
                    total_length = global_size
                
                # Debug output
                if self.pt.get_subrank() == 0:
                    print(f"[Node {self.pt.get_node()}] After split_by_node (multinode):", flush=True)
                    print(f"  total_length = {total_length}", flush=True)
                    print(f"  node_length = {node_length}", flush=True)
                    print(f"  scraped_length = {scraped_length}", flush=True)
                    print(f"  a_train shape = {self.pt.shared_arrays['a_train'].array.shape}", flush=True)
            else:
                # Single node mode - can handle testing/training split normally
                # Check if we have a testing split
                has_testing = False
                if self.pt.get_subrank() == 0:
                    if self.pt.get_rank() == 0:
                        has_testing = ('Testing' in self.pt.fitsnap_dict and 
                                       self.pt.fitsnap_dict['Testing'] is not None and 
                                       any(self.pt.fitsnap_dict['Testing']))
                
                # Broadcast within node
                has_testing = self.pt._sub_comm.bcast(has_testing, root=0)
                
                if has_testing:
                    # Handle testing/training split for single node
                    testing_array = None
                    training_mask = None
                    train_length = None
                    n_features = None
                    
                    if self.pt.get_subrank() == 0:
                        testing_array = np.array(self.pt.fitsnap_dict['Testing'], dtype=bool)
                        training_mask = ~testing_array
                        
                        # Get original arrays
                        a_orig = self.pt.shared_arrays['a'].array
                        b_orig = self.pt.shared_arrays['b'].array
                        w_orig = self.pt.shared_arrays['w'].array
                        
                        # Filter for training data only
                        a_train = a_orig[training_mask]
                        b_train = b_orig[training_mask]
                        w_train = w_orig[training_mask]
                        
                        # Get dimensions
                        train_length = len(a_train)
                        n_features = a_orig.shape[1] if len(a_orig.shape) > 1 else 1
                    
                    # Broadcast dimensions
                    train_length = self.pt._sub_comm.bcast(train_length, root=0)
                    n_features = self.pt._sub_comm.bcast(n_features, root=0)
                    
                    # Create shared arrays for training data
                    self.pt.create_shared_array('a_train', train_length, n_features, tm=0)
                    self.pt.create_shared_array('b_train', train_length, tm=0)
                    self.pt.create_shared_array('w_train', train_length, tm=0)
                    
                    if self.pt.get_subrank() == 0:
                        # Copy training data
                        self.pt.shared_arrays['a_train'].array[:] = a_train
                        self.pt.shared_arrays['b_train'].array[:] = b_train
                        self.pt.shared_arrays['w_train'].array[:] = w_train
                        self.training_mask = training_mask
                    else:
                        self.training_mask = None
                else:
                    # No testing split, use all data
                    self.pt.shared_arrays['a_train'] = self.pt.shared_arrays['a']
                    self.pt.shared_arrays['b_train'] = self.pt.shared_arrays['b']
                    self.pt.shared_arrays['w_train'] = self.pt.shared_arrays['w']
                    self.training_mask = None
                
                # Get dimensions for single node
                total_length = len(self.pt.shared_arrays['a_train'].array)
                node_length = total_length
                scraped_length = total_length
            
            # Ensure we have valid lengths
            if total_length <= 0 or node_length <= 0:
                raise ValueError(f"Invalid lengths: total={total_length}, node={node_length}")
            
            lengths = [total_length, node_length, scraped_length]
            
            if self.pt.get_subrank() == 0:
                # Get the data for this node after split_by_node
                # split_by_node may have redistributed data, so use actual array sizes
                w = self.pt.shared_arrays['w_train'].array[:]
                a = self.pt.shared_arrays['a_train'].array[:]
                b = self.pt.shared_arrays['b_train'].array[:]
                
                # Apply weights
                aw = w[:, np.newaxis] * a
                bw = w * b
                
                # Apply transpose trick if configured
                if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
                    if np.linalg.cond(aw)**2 < 1 / np.finfo(float).eps:
                        bw = aw.T @ bw
                        aw = aw.T @ aw
                    else:
                        if self.pt.get_rank() == 0:
                            print("The Matrix is ill-conditioned for the transpose trick")
                
                self.fit = lstsq(aw, bw, self.pt, lengths=lengths)
                # ScaLAPACK returns the result only on rank 0
                # We need to broadcast it to other ranks if needed
                if self.pt.get_rank() == 0 and self.fit is not None:
                    # Head node has the result
                    if self.pt.get_subrank() == 0:
                        print(f"[Node {self.pt.get_node()}] Received solution with shape: {self.fit.shape if self.fit is not None else 'None'}", flush=True)
                else:
                    # Other ranks don't have the result from ScaLAPACK
                    self.fit = None
                
                # Broadcast the result from rank 0 to all other ranks
                self.fit = self.pt._comm.bcast(self.fit, root=0)
            else:
                self.fit = dummy_lstsq(self.pt)

        def _dump_a(self):
            np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = self.pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)
            raise ModuleNotFoundError("ScaLAPACK module not installed in lib")

except ImportError:

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)
            raise ImportError("ScaLAPACK module not installed in lib")
