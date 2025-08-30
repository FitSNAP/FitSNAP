from fitsnap3lib.solvers.solver import Solver
import numpy as np


try:
    # Import from the module's __init__.py which handles the wrapper imports
    from fitsnap3lib.lib.scalapack_solver import lstsq, dummy_lstsq

    class ScaLAPACK(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)

        def perform_fit(self):
            # Check if we have a testing split on the head subrank of each node
            has_testing = False
            if self.pt.get_subrank() == 0:
                # Only rank 0 has the gathered fitsnap_dict with Testing info
                if self.pt.get_rank() == 0:
                    has_testing = ('Testing' in self.pt.fitsnap_dict and 
                                   self.pt.fitsnap_dict['Testing'] is not None and 
                                   any(self.pt.fitsnap_dict['Testing']))
                # Broadcast from rank 0 to all head subnodes
                if self.pt._number_of_nodes > 1:
                    has_testing = self.pt._head_group_comm.bcast(has_testing, root=0)
            
            # Broadcast within each node
            has_testing = self.pt._sub_comm.bcast(has_testing, root=0)
            
            # Handle testing/training split if needed
            if has_testing:
                # Get the Testing array and create training mask on head subrank
                testing_array = None
                training_mask = None
                train_length = None
                n_features = None
                
                if self.pt.get_subrank() == 0:
                    # Get the Testing array and create training mask
                    if self.pt.get_rank() == 0:
                        testing_array = np.array(self.pt.fitsnap_dict['Testing'], dtype=bool)
                    
                    # Broadcast testing array to all head subnodes
                    if self.pt._number_of_nodes > 1:
                        testing_array = self.pt._head_group_comm.bcast(testing_array, root=0)
                    
                    training_mask = ~testing_array  # Invert to get training samples
                    
                    # Get original arrays
                    a_orig = self.pt.shared_arrays['a'].array
                    b_orig = self.pt.shared_arrays['b'].array
                    w_orig = self.pt.shared_arrays['w'].array
                    
                    # Filter for training data only
                    a_train = a_orig[training_mask]
                    b_train = b_orig[training_mask]
                    w_train = w_orig[training_mask]
                    
                    # Get dimensions for creating shared arrays
                    train_length = len(a_train)
                    n_features = a_orig.shape[1] if len(a_orig.shape) > 1 else 1
                
                # Broadcast dimensions to all subranks for shared array creation
                train_length = self.pt._sub_comm.bcast(train_length, root=0)
                n_features = self.pt._sub_comm.bcast(n_features, root=0)
                
                # All subranks create the shared arrays (required for shared memory)
                self.pt.create_shared_array('a_train', train_length, n_features, 
                                             tm=self.config.sections["SOLVER"].true_multinode)
                self.pt.create_shared_array('b_train', train_length, 
                                             tm=self.config.sections["SOLVER"].true_multinode)
                self.pt.create_shared_array('w_train', train_length, 
                                             tm=self.config.sections["SOLVER"].true_multinode)
                
                # Only head subrank copies the training data
                if self.pt.get_subrank() == 0:
                    # Copy training data to new arrays
                    self.pt.shared_arrays['a_train'].array[:] = a_train
                    self.pt.shared_arrays['b_train'].array[:] = b_train
                    self.pt.shared_arrays['w_train'].array[:] = w_train
                    
                    # Store the training mask for error analysis later
                    self.training_mask = training_mask
                else:
                    self.training_mask = None
            else:
                # No testing split, use all data
                self.pt.shared_arrays['a_train'] = self.pt.shared_arrays['a']
                self.pt.shared_arrays['b_train'] = self.pt.shared_arrays['b']
                self.pt.shared_arrays['w_train'] = self.pt.shared_arrays['w']
                self.training_mask = None
                    
            # Now perform the fit with training data
            self.pt.split_by_node(self.pt.shared_arrays['w_train'])
            self.pt.split_by_node(self.pt.shared_arrays['a_train'])
            self.pt.split_by_node(self.pt.shared_arrays['b_train'])
            
            total_length = self.pt.shared_arrays['a_train'].get_total_length()
            node_length = self.pt.shared_arrays['a_train'].get_node_length()
            scraped_length = self.pt.shared_arrays['a_train'].get_scraped_length()
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
                if self.pt.get_subrank() == 0 and self.fit is not None:
                    self.fit = self.pt.gather_to_head_node(self.fit)[0]
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
