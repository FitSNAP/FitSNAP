
from fitsnap3lib.solvers.slate_common import SlateCommon
import numpy as np

class SLATE(SlateCommon):

    # --------------------------------------------------------------------------------------------

    def perform_fit(self):
        
        if self.method == "ARD":
            self.perform_fit_ard()
        else:
            self.perform_fit_ridge()
    
    # --------------------------------------------------------------------------------------------

    def perform_fit_ridge(self):
        
        pt = self.pt
        
        # Debug output
        if self.config.debug:
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=np.inf)
            np.set_printoptions(formatter={'float': '{:.4f}'.format})
            pt.sub_print(f"*** ------------------------\n"
                         f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                         f"--------------------------------\n")
        
        pt.sub_barrier()
        
        # Use common ridge solve method with uniform alpha
        self.fit = self._ridge_solve_iteration()
                
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            pt.all_print(f"*** self.fit ------------------------\n"
                f"{self.fit}\n-------------------------------------------------\n")

    # --------------------------------------------------------------------------------------------

    def perform_fit_ard(self):
        """
        ARD implementation using SLATE ridge solver at each iteration
        Replicates the existing ARD options: directmethod, scap, scai, logcut
        """
        pt = self.pt
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        # Initialize ARD parameters
        self.n_features = a.shape[1]
        
        if pt._rank == 0:
            # Initialize alphas based on directmethod setting
            if self.directmethod == 0:
                # Use automatic scaling based on data variance (like existing ARD)
                training = [not elem for elem in pt.fitsnap_dict['Testing']] if 'Testing' in pt.fitsnap_dict else [True] * len(b)
                bw_training = w[training] * b[training]
                ap = 1.0 / np.var(bw_training) if np.var(bw_training) > 0 else 1.0
                
                # Calculate automatic threshold like existing ARD
                threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
                initial_alpha = self.scap * ap
                
                pt.single_print(f'ARD: inverse variance in training data: {ap:.2e}')
                pt.single_print(f'ARD: automatic threshold_lambda: {threshold_lambda:.2e}') 
                pt.single_print(f'ARD: using automatic alpha scaling: {initial_alpha:.2e}')
                
            else:
                # Use direct method with fixed values
                initial_alpha = self.alpha
                threshold_lambda = 1e6  # Large threshold for pruning
                pt.single_print(f'ARD: using direct method with alpha: {initial_alpha:.2e}')
            
            # Initialize all alphas to the same small value
            self.alphas = np.full(self.n_features, initial_alpha)
            
        # Broadcast to all ranks
        self.alphas = pt._comm.bcast(self.alphas, root=0)
        
        # ARD iteration loop
        converged = False
        for iteration in range(self.max_iterations):
            
            # Perform SLATE ridge solve with current alphas
            solution = self._ridge_solve_iteration(alpha_values=self.alphas)
            
            if pt._rank == 0:
                # Update ARD parameters
                old_alphas = self.alphas.copy()
                
                # Compute effective number of parameters (gamma)
                # For now use simplified update without Hessian diagonal
                # gamma_i ≈ 1 for active parameters, 0 for inactive
                gamma = np.ones(self.n_features)
                
                # ARD update rule: alpha_i = gamma_i / w_i^2
                epsilon = 1e-12  # Numerical stability
                self.alphas = gamma / (solution**2 + epsilon)
                
                # Apply threshold for pruning (like existing ARD)
                if self.directmethod == 0:
                    self.alphas = np.minimum(self.alphas, threshold_lambda)
                
                # Check convergence
                alpha_change = np.mean(np.abs(self.alphas - old_alphas) / (old_alphas + epsilon))
                active_features = np.sum(self.alphas < 1e6)
                max_alpha = np.max(self.alphas)
                
                # Convergence criteria
                if alpha_change < self.tolerance:
                    converged = True
                    reason = "Alpha convergence"
                elif active_features <= 1:
                    converged = True
                    reason = "Too few active features"
                
                # Print progress
                if self.config.debug or iteration % 10 == 0:
                    pt.single_print(f"ARD Iter {iteration}: "
                                   f"α_change={alpha_change:.2e}, "
                                   f"active={active_features}/{self.n_features}, "
                                   f"max_α={max_alpha:.2e}")
                
                if converged:
                    pt.single_print(f"ARD converged after {iteration} iterations: {reason}")
                    break
            
            # Broadcast convergence decision and updated alphas
            self.alphas = pt._comm.bcast(self.alphas, root=0)
            converged = pt._comm.bcast(converged if pt._rank == 0 else False, root=0)
            
            if converged:
                break
        
        self.fit = solution
        
        if pt._rank == 0:
            active_count = np.sum(self.alphas < 1e6)
            pt.single_print(f"ARD completed: {active_count}/{self.n_features} features active")
            
            if self.config.debug:
                pt.all_print(f"*** ARD self.fit ------------------------\n"
                    f"{self.fit}\n-------------------------------------------------\n")


    # --------------------------------------------------------------------------------------------
