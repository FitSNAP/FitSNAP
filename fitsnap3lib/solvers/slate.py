from fitsnap3lib.solvers.slate_common import SlateCommon
import numpy as np

try:
    from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
except ImportError:
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
    except ImportError as e:
        print(f"Warning: Could not import SLATE ARD functions: {e}")
        slate_ard_update_cython = None

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

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
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        # Note: a, b, w remain unchanged - only aw, bw get modified by SLATE
        aw = pt.shared_arrays['aw'].array
        bw = pt.shared_arrays['bw'].array
        
        # Debug output - print all in one statement to avoid tangled output
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=np.inf)
            np.set_printoptions(formatter={'float': '{:.4f}'.format})
            pt.sub_print(f"*** ------------------------\n"
                         f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                         #f"a\n{a}\n"
                         #f"b {b}\n"
                         f"--------------------------------\n")
        
        pt.sub_barrier()
        
        # -------- LOCAL SLICE OF SHARED ARRAY AND REGULARIZATION ROWS --------

        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        aw_start_idx, aw_end_idx = pt.fitsnap_dict["sub_aw_indices"]
        reg_row_idx = pt.fitsnap_dict["reg_row_idx"]
        reg_col_idx = pt.fitsnap_dict["reg_col_idx"]
        reg_num_rows = pt.fitsnap_dict["reg_num_rows"]
        #pt.all_print(f"pt.fitsnap_dict {pt.fitsnap_dict}")
        if self.config.debug:
            pt.all_print(f"*** aw_start_idx {aw_start_idx} aw_end_idx {aw_end_idx} reg_row_idx {reg_row_idx} reg_col_idx {reg_col_idx} reg_num_rows {reg_num_rows}")
        
        # -------- WEIGHTS --------
  
        # Apply weights to my local slice
        local_slice = slice(a_start_idx, a_end_idx+1)
        w_local_slice = slice(aw_start_idx, (aw_end_idx-reg_num_rows+1))
        aw[w_local_slice] = w[local_slice, np.newaxis] * a[local_slice]
        bw[w_local_slice] = w[local_slice] * b[local_slice]

        # -------- TRAINING/TESTING SPLIT --------
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]:
                    if self.config.debug:
                        pt.all_print(f"*** removing i {i} aw_start_idx+i {aw_start_idx+i}")
                    aw[aw_start_idx+i,:] = 0.0
                    bw[aw_start_idx+i] = 0.0

        # -------- REGULARIZATION ROWS --------

        sqrt_alpha = np.sqrt(self.alpha)
        n = a.shape[1]
    
        for i in range(reg_num_rows):
            if reg_col_idx+i < n: # avoid out of bounds padding from multiple nodes
                aw[reg_row_idx+i, reg_col_idx+i] = sqrt_alpha
            bw[reg_row_idx+i] = 0.0

        # -------- SLATE AUGMENTED QR --------
        pt.sub_barrier() # make sure all sub ranks done filling local tiles
        m = aw.shape[0] * self.pt._number_of_nodes # global matrix total rows
        lld = aw.shape[0]  # local leading dimension column-major shared array
        
        if False:
            for i in range(lld):
                for j in range(n):
                    aw[i,j] = (pt._node_index*lld + i)*10 + j
                    if pt._node_index>0:
                        aw[i,j] = -aw[i,j]
                    
        np.set_printoptions(precision=3, suppress=True, floatmode='fixed', linewidth=np.inf)
        if False and self.config.debug:
            pt.sub_print(f"*** SENDING TO SLATE ------------------------\n"
                         f"aw\n{aw}\n"
                         f"bw {bw}\n"
                         f"--------------------------------\n")
                     
        # Determine debug flag from EXTRAS section
        debug_flag = 0
        if self.config.debug:
            debug_flag = 1
            
        slate_ridge_augmented_qr_cython(aw, bw, m, lld, debug_flag)
        
        # Broadcast solution from Node 0 to all nodes via head ranks
        if pt._sub_rank == 0:  # This rank is head of its node
            pt._head_group_comm.Bcast(bw[:n], root=0)

        self.fit = bw[:n]
                
        # *** DO NOT REMOVE !!! ***
        if self.config.debug:
            pt.all_print(f"*** self.fit ------------------------\n"
                f"{self.fit}\n-------------------------------------------------\n")
            


    # --------------------------------------------------------------------------------------------

    def perform_fit_ard(self):
        """
        Perform ARD (Automatic Relevance Determination) regression using SLATE.
        
        Implements the sklearn ARDRegression algorithm for distributed matrices:
        
        Iteratively updates:
        - sigma = inv(alpha * X.T @ X + diag(lambda))  [covariance of coefficients]
        - coef = alpha * sigma @ X.T @ y                [coefficient estimates]
        - lambda = (gamma + 2*lambda_1) / (coef^2 + 2*lambda_2)  [feature precisions]
        - alpha = (m - gamma.sum() + 2*alpha_1) / (SSE + 2*alpha_2)  [noise precision]
        
        where gamma = 1 - lambda * diag(sigma)
        
        Assumes m >> n (many samples, fewer features) so no Woodbury formula needed.
        """
        
        pt = self.pt
        a = pt.shared_arrays['a'].array  # X: design matrix (local portion)
        b = pt.shared_arrays['b'].array  # y: target vector (local portion)
        w = pt.shared_arrays['w'].array  # weights
        
        # Note: a, b, w remain unchanged - only aw, bw get modified
        aw = pt.shared_arrays['aw'].array
        bw = pt.shared_arrays['bw'].array

        # Get dimensions
        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(a_start_idx, a_end_idx+1)
        m = aw.shape[0] * pt._number_of_nodes  # global number of samples
        n = aw.shape[1]  # number of features
        lld = aw.shape[0]  # local leading dimension
        
        # Apply weights to my local slice
        aw[local_slice] = w[local_slice, np.newaxis] * a[local_slice]
        bw[local_slice] = w[local_slice] * b[local_slice]

        # -------- TRAINING/TESTING SPLIT --------
        # Zero out test samples (they won't contribute to the fit)
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]:
                    aw[a_start_idx+i,:] = 0.0
                    bw[a_start_idx+i] = 0.0

        # Initialize ARD parameters
        eps = np.finfo(np.float64).eps
        
        # Compute initial alpha from variance of y (requires MPI reduction)
        # Count training samples (after zeroing out test samples above)
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            local_n_training = np.sum(~np.array(testing_mask, dtype=bool))
        else:
            local_n_training = a_end_idx - a_start_idx + 1
        
        # Compute variance (test samples are already zero'd in bw, so they won't affect mean/variance)
        local_bw = bw[local_slice]
        local_sum_bw = np.sum(local_bw)
        local_sum_bw2 = np.sum(local_bw**2)
        global_sum_bw = pt._comm.allreduce(local_sum_bw, op=MPI.SUM)
        global_sum_bw2 = pt._comm.allreduce(local_sum_bw2, op=MPI.SUM)
        global_n_training = pt._comm.allreduce(local_n_training, op=MPI.SUM)
        mean_bw = global_sum_bw / global_n_training
        var_bw = (global_sum_bw2 / global_n_training) - mean_bw**2
        
        # Compute adaptive hyperparameters (matching legacy ARD)
        ap = 1.0 / (var_bw + eps)  # inverse variance ("alpha prior")
        
        pt.single_print(f"inverse variance in training data: {ap:.6f}, logscale for threshold_lambda: {np.log10(ap):.6f}")
        
        if self.directmethod:
            # Direct method: use specified hyperparameters
            self.alpha_1 = self.alphabig
            self.alpha_2 = self.alphabig
            self.lambda_1 = self.lambdasmall
            self.lambda_2 = self.lambdasmall
            if self.threshold_lambda_config > 0:
                self.threshold_lambda = self.threshold_lambda_config
            else:
                # Auto-compute threshold if not specified
                self.threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
            pt.single_print(f"ARD directmethod: alpha_1={self.alpha_1:.2e}, lambda_1={self.lambda_1:.2e}, threshold_lambda={self.threshold_lambda:.2e}")
        else:
            # Adaptive method: scale by inverse variance
            self.alpha_1 = self.scap * ap
            self.alpha_2 = self.scap * ap
            self.lambda_1 = self.scai * ap
            self.lambda_2 = self.scai * ap
            if self.threshold_lambda_config > 0:
                self.threshold_lambda = self.threshold_lambda_config
            else:
                # Auto-compute threshold: 10^(int(abs(log10(ap))) + logcut)
                self.threshold_lambda = 10**(int(np.abs(np.log10(ap))) + self.logcut)
            pt.single_print(f"automated threshold_lambda will be 10**({self.logcut:.6f} + {np.abs(np.log10(ap)):.3f})")
            pt.single_print(f"ARD adaptive: scap={self.scap:.2e}, scai={self.scai:.2e}, alpha_1={self.alpha_1:.6e}, lambda_1={self.lambda_1:.6e}, threshold_lambda={self.threshold_lambda:.2e}")
        
        alpha_ = 1.0 / (var_bw + eps)
        lambda_ = np.ones(n, dtype=np.float64)
        coef_ = np.zeros(n, dtype=np.float64)
        keep_lambda = np.ones(n, dtype=bool)
        coef_old_ = None
        
        pt.single_print(f"ARD: m {m} n {n} var(bw)={var_bw:.9f} alpha={alpha_:.2f}")
        
        np.set_printoptions(
            precision=4, suppress=False, floatmode='fixed', linewidth=800,
            formatter={'float': '{:.4f}'.format}, threshold = 800, edgeitems=5
        )
        
        pt.debug_sub_print(f"aw\n{aw}\nbw {bw}\n\n")

        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            # Get active indices
            active_indices = np.where(keep_lambda)[0]
            n_active = len(active_indices)
            
            if n_active == 0:
                if self.config.debug and pt._rank == 0:
                    pt.single_print(f"ARD: all features pruned at iteration {iter_}")
                break
            
            # Extract active columns from aw and active lambda values
            aw_active = np.asfortranarray(aw[:, active_indices])  # column-major for SLATE
            lambda_active = lambda_[active_indices]
            
            # Update sigma and coef using SLATE C++ functions
            sigma_, coef_active_ = slate_ard_update_cython(
                aw_active, bw, lambda_active, alpha_, m, n_active, lld, self.config.debug
            )
            
            # Map active coefficients back to full coefficient vector
            coef_ = np.zeros(n, dtype=np.float64)
            coef_[active_indices] = coef_active_
            
            # Compute SSE (sum of squared errors) across all ranks
            # Use weighted data consistently: residual = bw - aw @ coef
            local_pred = aw[local_slice] @ coef_
            local_residual = bw[local_slice] - local_pred
            local_sse = np.sum(local_residual**2)  # Already weighted, don't apply weights again
            sse_ = pt._comm.allreduce(local_sse, op=MPI.SUM)
            
            # Update alpha and lambda (on all ranks to stay synchronized)
            gamma_ = 1.0 - lambda_active * np.diag(sigma_)
            
            lambda_[active_indices] = (gamma_ + 2.0 * self.lambda_1) / (
                coef_active_**2 + 2.0 * self.lambda_2
            )
            
            # DEBUG: Print alpha calculation components
            if pt._rank == 0:
                pt.single_print(f"\n[ALPHA DEBUG] Iteration {iter_}:")
                pt.single_print(f"  global_n_training = {global_n_training}")
                pt.single_print(f"  gamma_.sum() = {gamma_.sum():.12f}")
                pt.single_print(f"  self.alpha_1 = {self.alpha_1}")
                pt.single_print(f"  self.alpha_2 = {self.alpha_2}")
                pt.single_print(f"  sse_ = {sse_:.12f}")
                pt.single_print(f"  numerator = {global_n_training - gamma_.sum() + 2.0 * self.alpha_1:.12f}")
                pt.single_print(f"  denominator = {sse_ + 2.0 * self.alpha_2:.12f}")
            
            alpha_ = (global_n_training - gamma_.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)
            
            if pt._rank == 0:
                pt.single_print(f"  alpha_ = {alpha_:.12f}\n")
            
            pt.single_print(f"*** Iteration {iter_}: alpha={alpha_:.6f}, sse={sse_:.6f}, gamma_sum={gamma_.sum():.6f}, n_active={n_active} sigma_\n{sigma_}")
 
            #pt.single_print(f"*** lambda_ {lambda_} self.threshold_lambda {self.threshold_lambda}")

            # Prune features with high lambda (low relevance)
            keep_lambda = lambda_ < self.threshold_lambda
            coef_[~keep_lambda] = 0
            
            # Check for convergence
            if coef_old_ is not None:
                coef_change = np.sum(np.abs(coef_old_ - coef_))
                if coef_change < self.tol:
                    if self.config.debug and pt._rank == 0:
                        active_features = np.sum(keep_lambda)
                        pt.single_print(f"ARD converged after {iter_} iterations, "
                                      f"{active_features}/{n} features active")
                    break
            
            coef_old_ = np.copy(coef_)
        
        # Store final solution
        self.fit = coef_
        
        if self.config.debug and pt._rank == 0:
            active_features = np.sum(keep_lambda)
            pt.single_print(f"ARD final: {active_features}/{n} features active, "
                          f"alpha={alpha_:.2e}, lambda range=[{np.min(lambda_):.2e}, {np.max(lambda_):.2e}]")

    # --------------------------------------------------------------------------------------------
