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
        
        for i in range(lld):
            for j in range(n):
                aw[i,j] = (pt._node_index*lld + i)*10 + j
                
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
        
        # Note: a, b, w remain unchanged - only aw, bw get modified by SLATE
        aw = pt.shared_arrays['aw'].array
        bw = pt.shared_arrays['bw'].array

        # Get dimensions
        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(a_start_idx, a_end_idx+1)
        m = a.shape[0] * pt._number_of_nodes  # global number of samples
        n = a.shape[1]  # number of features
        lld = a.shape[0]  # local leading dimension
        
        # We'll handle training/testing split by zeroing out test samples below
        
        # -------- WEIGHTS --------
  
        # Apply weights to my local slice
        local_slice = slice(a_start_idx, a_end_idx+1)
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
        # IMPORTANT: sklearn ARDRegression in reference code receives weighted y (bw = w * y)
        # and computes variance on that weighted data, training samples only
       
        # Variance of weighted y (bw) across all ranks, training samples only
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
        
        if self.config.debug and pt._rank == 0:
            pt.single_print(f"DEBUG: global_n_training={global_n_training} global_mean(bw)={mean_bw:.6f} global_var(bw)={var_bw:.9f}")
        
        alpha_ = 1.0 / (var_bw + eps)
        lambda_ = np.ones(n, dtype=np.float64)
        coef_ = np.zeros(n, dtype=np.float64)
        keep_lambda = np.ones(n, dtype=bool)
        
        coef_old_ = None
        
        if self.config.debug and pt._rank == 0:
            pt.single_print(f"DEBUG: Before iteration - aw[0:5, 0:3]:\n{aw[local_slice][0:5, 0:3]}")
            pt.single_print(f"DEBUG: Before iteration - bw[0:10]: {bw[local_slice][0:10]}")
            pt.single_print(f"DEBUG: Before iteration - w[0:10]: {w[local_slice][0:10]}")
            pt.single_print(f"ARD: m {m} n {n} var(bw)={var_bw:.9f} alpha={alpha_:.2f}")
        
        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            # Update sigma and coef using SLATE C++ functions
            # These compute:
            #   sigma = inv(alpha * X.T @ X + diag(lambda))
            #   coef = alpha * sigma @ X.T @ y
            # in a distributed manner
            sigma_, coef_ = slate_ard_update_cython(
                aw, bw, coef_, alpha_, lambda_, keep_lambda, m, self.config.debug
            )
            
            # Compute SSE (sum of squared errors) across all ranks
            # Use weighted data consistently: residual = bw - aw @ coef
            local_pred = aw[local_slice] @ coef_
            local_residual = bw[local_slice] - local_pred
            local_sse = np.sum(local_residual**2)  # Already weighted, don't apply weights again
            sse_ = pt._comm.allreduce(local_sse, op=MPI.SUM)
            
            # Update alpha and lambda (on all ranks to stay synchronized)
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            
            lambda_[keep_lambda] = (gamma_ + 2.0 * self.lambda_1) / (
                coef_[keep_lambda]**2 + 2.0 * self.lambda_2
            )
            
            alpha_ = (m - gamma_.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)
            
            if self.config.debug and pt._rank == 0:
                pt.single_print(f"Iteration {iter_}: alpha={alpha_:.6e}, sse={sse_:.6e}, gamma_sum={gamma_.sum():.6f}, n_active={np.sum(keep_lambda)}")
            
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
            
            if not keep_lambda.any():
                if self.config.debug and pt._rank == 0:
                    pt.single_print(f"ARD: all features pruned at iteration {iter_}")
                break
        
        # Store final solution
        self.fit = coef_
        
        if self.config.debug and pt._rank == 0:
            active_features = np.sum(keep_lambda)
            pt.single_print(f"ARD final: {active_features}/{n} features active, "
                          f"alpha={alpha_:.2e}, lambda range=[{np.min(lambda_):.2e}, {np.max(lambda_):.2e}]")

    # --------------------------------------------------------------------------------------------
