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
        aw = pt.shared_arrays['a'].array
        bw = pt.shared_arrays['b'].array

        # Get dimensions
        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(a_start_idx, a_end_idx+1)
        m = a.shape[0] * pt._number_of_nodes  # global number of samples
        n = a.shape[1]  # number of features
        lld = a.shape[0]  # local leading dimension
        
        # Filter to training data only (matching reference implementation)
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            training_mask = ~np.array(testing_mask, dtype=bool)
        else:
            training_mask = np.ones(len(local_b_slice), dtype=bool)
        
        # -------- WEIGHTS --------
  
        # Apply weights to my local slice
        local_slice = slice(a_start_idx, a_end_idx+1)
        aw[local_slice] = w[local_slice, np.newaxis] * a[local_slice]
        bw[local_slice] = w[local_slice] * b[local_slice]

        # -------- TRAINING/TESTING SPLIT --------
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            testing_mask = pt.fitsnap_dict['Testing'][local_slice]
            for i in range(a_end_idx-a_start_idx+1):
                if testing_mask[i]:
                    aw[aw_start_idx+i,:] = 0.0
                    bw[aw_start_idx+i] = 0.0

        # Initialize ARD parameters
        eps = np.finfo(np.float64).eps
        
        # Compute initial alpha from variance of y (requires MPI reduction)
        # IMPORTANT: sklearn ARDRegression in reference code receives weighted y (bw = w * y)
        # and computes variance on that weighted data, training samples only
       
        # Variance of weighted y (bw) across all ranks, training samples only
        local_sum_bw = np.sum(bw[local_slice])
        local_sum_bw2 = np.sum(bw[local_slice]**2)
        local_n_training = a_end_idx - a_end_idx
        
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
            local_pred = a[local_slice] @ coef_
            local_residual = bw[local_slice] - local_pred
            local_sse = np.sum(w[local_slice] * local_residual**2)
            sse_ = pt._comm.allreduce(local_sse, op=MPI.SUM)
            
            # Update alpha and lambda (on all ranks to stay synchronized)
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            
            lambda_[keep_lambda] = (gamma_ + 2.0 * self.lambda_1) / (
                coef_[keep_lambda]**2 + 2.0 * self.lambda_2
            )
            
            alpha_ = (m - gamma_.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)
            
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
