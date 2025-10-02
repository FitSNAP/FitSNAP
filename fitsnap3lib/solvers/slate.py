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
        
        # Get dimensions
        a_start_idx, a_end_idx = pt.fitsnap_dict["sub_a_indices"]
        local_slice = slice(a_start_idx, a_end_idx+1)
        m = a.shape[0] * pt._number_of_nodes  # global number of samples
        n = a.shape[1]  # number of features
        lld = a.shape[0]  # local leading dimension
        
        # Initialize ARD parameters
        eps = np.finfo(np.float64).eps
        
        # Compute initial alpha from variance of y (requires MPI reduction)
        local_b_slice = b[local_slice]
        local_w_slice = w[local_slice]
        
        # Weighted variance of y across all ranks
        local_sum_wy = np.sum(local_w_slice * local_b_slice)
        local_sum_wy2 = np.sum(local_w_slice * local_b_slice**2)
        local_sum_w = np.sum(local_w_slice)
        
        global_sum_wy = pt._comm.allreduce(local_sum_wy, op=MPI.SUM)
        global_sum_wy2 = pt._comm.allreduce(local_sum_wy2, op=MPI.SUM)
        global_sum_w = pt._comm.allreduce(local_sum_w, op=MPI.SUM)
        
        weighted_mean_y = global_sum_wy / global_sum_w if global_sum_w > 0 else 0.0
        weighted_var_y = (global_sum_wy2 / global_sum_w - weighted_mean_y**2) if global_sum_w > 0 else 1.0
        
        alpha_ = 1.0 / (weighted_var_y + eps)
        lambda_ = np.ones(n, dtype=np.float64)
        coef_ = np.zeros(n, dtype=np.float64)
        keep_lambda = np.ones(n, dtype=bool)
        
        coef_old_ = None
        
        if self.config.debug and pt._rank == 0:
            pt.single_print(f"ARD starting: m={m}, n={n}, alpha={alpha_:.2e}")
        
        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            # Update sigma and coef using SLATE C++ functions
            # These compute:
            #   sigma = inv(alpha * X.T @ X + diag(lambda))
            #   coef = alpha * sigma @ X.T @ y
            # in a distributed manner
            sigma_, coef_ = slate_ard_update_cython(
                a, b, w, coef_, alpha_, lambda_, keep_lambda, m, self.config.debug
            )
            
            # Compute SSE (sum of squared errors) across all ranks
            local_pred = a[local_slice] @ coef_
            local_residual = local_b_slice - local_pred
            local_sse = np.sum(local_w_slice * local_residual**2)
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
