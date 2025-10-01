
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
        
        dtype = X.dtype

        n_samples, n_features = X.shape
        coef_ = np.zeros(n_features, dtype=dtype)

        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=self.copy_X
        )

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_

        # Launch the convergence loop
        keep_lambda = np.ones(n_features, dtype=bool)

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        verbose = self.verbose

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero.
        # Explicitly set dtype to avoid unintended type promotion with numpy 2.
        alpha_ = np.asarray(1.0 / (np.var(y) + eps), dtype=dtype)
        lambda_ = np.ones(n_features, dtype=dtype)

        self.scores_ = list()
        coef_old_ = None

        def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
            coef_[keep_lambda] = alpha_ * np.linalg.multi_dot(
                [sigma_, X[:, keep_lambda].T, y]
            )
            return coef_

        update_sigma = (
            self._update_sigma
            if n_samples >= n_features
            else self._update_sigma_woodbury
        )
        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)

            # Update alpha and lambda
            sse_ = np.sum((y - np.dot(X, coef_)) ** 2)
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            lambda_[keep_lambda] = (gamma_ + 2.0 * lambda_1) / (
                (coef_[keep_lambda]) ** 2 + 2.0 * lambda_2
            )
            alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_1) / (sse_ + 2.0 * alpha_2)

            # Prune the weights with a precision over a threshold
            keep_lambda = lambda_ < self.threshold_lambda
            coef_[~keep_lambda] = 0

            # Compute the objective function
            if self.compute_score:
                s = (lambda_1 * np.log(lambda_) - lambda_2 * lambda_).sum()
                s += alpha_1 * log(alpha_) - alpha_2 * alpha_
                s += 0.5 * (
                    fast_logdet(sigma_)
                    + n_samples * log(alpha_)
                    + np.sum(np.log(lambda_))
                )
                s -= 0.5 * (alpha_ * sse_ + (lambda_ * coef_**2).sum())
                self.scores_.append(s)

            # Check for convergence
            if iter_ > 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Converged after %s iterations" % iter_)
                break
            coef_old_ = np.copy(coef_)

            if not keep_lambda.any():
                break

        self.n_iter_ = iter_ + 1

        if keep_lambda.any():
            # update sigma and mu using updated params from the last iteration
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
        else:
            sigma_ = np.array([]).reshape(0, 0)

        self.coef_ = coef_
        self.alpha_ = alpha_
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self._set_intercept(X_offset_, y_offset_, X_scale_)
        return self

        """
        
        pt = self.pt
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        n = a.shape[1]
        lambda_mask = np.ones(n, dtype=bool)


        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            sigma_ = slate_ard_update_sigma_cython(X, alpha_, lambda_, lambda_mask)
            coef_ = slate_ard_update_coeff_cython(X, y, coef_, alpha_, lambda_mask, sigma_)

            # Update alpha and lambda
            sse_ = np.sum((y - np.dot(X, coef_)) ** 2)
            gamma_ = 1.0 - lambda_[lambda_mask] * np.diag(sigma_)
            lambda_[lambda_mask] = (gamma_ + 2.0 * self.lambda_1) / (
                (coef_[lambda_mask]) ** 2 + 2.0 * self.lambda_2
            )
            alpha_ = (n - gamma_.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)

            # Prune the weights with a precision over a threshold
            lambda_mask = lambda_ < self.threshold_lambda
            coef_[~lambda_mask] = 0


            # Check for convergence
            if iter_ > 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if self.config.debug:
                    pt.single_print(f"SLATE/ARD converged after {iter_} iterations")
                break
            coef_old_ = np.copy(coef_)

            if not lambda_mask.any():
                break


        



    # --------------------------------------------------------------------------------------------





































        """
                
                # Print progress
                if iteration == 0 or iteration % 10 == 0 or converged:
                    pt.single_print(f"ARD Iter {iteration}: "
                                   f"Δα={alpha_change:.2e}, "
                                   f"active={active_features}/{self.n_features}, "
                                   f"α∈[{min_alpha:.2e}, {max_alpha:.2e}]")
                
                if converged:
                    pt.single_print(f"ARD converged: {reason}")
            
            # Broadcast convergence and alphas to all ranks
            self.alphas = pt._comm.bcast(self.alphas if pt._rank == 0 else None, root=0)
            converged = pt._comm.bcast(converged if pt._rank == 0 else None, root=0)
            
            if converged:
                break
        
        # Store final solution
        self.fit = solution
        
        if pt._rank == 0:
            active_count = np.sum(self.alphas < threshold_lambda)
            pruned_count = self.n_features - active_count
            pt.single_print(f"ARD completed: {active_count}/{self.n_features} features active, "
                          f"{pruned_count} pruned")
            
            if self.config.debug:
                # Show distribution of alpha values
                alpha_stats = f"Alpha statistics: min={np.min(self.alphas):.2e}, "
                alpha_stats += f"median={np.median(self.alphas):.2e}, max={np.max(self.alphas):.2e}"
                pt.single_print(alpha_stats)
                
                # Show which features are active
                active_mask = self.alphas < threshold_lambda
                active_indices = np.where(active_mask)[0]
                pt.single_print(f"Active feature indices (first 20): {active_indices[:20]}")
                
                pt.all_print(f"*** ARD self.fit (first 20 coefficients) ------------------------\n"
                    f"{self.fit[:20]}\n-------------------------------------------------\n")

        """


    # --------------------------------------------------------------------------------------------
