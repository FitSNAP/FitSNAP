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
        
        pt.debug_single_print(f"inverse variance in training data: {ap:.6f}, logscale for threshold_lambda: {np.log10(ap):.6f}")
        
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
            pt.debug_single_print(f"automated threshold_lambda will be 10**({self.logcut:.6f} + {np.abs(np.log10(ap)):.3f})")
            pt.debug_single_print(f"ARD adaptive: scap={self.scap:.2e}, scai={self.scai:.2e}, alpha_1={self.alpha_1:.6e}, lambda_1={self.lambda_1:.6e}, threshold_lambda={self.threshold_lambda:.2e}")
        
        alpha_ = 1.0 / (var_bw + eps)
        lambda_ = np.ones(n, dtype=np.float64)
        coef_ = np.zeros(n, dtype=np.float64)
        lambda_mask = np.ones(n, dtype=bool)
        coef_old_ = None
        
        # Initialize gamma history tracking if validation is enabled
        if self.config.sections["OUTFILE"].validation:
            self.gamma_history = []
        
        pt.debug_single_print(f"ARD: m {m} n {n} var(bw)={var_bw:.9f} alpha={alpha_:.2f}")
        
        np.set_printoptions(
            precision=4, suppress=False, floatmode='fixed', linewidth=np.inf,
            formatter={'float': '{:.3f}'.format}, threshold = 800, edgeitems=5
        )
      
        # Iterative procedure of ARDRegression
        for iter_ in range(self.max_iter):
            # Get active indices
            active_indices = np.where(lambda_mask)[0]
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
            gamma_active = 1.0 - lambda_active * np.diag(sigma_)
            
            # Map gamma back to full feature set
            gamma_ = np.zeros(n, dtype=np.float64)
            gamma_[active_indices] = gamma_active
            
            # Store gamma history if validation enabled
            if self.config.sections["OUTFILE"].validation:
                self.gamma_history.append(gamma_.copy())
            
            lambda_[active_indices] = (gamma_active + 2.0 * self.lambda_1) / (
                coef_active_**2 + 2.0 * self.lambda_2
            )
                        
            alpha_ = (global_n_training - gamma_active.sum() + 2.0 * self.alpha_1) / (sse_ + 2.0 * self.alpha_2)
            
            # Prune features based on selected method
            if self.pruning_method.lower() == 'gamma':
                # Gamma-based pruning: keep features with gamma > threshold
                lambda_mask = gamma_ > self.threshold_gamma
                n_pruned = np.sum(~lambda_mask)
                
                pt.single_print(f"SLATE ARD: iteration {iter_} alpha {alpha_:.6f} sse {sse_:.6f} gamma_sum {gamma_active.sum():.6f} n_active {n_active}")
                
                if self.config.debug:
                    # Show gamma distribution
                    gamma_nonzero = gamma_[gamma_ > 0]
                    pt.single_print(f"  Gamma pruning: keeping {np.sum(lambda_mask)}/{n} features with gamma > {self.threshold_gamma:.3f}")
                    pt.single_print(f"  Gamma range: [{gamma_[gamma_ > 0].min():.4f}, {gamma_.max():.4f}]")
                    pt.single_print(f"  Gamma stats: mean={gamma_nonzero.mean():.3f}, median={np.median(gamma_nonzero):.3f}")
                    pt.single_print(f"  Gamma > 0.5: {np.sum(gamma_ > 0.5)}, > 0.3: {np.sum(gamma_ > 0.3)}, > 0.1: {np.sum(gamma_ > 0.1)}")
            else:
                # Lambda-based pruning: keep features with lambda < threshold (original method)
                lambda_mask = lambda_ < self.threshold_lambda
                n_pruned = np.sum(~lambda_mask)
                
                pt.single_print(f"SLATE ARD: iteration {iter_} alpha {alpha_:.6f} sse {sse_:.6f} gamma_sum {gamma_active.sum():.6f} n_active {n_active}")
                pt.single_print(f"  Lambda pruning: keeping {np.sum(lambda_mask)}/{n} features with lambda < {self.threshold_lambda:.1f}")
                pt.single_print(f"  Lambda range: [{lambda_[lambda_ > 0].min():.4e}, {lambda_.max():.4e}]")
            
            coef_[~lambda_mask] = 0
            
            # Check for convergence
            if coef_old_ is not None:
                coef_change = np.sum(np.abs(coef_old_ - coef_))
                if coef_change < self.tol:
                    if self.config.debug and pt._rank == 0:
                        active_features = np.sum(lambda_mask)
                        pt.single_print(f"ARD converged after {iter_} iterations, "
                                      f"{active_features}/{n} features active")
                    break
            
            coef_old_ = np.copy(coef_)
        
        # Store final solution
        if "PYACE" in self.config.sections:
            pyace_section = self.config.sections["PYACE"]
            if pyace_section.bzeroflag:
                pyace_section.lambda_mask = lambda_mask
            else:
                pyace_section.lambda_mask = lambda_mask[pyace_section.numtypes:]

        self.fit = coef_
        
        # Save gamma history to pickle if validation enabled
        if self.config.sections["OUTFILE"].validation and pt._rank == 0:
            import pickle
            output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
            gamma_history_file = f"{output_prefix}_gamma_history.pkl"
            
            # Get rank information from PYACE basis if available
            feature_ranks = None
            if "PYACE" in self.config.sections:
                pyace_section = self.config.sections["PYACE"]
                if hasattr(pyace_section, 'ctilde_basis'):
                    feature_ranks = []
                    # Rank 1 functions
                    for element_basis_rank1_functions in pyace_section.ctilde_basis.basis_rank1:
                        for basis_rank1_function in element_basis_rank1_functions:
                            feature_ranks.append(int(basis_rank1_function.rank))
                    # Higher rank functions
                    for element_basis_functions in pyace_section.ctilde_basis.basis:
                        for basis_function in element_basis_functions:
                            feature_ranks.append(int(basis_function.rank))
            
            # Save both gamma history and feature ranks
            save_data = {
                'gamma_history': self.gamma_history,
                'feature_ranks': feature_ranks
            }
            
            with open(gamma_history_file, 'wb') as f:
                pickle.dump(save_data, f)
            pt.single_print(f"Saved gamma history and feature ranks to {gamma_history_file}")
        
        if self.config.debug and pt._rank == 0:
            active_features = np.sum(lambda_mask)
            pt.single_print(f"\nARD final: {active_features}/{n} features active, "
                          f"alpha={alpha_:.2e}, lambda range=[{np.min(lambda_):.2e}, {np.max(lambda_):.2e}]")
            if self.pruning_method.lower() == 'gamma':
                gamma_active_final = gamma_[gamma_ > 0]
                pt.single_print(f"Gamma range: [{gamma_active_final.min():.4f}, {gamma_active_final.max():.4f}], "
                              f"mean={gamma_active_final.mean():.4f}")

    # --------------------------------------------------------------------------------------------

    def validation(self):
    
        if self.pt._rank != 0 or not self.config.sections["OUTFILE"].validation:
            return
    
        import pandas as pd
        
        output_prefix = self.config.sections['OUTFILE'].metrics.replace('.md', '')
        notebook_file = f"{output_prefix}_validation.ipynb"
        gamma_history_file = f"{output_prefix}_gamma_history.pkl"
        
        # Create Jupyter notebook structure
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Cell 1: Title
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# ARD Validation Report: {output_prefix}\n", "\n", "This notebook contains validation analysis from the FitSNAP ARD run."]
        })
        
        # Cell 2: Config dictionary (collapsed by default)
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Configuration"]
        })
        
        # Extract relevant config info
        config_dict = {}
        for section_name, section in self.config.sections.items():
            if hasattr(section, '__dict__'):
                config_dict[section_name] = {k: v for k, v in section.__dict__.items() if not k.startswith('_')}
        
        import json
        config_json = json.dumps(config_dict, indent=2, default=str)
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"collapsed": True, "jupyter": {"source_hidden": True}},
            "outputs": [],
            "source": [
                "import json\n",
                "\n",
                "# Configuration from FitSNAP run\n",
                f"config = {config_json}\n",
                "\n",
                "# Display key settings\n",
                "print('ARD Settings:')\n",
                "for key in ['max_iter', 'tol', 'threshold_lambda', 'pruning_method']:\n",
                "    if key in config.get('SLATE', {}): print(f'  {key}: {config[\"SLATE\"][key]}')"
            ]
        })
        
        # Cell 3: Error analysis tables as markdown
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Error Analysis"]
        })
        
        # Generate HTML tables from error_analysis DataFrame
        if hasattr(self, 'errors') and self.errors is not None and not self.errors.empty:
            # Determine which row types exist
            row_types = self.errors.index.get_level_values('Subsystem').unique()
            
            for row_type in row_types:
                if row_type in ['Energy', 'Force', 'Stress']:
                    # Create a markdown cell for this subsystem
                    subsystem_lines = [f"### {row_type}\n\n"]
                    
                    # Get data for this row type
                    try:
                        df_subset = self.errors.xs(row_type, level='Subsystem')
                        
                        # Process weighted and unweighted separately
                        for weighting in ['Unweighted', 'weighted']:
                            try:
                                df_weight = df_subset.xs(weighting, level='Weighting')
                                
                                # Pivot to get training and testing side by side
                                df_pivot = df_weight.reset_index()
                                
                                # Create separate dataframes for training and testing
                                df_train = df_pivot[df_pivot['Testing'] == 'Training'].set_index('Group')
                                df_test = df_pivot[df_pivot['Testing'] == 'Testing'].set_index('Group')
                                
                                # Merge them
                                df_combined = df_train[['ncount', 'mae', 'rmse', 'rsq']].join(
                                    df_test[['mae', 'rmse', 'rsq']],
                                    how='outer',
                                    rsuffix='_test'
                                )
                                
                                # Sort by Test RMSE descending, but keep *ALL at top
                                all_key = '*ALL'
                                if all_key in df_combined.index:
                                    df_all = df_combined.loc[[all_key]]
                                    df_rest = df_combined.drop(all_key)
                                else:
                                    df_all = pd.DataFrame()
                                    df_rest = df_combined
                                
                                df_rest = df_rest.sort_values('rmse_test', ascending=False)
                                df_combined = pd.concat([df_all, df_rest])
                                
                                # Build HTML table in booktabs style
                                subsystem_lines.append(f"**{weighting.capitalize()} Metrics**\n\n")
                                
                                html = '<table style="border-collapse: collapse; table-layout: fixed; width: 100%; font-size: 14px;">\n'
                                
                                # Header row with toprule
                                html += '  <thead>\n'
                                html += '    <tr style="border-top: 2px solid black; border-bottom: 2px solid black;">\n'
                                html += '      <th style="text-align: left; padding: 8px 12px; width: 20%;">Group</th>\n'
                                html += '      <th style="text-align: center; padding: 8px 12px; width: 5%;">N</th>\n'
                                html += '      <th style="text-align: right; padding: 8px 12px; width: 11%;">MAE</th>\n'
                                html += '      <th style="text-align: right; padding: 8px 12px; width: 11%;">RMSE</th>\n'
                                html += '      <th style="text-align: right; padding: 8px 12px; width: 11%;">R²</th>\n'
                                html += '      <th style="text-align: right; padding: 8px 12px; width: 11%;">MAE</th>\n'
                                html += '      <th style="text-align: right; padding: 8px 12px; width: 11%;">RMSE</th>\n'
                                html += '      <th style="text-align: right; padding: 8px 12px; width: 11%;">R²</th>\n'
                                html += '    </tr>\n'
                                html += '  </thead>\n'
                                
                                # Body with Training/Testing labels and cmidrules
                                html += '  <tbody>\n'
                                html += '    <tr>\n'
                                html += '      <td style="padding: 4px 12px;"></td>\n'
                                html += '      <td style="padding: 4px 12px;"></td>\n'
                                html += '      <td colspan="3" style="text-align: center; padding: 4px 12px; font-style: italic; border-bottom: 1px solid #999;">Training</td>\n'
                                html += '      <td colspan="3" style="text-align: center; padding: 4px 12px; font-style: italic; border-bottom: 1px solid #999;">Testing</td>\n'
                                html += '    </tr>\n'
                                
                                # Helper function to format numbers
                                def format_value(val):
                                    # Handle NaN values
                                    if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                                        return "-"
                                    if abs(val) < 1e-6 and val != 0:
                                        # Use exponential notation: 1.23e-07 (8 chars like 0.123456)
                                        return f"{val:.2e}"
                                    else:
                                        return f"{val:.6f}"
                                
                                # Data rows
                                for idx, row in df_combined.iterrows():
                                    # Check if this is the ALL row
                                    is_all = (idx == '*ALL')
                                    
                                    # Clean group name and format
                                    group_name = idx.replace('*', '')
                                    if is_all:
                                        group_display = f'<strong>{group_name}</strong>'
                                        row_style = 'font-weight: bold;'
                                    else:
                                        group_display = f'&nbsp;&nbsp;{group_name}'
                                        row_style = ''
                                    
                                    html += f'    <tr style="{row_style}">\n'
                                    html += f'      <td style="text-align: left; padding: 4px 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{group_display}</td>\n'
                                    html += f'      <td style="text-align: center; padding: 4px 12px; border-left: 5px solid white;">{int(row["ncount"])}</td>\n'
                                    html += f'      <td style="text-align: right; padding: 4px 12px; font-family: monospace; border-left: 5px solid white;">{format_value(row["mae"])}</td>\n'
                                    html += f'      <td style="text-align: right; padding: 4px 12px; font-family: monospace;">{format_value(row["rmse"])}</td>\n'
                                    html += f'      <td style="text-align: right; padding: 4px 12px 4px 16px; font-family: monospace;">{format_value(row["rsq"])}</td>\n'
                                    html += f'      <td style="text-align: right; padding: 4px 12px; font-family: monospace; border-left: 5px solid white;">{format_value(row["mae_test"])}</td>\n'
                                    html += f'      <td style="text-align: right; padding: 4px 12px; font-family: monospace;">{format_value(row["rmse_test"])}</td>\n'
                                    html += f'      <td style="text-align: right; padding: 4px 12px 4px 16px; font-family: monospace;">{format_value(row["rsq_test"])}</td>\n'
                                    html += '    </tr>\n'
                                
                                # Bottomrule
                                html += '    <tr style="border-bottom: 2px solid black;">\n'
                                html += '      <td colspan="8"></td>\n'
                                html += '    </tr>\n'
                                
                                html += '  </tbody>\n'
                                html += '</table>\n\n'
                                
                                subsystem_lines.append(html)
                                subsystem_lines.append("\n")
                                
                            except KeyError:
                                # This weighting type doesn't exist
                                pass
                                
                    except KeyError:
                        # This row type doesn't exist
                        pass
                    
                    # Add this subsystem's markdown cell
                    notebook["cells"].append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": subsystem_lines
                    })
        
        # Cell 4: Load gamma history
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Gamma Evolution Heatmaps"]
        })
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pickle\n",
                "import numpy as np\n",
                "\n",
                f"# Load gamma history and feature ranks\n",
                f"with open('{gamma_history_file}', 'rb') as f:\n",
                "    data = pickle.load(f)\n",
                "\n",
                "# Handle both old and new pickle formats\n",
                "if isinstance(data, dict):\n",
                "    gamma_history = data['gamma_history']\n",
                "    feature_ranks = data.get('feature_ranks', None)\n",
                "else:\n",
                "    # Old format: just gamma_history\n",
                "    gamma_history = data\n",
                "    feature_ranks = None\n",
                "\n",
                "gamma_array = np.array(gamma_history)\n",
                "n_iterations, n_features = gamma_array.shape\n",
                "print(f'Loaded gamma history: {n_iterations} iterations, {n_features} features')\n",
                "\n",
                "if feature_ranks is not None:\n",
                "    print(f'Feature ranks available: {len(feature_ranks)} features')\n",
                "    unique_ranks = sorted(set(feature_ranks))\n",
                "    print(f'PACE ranks present: {unique_ranks}')\n",
                "    for rank in unique_ranks:\n",
                "        count = sum(1 for r in feature_ranks if r == rank)\n",
                "        print(f'  Rank {rank}: {count} functions')\n",
                "else:\n",
                "    print('Warning: Feature ranks not available, will use approximate division')"
            ]
        })
        
        # Cell 5: No longer needed - ranks come from pickle
        # Skip this cell since we load feature_ranks directly
        
        # Cell 6: Create and display heatmaps
        # We need to execute this cell and capture the output
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Load gamma history to create plots - use actual feature ranks
        import pickle
        with open(gamma_history_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            gamma_history = data['gamma_history']
            feature_ranks = data.get('feature_ranks', None)
        else:
            gamma_history = data
            feature_ranks = None
        
        gamma_array = np.array(gamma_history)
        n_iterations, n_features = gamma_array.shape
        
        # Determine which ranks to plot
        if feature_ranks is not None:
            unique_ranks = sorted(set(feature_ranks))
            feature_ranks_array = np.array(feature_ranks)
        else:
            # Fallback: assume 4 ranks with equal division
            unique_ranks = [1, 2, 3, 4]
            features_per_rank = n_features // 4
            feature_ranks_array = np.repeat(unique_ranks, features_per_rank)
            if len(feature_ranks_array) < n_features:
                feature_ranks_array = np.concatenate([feature_ranks_array, 
                    np.full(n_features - len(feature_ranks_array), unique_ranks[-1])])
        
        # Create the heatmap plot - 4x1 layout with custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create custom colormap: white for 0, turbo for >0
        turbo = plt.cm.turbo
        colors = [(1, 1, 1, 1)] + [turbo(i) for i in range(1, turbo.N)]
        custom_cmap = LinearSegmentedColormap.from_list('custom_turbo', colors, N=256)
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # Create heatmaps for each rank
        for i, rank in enumerate(unique_ranks[:4]):
            ax = axes[i]
            # Filter features by rank
            rank_mask = feature_ranks_array == rank
            rank_gamma = gamma_array[:, rank_mask]
            
            im = ax.imshow(rank_gamma, aspect='auto', cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)
            ax.set_title(f'PACE Rank {rank} Gamma Evolution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Index', fontsize=12)
            ax.set_ylabel('Iteration', fontsize=12)
            
            # Add statistics
            final_gamma = rank_gamma[-1, :]
            n_active_final = np.sum(final_gamma > 0.01)
            ax.text(0.02, 0.98, f'Active: {n_active_final}/{rank_gamma.shape[1]}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add single colorbar at bottom
        fig.subplots_adjust(bottom=0.08)
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Gamma Value (white=0, removed features)', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        # Save to buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": img_base64
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "from matplotlib.colors import LinearSegmentedColormap\n",
                "\n",
                "# Create custom colormap: white for 0 (removed features), turbo for >0\n",
                "turbo = plt.cm.turbo\n",
                "colors = [(1, 1, 1, 1)] + [turbo(i) for i in range(1, turbo.N)]\n",
                "custom_cmap = LinearSegmentedColormap.from_list('custom_turbo', colors, N=256)\n",
                "\n",
                "# Determine which ranks to plot\n",
                "if feature_ranks is not None:\n",
                "    unique_ranks = sorted(set(feature_ranks))\n",
                "    feature_ranks_array = np.array(feature_ranks)\n",
                "else:\n",
                "    # Fallback: assume 4 ranks with equal division\n",
                "    unique_ranks = [1, 2, 3, 4]\n",
                "    features_per_rank = n_features // 4\n",
                "    feature_ranks_array = np.repeat(unique_ranks, features_per_rank)\n",
                "    if len(feature_ranks_array) < n_features:\n",
                "        feature_ranks_array = np.concatenate([feature_ranks_array,\n",
                "            np.full(n_features - len(feature_ranks_array), unique_ranks[-1])])\n",
                "\n",
                "# Create 4x1 layout for PACE ranks\n",
                "fig, axes = plt.subplots(4, 1, figsize=(16, 16))\n",
                "\n",
                "# Create heatmaps for each rank\n",
                "for i, rank in enumerate(unique_ranks[:4]):\n",
                "    ax = axes[i]\n",
                "    # Filter features by rank\n",
                "    rank_mask = feature_ranks_array == rank\n",
                "    rank_gamma = gamma_array[:, rank_mask]\n",
                "    \n",
                "    # Create heatmap\n",
                "    im = ax.imshow(rank_gamma, aspect='auto', cmap=custom_cmap, interpolation='nearest', vmin=0, vmax=1)\n",
                "    ax.set_title(f'PACE Rank {rank} Gamma Evolution', fontsize=14, fontweight='bold')\n",
                "    ax.set_xlabel('Feature Index', fontsize=12)\n",
                "    ax.set_ylabel('Iteration', fontsize=12)\n",
                "    \n",
                "    # Add statistics\n",
                "    final_gamma = rank_gamma[-1, :]\n",
                "    n_active_final = np.sum(final_gamma > 0.01)\n",
                "    ax.text(0.02, 0.98, f'Active: {n_active_final}/{rank_gamma.shape[1]}',\n",
                "            transform=ax.transAxes, fontsize=10, verticalalignment='top',\n",
                "            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n",
                "\n",
                "# Add single horizontal colorbar at bottom\n",
                "fig.subplots_adjust(bottom=0.08)\n",
                "cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])\n",
                "cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')\n",
                "cbar.set_label('Gamma Value (white=0, removed features)', fontsize=12)\n",
                "\n",
                "plt.tight_layout(rect=[0, 0.04, 1, 1])\n",
                "plt.show()"
            ]
        })
        
        # Cell 7: Summary statistics with plots
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Summary Statistics"]
        })
        
        # Create summary plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Active features over iterations
        active_counts = [np.sum(gamma_array[i, :] > 0.01) for i in range(n_iterations)]
        ax1.plot(active_counts, linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Number of Active Features', fontsize=12)
        ax1.set_title('Feature Pruning Progress', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Gamma distribution at final iteration
        final_gamma = gamma_array[-1, :]
        ax2.hist(final_gamma[final_gamma > 0], bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Gamma Value', fontsize=12)
        ax2.set_ylabel('Number of Features', fontsize=12)
        ax2.set_title('Final Gamma Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64_summary = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Create text output for statistics
        stats_text = f"Feature Selection Summary:\nTotal features: {n_features}\n"
        for iteration in [0, n_iterations//2, n_iterations-1]:
            gamma_at_iter = gamma_array[iteration, :]
            n_active = np.sum(gamma_at_iter > 0.01)
            stats_text += f"\nIteration {iteration}:\n"
            stats_text += f"  Active features: {n_active}/{n_features} ({100*n_active/n_features:.1f}%)\n"
            stats_text += f"  Gamma range: [{gamma_at_iter.min():.4f}, {gamma_at_iter.max():.4f}]\n"
            stats_text += f"  Gamma mean: {gamma_at_iter.mean():.4f}\n"
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [stats_text]
                },
                {
                    "data": {
                        "image/png": img_base64_summary
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "\n",
                "# Compute summary statistics across all ranks\n",
                "print('Feature Selection Summary:')\n",
                "print(f'Total features: {n_features}')\n",
                "\n",
                "for iteration in [0, n_iterations//2, n_iterations-1]:\n",
                "    gamma_at_iter = gamma_array[iteration, :]\n",
                "    n_active = np.sum(gamma_at_iter > 0.01)\n",
                "    print(f'\\nIteration {iteration}:')\n",
                "    print(f'  Active features: {n_active}/{n_features} ({100*n_active/n_features:.1f}%)') \n",
                "    print(f'  Gamma range: [{gamma_at_iter.min():.4f}, {gamma_at_iter.max():.4f}]')\n",
                "    print(f'  Gamma mean: {gamma_at_iter.mean():.4f}')\n",
                "\n",
                "# Plot feature selection trajectory\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Active features over iterations\n",
                "active_counts = [np.sum(gamma_array[i, :] > 0.01) for i in range(n_iterations)]\n",
                "ax1.plot(active_counts, linewidth=2)\n",
                "ax1.set_xlabel('Iteration', fontsize=12)\n",
                "ax1.set_ylabel('Number of Active Features', fontsize=12)\n",
                "ax1.set_title('Feature Pruning Progress', fontsize=14, fontweight='bold')\n",
                "ax1.grid(True, alpha=0.3)\n",
                "\n",
                "# Gamma distribution at final iteration\n",
                "final_gamma = gamma_array[-1, :]\n",
                "ax2.hist(final_gamma[final_gamma > 0], bins=50, edgecolor='black', alpha=0.7)\n",
                "ax2.set_xlabel('Gamma Value', fontsize=12)\n",
                "ax2.set_ylabel('Number of Features', fontsize=12)\n",
                "ax2.set_title('Final Gamma Distribution', fontsize=14, fontweight='bold')\n",
                "ax2.grid(True, alpha=0.3, axis='y')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        })
        
        # Write notebook to file
        with open(notebook_file, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        self.pt.single_print(f"Created validation notebook: {notebook_file}")
            
            
