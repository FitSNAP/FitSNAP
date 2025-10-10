from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import DistributedList
import numpy as np

# --------------------------------------------------------------------------------------------

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Import the SLATE module
SLATE_AVAILABLE = False
slate_ridge_augmented_qr_cython = None
slate_ard_update_cython = None

try:
    # Primary import method (after pip install -e .)
    from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
    SLATE_AVAILABLE = True
except ImportError as e:
    # Fallback: try direct path import for in-place builds
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        from slate_wrapper import slate_ridge_augmented_qr_cython, slate_ard_update_cython
        SLATE_AVAILABLE = True
    except ImportError:
        print(f"Warning: SLATE module import failed: {e}")
        print("To install: cd fitsnap3lib/lib/slate_solver && pip install -e .")
        slate_ridge_augmented_qr_cython = None
        slate_ard_update_cython = None
        SLATE_AVAILABLE = False
        
# --------------------------------------------------------------------------------------------

class SlateCommon(Solver):
    """
    Multi-node Ridge regression solver using SLATE (Software for Linear Algebra Targeting Exascale).
    
    This solver leverages SLATE's distributed matrix operations to solve ridge regression
    problems across multiple nodes efficiently.
    
    Solves: (A^T A + alpha * I) x = A^T b
    """
    
    def __init__(self, name, pt, config):
        super().__init__(name, pt, config, linear=True)
        
        # Check that SLATE is available
        if not SLATE_AVAILABLE:
            error_msg = f"[Rank {self.pt._rank}, Node {self.pt._node_index}] SLATE module not available. Please compile it first."
            pt.single_print(error_msg)
            raise RuntimeError(error_msg)
        
        # Get parameters from SLATE section
        if 'SLATE' in self.config.sections:
        
            slate_config = self.config.sections['SLATE']
            
            # Method selection: RIDGE or ARD
            self.method = slate_config.method.upper()
        
            if self.method == 'RIDGE':
                self.alpha = slate_config.alpha
            elif self.method == 'ARD':
                self.max_iter = slate_config.max_iter
                self.tol = slate_config.tol
                
                # Store config parameters for adaptive hyperparameter computation
                self.directmethod = slate_config.directmethod
                self.scap = slate_config.scap
                self.scai = slate_config.scai
                self.logcut = slate_config.logcut
                self.alphabig = slate_config.alphabig
                self.alphasmall = slate_config.alphasmall
                self.lambdabig = slate_config.lambdabig
                self.lambdasmall = slate_config.lambdasmall
                self.threshold_lambda_config = slate_config.threshold_lambda
                
                # These will be set in perform_fit_ard based on data variance
                self.alpha_1 = None
                self.alpha_2 = None
                self.lambda_1 = None
                self.lambda_2 = None
                self.threshold_lambda = None
        
        
    # --------------------------------------------------------------------------------------------

    def error_analysis(self):
        """
        Scalable error analysis using hierarchical MPI reduction for group metrics.
        Implements the same logic as solver.py but distributed across multiple nodes.
        """

        pt = self.pt

        # -------- SCALABLE GROUP METRICS COMPUTATION --------
        
        if self.fit is not None:
            # Only compute error analysis if we have a fit
            # a, b, w are unchanged from original data (only aw, bw were modified by SLATE)
            
            fs_dict = pt.fitsnap_dict
            
            # Create DataFrame like the legacy solver does
            from pandas import DataFrame
            
            # Use only local slice (excluding regularization rows) for error analysis
            start_idx, end_idx = pt.fitsnap_dict["sub_a_indices"]
            local_slice = slice(start_idx, end_idx+1)
            local_a = pt.shared_arrays['a'].array[local_slice]
            local_b = pt.shared_arrays['b'].array[local_slice]
            local_w = pt.shared_arrays['w'].array[local_slice]
            
            df_local = DataFrame(local_a)
            df_local['truths'] = local_b.tolist()
            
            # Check for numerical issues before prediction
            if np.any(np.isnan(local_a)) or np.any(np.isinf(local_a)):
                self.pt.single_print(f"WARNING: NaN or Inf found in descriptor matrix (local_a)")
                self.pt.single_print(f"  NaN count: {np.sum(np.isnan(local_a))}, Inf count: {np.sum(np.isinf(local_a))}")
            if np.any(np.isnan(self.fit)) or np.any(np.isinf(self.fit)):
                self.pt.single_print(f"WARNING: NaN or Inf found in fit coefficients")
                self.pt.single_print(f"  NaN count: {np.sum(np.isnan(self.fit))}, Inf count: {np.sum(np.isinf(self.fit))}")
            
            # Compute predictions with NaN/Inf handling
            preds = local_a @ self.fit
            
            # Replace NaN/Inf with 0 for error analysis (or could skip these rows)
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                num_bad = np.sum(np.isnan(preds) | np.isinf(preds))
                self.pt.single_print(f"WARNING: {num_bad} NaN/Inf predictions found, replacing with corresponding truth values for error analysis")
                # Replace bad predictions with truth values (gives 0 error for those points)
                # Alternatively, you could filter these rows out entirely
                bad_mask = np.isnan(preds) | np.isinf(preds)
                preds[bad_mask] = local_b[bad_mask]
            
            df_local['preds'] = preds.tolist()
            df_local['weights'] = local_w.tolist()
            
            # Add metadata columns for local slice
            for key in ['Groups', 'Testing', 'Row_Type']:
                if key in fs_dict and isinstance(fs_dict[key], list):
                    local_values = fs_dict[key][local_slice]
                    df_local[key] = local_values
                else:
                    # Set defaults
                    if key == 'Groups':
                        df_local[key] = ['*ALL'] * len(df_local)
                    elif key == 'Testing':
                        df_local[key] = [False] * len(df_local)
                    elif key == 'Row_Type':
                        df_local[key] = ['Energy'] * len(df_local)
            
            # Convert local DataFrame to group metrics using legacy solver approach
            local_group_data = self._compute_local_group_sums_from_df(df_local)
            
            # Hierarchical reduction across all MPI ranks
            global_group_data = self._hierarchical_group_reduce(local_group_data, pt._comm)
            
            # Two-pass algorithm for exact R² calculation
            # First, add '*ALL' groups to global data on ALL ranks
            global_group_data_with_all = self._add_all_groups_to_global_data(global_group_data)
            
            if pt._rank == 0:
                final_results = self._compute_final_metrics_twopass_from_df(
                    global_group_data_with_all, df_local, pt._comm
                )
                
                # Convert to pandas DataFrame format matching solver.py
                self._format_results_as_dataframe(final_results)
            else:
                # Non-root ranks participate in two-pass but don't store results
                self._compute_final_metrics_twopass_from_df(
                    global_group_data_with_all, df_local, pt._comm
                )
                self.errors = []
    
    # --------------------------------------------------------------------------------------------

    def _compute_local_group_sums(self, groups, testing, row_types, truths, preds, weights):
        """Compute partial sums for each group on local data"""
        from collections import defaultdict
        
        local_group_data = defaultdict(lambda: {
            'n': 0,
            'sum_weights': 0.0,
            'sum_truths_weighted': 0.0,
            'sum_ae': 0.0,
            'sum_se': 0.0
        })
        
        for i in range(len(truths)):
            group_key = (groups[i], testing[i], row_types[i])
            
            weight = weights[i]
            truth = truths[i]
            pred = preds[i]
            
            stats = local_group_data[group_key]
            stats['n'] += 1
            stats['sum_weights'] += weight
            stats['sum_truths_weighted'] += weight * truth
            stats['sum_ae'] += weight * abs(truth - pred)
            stats['sum_se'] += weight * (truth - pred)**2
        
        return dict(local_group_data)

    # --------------------------------------------------------------------------------------------

    def _compute_local_group_sums_from_df(self, df_local):
        """Compute partial sums for each group from DataFrame (like legacy solver)"""
        from collections import defaultdict
        
        local_group_data = defaultdict(lambda: {
            'n': 0,
            'sum_weights': 0.0,
            'sum_truths_weighted': 0.0,
            'sum_ae': 0.0,
            'sum_se': 0.0,
            # Add unweighted sums for correct unweighted metrics
            'sum_truths_unweighted': 0.0,
            'sum_ae_unweighted': 0.0,
            'sum_se_unweighted': 0.0
        })
        
        for _, row in df_local.iterrows():
            group_key = (row['Groups'], row['Testing'], row['Row_Type'])
            
            weight = row['weights']
            truth = row['truths']
            pred = row['preds']
            
            stats = local_group_data[group_key]
            stats['n'] += 1
            
            # Weighted sums
            stats['sum_weights'] += weight
            stats['sum_truths_weighted'] += weight * truth
            stats['sum_ae'] += weight * abs(truth - pred)
            stats['sum_se'] += weight * (truth - pred)**2
            
            # Unweighted sums (ignore weights entirely)
            stats['sum_truths_unweighted'] += truth
            stats['sum_ae_unweighted'] += abs(truth - pred)
            stats['sum_se_unweighted'] += (truth - pred)**2
        
        return dict(local_group_data)

    # --------------------------------------------------------------------------------------------

    def _hierarchical_group_reduce(self, local_group_data, comm):
        """Hierarchical reduction to avoid O(P²) memory growth"""
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Use tree reduction: log(P) steps instead of P steps
        step = 1
        current_data = local_group_data.copy()
        
        while step < size:
            if rank % (2 * step) == 0:  # Receiver
                source = rank + step
                if source < size:
                    # Receive data from partner
                    partner_data = comm.recv(source=source, tag=0)
                    
                    # Merge with current data
                    for group_key, group_stats in partner_data.items():
                        if group_key not in current_data:
                            current_data[group_key] = group_stats.copy()
                        else:
                            # Aggregate the sums
                            for key in ['n', 'sum_weights', 'sum_truths_weighted', 'sum_ae', 'sum_se',
                                       'sum_truths_unweighted', 'sum_ae_unweighted', 'sum_se_unweighted']:
                                current_data[group_key][key] += group_stats[key]
            
            elif rank % (2 * step) == step:  # Sender
                dest = rank - step
                comm.send(current_data, dest=dest, tag=0)
                break  # This rank is done
            
            step *= 2
        
        # Broadcast final result from rank 0 to all ranks
        if rank == 0:
            final_data = current_data
        else:
            final_data = None
        
        final_data = comm.bcast(final_data, root=0)
        return final_data

    # --------------------------------------------------------------------------------------------

    def _compute_final_metrics_twopass(self, global_group_data, local_groups, local_testing, local_row_types, local_truths, local_weights, comm):
        """Two-pass algorithm for exact R² with minimal communication"""
        rank = comm.Get_rank()
        
        # Pass 1: Compute global means (already have this from reduction)
        global_means = {}
        for group_key, stats in global_group_data.items():
            if stats['sum_weights'] > 0:
                global_means[group_key] = stats['sum_truths_weighted'] / stats['sum_weights']
            else:
                global_means[group_key] = 0.0
        
        # Pass 2: Compute SS_tot using global means
        local_ss_tot = {}
        for i in range(len(local_truths)):
            group_key = (local_groups[i], local_testing[i], local_row_types[i])
            if group_key in global_means:
                weight = local_weights[i]
                truth = local_truths[i]
                global_mean = global_means[group_key]
                
                if group_key not in local_ss_tot:
                    local_ss_tot[group_key] = 0.0
                local_ss_tot[group_key] += weight * (truth - global_mean)**2
        
        # Reduce SS_tot values (much smaller than full data)
        global_ss_tot = self._hierarchical_reduce_dict(local_ss_tot, comm)
        
        # Compute final metrics (only on rank 0)
        if rank == 0:
            final_results = []
            for group_key, stats in global_group_data.items():
                if stats['sum_weights'] > 0:
                    mae = stats['sum_ae'] / stats['sum_weights']
                    rmse = np.sqrt(stats['sum_se'] / stats['sum_weights'])
                    
                    ss_tot = global_ss_tot.get(group_key, 0.0)
                    rsq = 1 - (stats['sum_se'] / ss_tot) if ss_tot != 0 else 0
                    
                    final_results.append({
                        'group': group_key,
                        'ncount': stats['n'],
                        'mae': mae,
                        'rmse': rmse,
                        'rsq': rsq,
                        '_sum_weights': stats['sum_weights'],
                        '_sum_ae': stats['sum_ae'],
                        '_sum_se': stats['sum_se'],
                        '_sum_ss_tot': ss_tot
                    })
            
            return final_results
        
        return None

    # --------------------------------------------------------------------------------------------

    def _compute_final_metrics_twopass_from_df(self, global_group_data, df_local, comm):
        """Two-pass algorithm for exact R² with DataFrame input"""
        rank = comm.Get_rank()
        
        # Pass 1: Compute global means (both weighted and unweighted)
        global_means_weighted = {}
        global_means_unweighted = {}
        
        for group_key, stats in global_group_data.items():
            # Calculate mean for all groups (including '*ALL')
            # Weighted mean
            if stats['sum_weights'] > 0:
                global_means_weighted[group_key] = stats['sum_truths_weighted'] / stats['sum_weights']
            else:
                global_means_weighted[group_key] = 0.0
            
            # Unweighted mean  
            if stats['n'] > 0:
                global_means_unweighted[group_key] = stats['sum_truths_unweighted'] / stats['n']
            else:
                global_means_unweighted[group_key] = 0.0
        
        # Pass 2: Compute SS_tot using global means
        local_ss_tot_weighted = {}
        local_ss_tot_unweighted = {}
        
        for _, row in df_local.iterrows():
            group_key = (row['Groups'], row['Testing'], row['Row_Type'])
            
            if group_key in global_means_weighted:
                weight = row['weights']
                truth = row['truths']
                
                # Weighted SS_tot for individual group
                weighted_mean = global_means_weighted[group_key]
                if group_key not in local_ss_tot_weighted:
                    local_ss_tot_weighted[group_key] = 0.0
                local_ss_tot_weighted[group_key] += weight * (truth - weighted_mean)**2
                
                # Unweighted SS_tot for individual group
                unweighted_mean = global_means_unweighted[group_key]
                if group_key not in local_ss_tot_unweighted:
                    local_ss_tot_unweighted[group_key] = 0.0
                local_ss_tot_unweighted[group_key] += (truth - unweighted_mean)**2
                
                # Also contribute to "*ALL" groups
                all_key = ('*ALL',) + group_key[1:]  # Replace Groups with '*ALL'
                
                if all_key in global_means_weighted:
                    # Weighted SS_tot for *ALL group
                    all_weighted_mean = global_means_weighted[all_key]
                    if all_key not in local_ss_tot_weighted:
                        local_ss_tot_weighted[all_key] = 0.0
                    local_ss_tot_weighted[all_key] += weight * (truth - all_weighted_mean)**2
                    
                    # Unweighted SS_tot for *ALL group
                    all_unweighted_mean = global_means_unweighted[all_key]
                    if all_key not in local_ss_tot_unweighted:
                        local_ss_tot_unweighted[all_key] = 0.0
                    local_ss_tot_unweighted[all_key] += (truth - all_unweighted_mean)**2
        
        # Reduce SS_tot values (much smaller than full data)
        global_ss_tot_weighted = self._hierarchical_reduce_dict(local_ss_tot_weighted, comm)
        global_ss_tot_unweighted = self._hierarchical_reduce_dict(local_ss_tot_unweighted, comm)
        
        # Compute final metrics (only on rank 0)
        if rank == 0:
            final_results = []
            for group_key, stats in global_group_data.items():
                if stats['n'] > 0:
                    # Weighted metrics (using proper statistical methodology)
                    weighted_mae = stats['sum_ae'] / stats['sum_weights'] if stats['sum_weights'] > 0 else 0
                    weighted_rmse = np.sqrt(stats['sum_se'] / stats['sum_weights']) if stats['sum_weights'] > 0 else 0
                    
                    ss_tot_weighted = global_ss_tot_weighted.get(group_key, 0.0)
                    weighted_rsq = 1 - (stats['sum_se'] / ss_tot_weighted) if ss_tot_weighted != 0 else 0
                    
                    # Unweighted metrics (standard calculation)
                    unweighted_mae = stats['sum_ae_unweighted'] / stats['n']
                    unweighted_rmse = np.sqrt(stats['sum_se_unweighted'] / stats['n'])
                    
                    ss_tot_unweighted = global_ss_tot_unweighted.get(group_key, 0.0)
                    unweighted_rsq = 1 - (stats['sum_se_unweighted'] / ss_tot_unweighted) if ss_tot_unweighted != 0 else 0
                    
                    final_results.append({
                        'group': group_key,
                        'ncount': stats['n'],
                        'weighted_mae': weighted_mae,
                        'weighted_rmse': weighted_rmse,
                        'weighted_rsq': weighted_rsq,
                        'unweighted_mae': unweighted_mae,
                        'unweighted_rmse': unweighted_rmse,
                        'unweighted_rsq': unweighted_rsq,
                        '_sum_weights': stats['sum_weights'],
                        '_sum_ae': stats['sum_ae'],
                        '_sum_se': stats['sum_se'],
                        '_sum_ss_tot_weighted': ss_tot_weighted,
                        '_sum_ss_tot_unweighted': ss_tot_unweighted
                    })
            
            return final_results
        
        return None

    # --------------------------------------------------------------------------------------------

    def _hierarchical_reduce_dict(self, local_dict, comm):
        """Reduce dictionary values hierarchically"""
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        step = 1
        current_dict = local_dict.copy()
        
        while step < size:
            if rank % (2 * step) == 0:
                source = rank + step
                if source < size:
                    partner_dict = comm.recv(source=source, tag=1)
                    for key, value in partner_dict.items():
                        current_dict[key] = current_dict.get(key, 0.0) + value
            elif rank % (2 * step) == step:
                dest = rank - step
                comm.send(current_dict, dest=dest, tag=1)
                break
            step *= 2
        
        return current_dict if rank == 0 else {}
 
    # --------------------------------------------------------------------------------------------

    def _format_results_as_dataframe(self, results):
        """Convert results to pandas DataFrame format matching solver.py"""
        from pandas import DataFrame, concat
        
        if not results:
            self.errors = DataFrame()
            return
        
        # Add '*ALL' groups by aggregating across Groups dimension
        all_results = self._add_all_groups(results)
        
        # Create both weighted and unweighted versions
        formatted_results = []
        
        for result in all_results:
            group_key = result['group']
            
            # Use the correctly computed weighted and unweighted metrics
            unweighted_mae = result['unweighted_mae']
            unweighted_rmse = result['unweighted_rmse']
            unweighted_rsq = result['unweighted_rsq']
            
            weighted_mae = result['weighted_mae']
            weighted_rmse = result['weighted_rmse']
            weighted_rsq = result['weighted_rsq']
            
            # Add both versions with proper indexing
            testing_str = 'Testing' if group_key[1] else 'Training'
            
            formatted_results.extend([
                {
                    'Groups': group_key[0],
                    'Weighting': 'Unweighted', 
                    'Testing': testing_str,
                    'Row_Type': group_key[2],
                    'ncount': result['ncount'],
                    'mae': unweighted_mae,
                    'rmse': unweighted_rmse,
                    'rsq': unweighted_rsq
                },
                {
                    'Groups': group_key[0],
                    'Weighting': 'weighted',
                    'Testing': testing_str, 
                    'Row_Type': group_key[2],
                    'ncount': result['ncount'],
                    'mae': weighted_mae,
                    'rmse': weighted_rmse,
                    'rsq': weighted_rsq
                }
            ])
        
        # Convert to DataFrame with proper MultiIndex
        df = DataFrame(formatted_results)
        if not df.empty:
            df = df.set_index(['Groups', 'Weighting', 'Testing', 'Row_Type']).sort_index()
            df.index.rename(["Group", "Weighting", "Testing", "Subsystem"], inplace=True)
        
        self.errors = df
  
    # --------------------------------------------------------------------------------------------

    def _add_all_groups(self, group_results):
        """Add '*ALL' groups - now just returns results since aggregation happens earlier"""
        return group_results
    
    # --------------------------------------------------------------------------------------------

    def _add_all_groups_to_global_data(self, global_group_data):
        """Add '*ALL' groups by aggregating raw sums at the global_group_data level"""
        
        # Organize by aggregation keys
        aggregations = {}
        
        for group_key, stats in global_group_data.items():
            # Skip if already an '*ALL' group
            if group_key[0] == '*ALL':
                continue
                
            # Create aggregation key: replace Groups with '*ALL'
            agg_key = ('*ALL',) + group_key[1:]
            
            if agg_key not in aggregations:
                aggregations[agg_key] = {
                    'n': 0,
                    'sum_weights': 0.0,
                    'sum_truths_weighted': 0.0,
                    'sum_ae': 0.0,
                    'sum_se': 0.0,
                    'sum_truths_unweighted': 0.0,
                    'sum_ae_unweighted': 0.0,
                    'sum_se_unweighted': 0.0
                }
            
            # Aggregate the raw sums
            agg = aggregations[agg_key]
            for key in ['n', 'sum_weights', 'sum_truths_weighted', 'sum_ae', 'sum_se',
                       'sum_truths_unweighted', 'sum_ae_unweighted', 'sum_se_unweighted']:
                agg[key] += stats[key]
        
        # Add aggregated groups to global data
        result = global_group_data.copy()
        result.update(aggregations)
        
        return result

    # --------------------------------------------------------------------------------------------

