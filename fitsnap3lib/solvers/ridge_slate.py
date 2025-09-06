from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import DistributedList
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Import the SLATE module
SLATE_AVAILABLE = False
ridge_solve_qr = None

try:
    # Primary import method (after pip install -e .)
    from slate_wrapper import ridge_solve_qr
    SLATE_AVAILABLE = True
except ImportError as e:
    # Fallback: try direct path import for in-place builds
    try:
        import sys
        import os
        slate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib', 'slate_solver')
        if slate_path not in sys.path:
            sys.path.insert(0, slate_path)
        import slate_wrapper
        ridge_solve_qr = slate_wrapper.ridge_solve_qr
        SLATE_AVAILABLE = True
    except ImportError:
        print(f"Warning: SLATE module import failed: {e}")
        print("To install: cd fitsnap3lib/lib/slate_solver && pip install -e .")
        ridge_solve_qr = None
        SLATE_AVAILABLE = False

class RidgeSlate(Solver):
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
        
        # Get regularization parameter and tile size from RIDGE section
        if 'RIDGE' in self.config.sections:
            self.alpha = self.config.sections['RIDGE'].alpha
            self.tile_size = self.config.sections['RIDGE'].tile_size if hasattr(self.config.sections['RIDGE'], 'tile_size') else 256
        else:
            self.alpha = 1e-6
            self.tile_size = 256  # Default tile size for blocking

    def perform_fit(self):
        """
        Perform ridge regression fit on the linear system using SLATE for distributed computation.
        The fit is stored as a member `self.fit`.
        """
        
        pt = self.pt
        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        
        # Debug output - print all in one statement to avoid tangled output
        # *** DO NOT REMOVE !!! ***
        np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=np.inf)
        pt.sub_print(f"*** ------------------------\n"
                     #f"pt.fitsnap_dict['Testing']\n{pt.fitsnap_dict['Testing']}\n"
                     f"a\n{a}\n"
                     f"b {b}\n"
                     f"--------------------------------\n")
        
        pt.sub_barrier()
        
        # -------- LOCAL SLICE OF SHARED ARRAY AND REGULARIZATION ROWS --------

        start_idx, end_idx = pt.fitsnap_dict["sub_a_indices"]
        reg_row_idx = pt.fitsnap_dict["reg_row_idx"]
        reg_col_idx = pt.fitsnap_dict["reg_col_idx"]
        reg_num_rows = end_idx - reg_row_idx + 1
        #pt.all_print(f"pt.fitsnap_dict {pt.fitsnap_dict}")
        pt.all_print(f"*** start_idx {start_idx} end_idx {end_idx} reg_row_idx {reg_row_idx} reg_col_idx {reg_col_idx} reg_num_rows {reg_num_rows}")
        
        # -------- TRAINING/TESTING SPLIT --------
        
        if 'Testing' in pt.fitsnap_dict and pt.fitsnap_dict['Testing'] is not None:
            
            # set weights to 0 in place for testing rows
            testing_mask = pt.fitsnap_dict['Testing']
            for i in range(start_idx, reg_row_idx):
                if testing_mask[i]:
                    w[i] = 0.0

            testing_mask_local = testing_mask[start_idx:reg_row_idx]
            a_test_local = a[start_idx:reg_row_idx][testing_mask_local]
            b_test_local = b[start_idx:reg_row_idx][testing_mask_local]
            
            pt.all_print(f"***testing_mask_local {testing_mask_local}\n"
                f"a_test_local\n{a_test_local}\nb_test_local {b_test_local}")

        # -------- WEIGHTS --------
  
        # Apply weights in place to my slice
        a[start_idx:reg_row_idx] *= w[start_idx:reg_row_idx, np.newaxis]
        b[start_idx:reg_row_idx] *= w[start_idx:reg_row_idx]

        # -------- REGULARIZATION ROWS --------

        sqrt_alpha = np.sqrt(self.alpha)
        n = a.shape[1]
        a[reg_row_idx:end_idx+1,:] = 0
    
        for i in range(reg_num_rows):
            if reg_col_idx+i < n: # avoid out of bounds padding from multiple nodes
                a[reg_row_idx+i, reg_col_idx+i] = sqrt_alpha
            b[reg_row_idx+i] = 0.0

        # -------- SLATE AUGMENTED QR --------
        pt.sub_barrier() # make sure all sub ranks done filling local tiles
        m = a.shape[0] * self.pt._number_of_nodes # global matrix total rows
        lld = a.shape[0]  # local leading dimension column-major shared array
        ridge_solve_qr(a, b, m, lld, self.pt._comm)
        self.fit = b[:n]
        
        # Solution is available on all processes
        # *** DO NOT REMOVE !!! ***
        pt.all_print(f"*** self.fit ------------------------\n"
            f"{self.fit}\n-------------------------------------------------\n")
    
    
    def _dump_a(self):
        """Save the A matrix to file."""
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)


    def _dump_x(self):
        """Save the solution vector to file."""
        np.savez_compressed('x.npz', x=self.fit)


    def _dump_b(self):
        """Save the predicted values to file."""
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

"""
            self.df = DataFrame(a)
            self.df['truths'] = b.tolist()
            if self.fit is not None:
                self.df['preds'] = a @ self.fit
            self.df['weights'] = w.tolist()
            for key in fs_dict.keys():
                if isinstance(fs_dict[key], list) and \
                    len(fs_dict[key]) == len(self.df.index):
                    self.df[key] = fs_dict[key]
            if self.config.sections["EXTRAS"].dump_dataframe:
                self.df.to_pickle(self.config.sections['EXTRAS'].dataframe_file)

            # Proceed with error analysis if doing a fit.
            # if self.fit is not None and not self.config.sections["SOLVER"].multinode:
            if self.fit is not None:

                # Return data for each group.
                
                # resolve pandas FutureWarning by explicitly excluding the grouping columns
                # from the operation, which will be the default behavior in future versions

                grouped = self.df.groupby(['Groups', 'Testing', 'Row_Type']).apply(
                  self._ncount_mae_rmse_rsq_unweighted_and_weighted,
                  include_groups=False
                )

                # reformat the weighted and unweighted data into separate rows

                grouped = concat({'Unweighted':grouped[['ncount', 'mae', 'rmse', 'rsq']], \
                    'weighted':grouped[['w_ncount', 'w_mae', 'w_rmse', 'w_rsq']].\
                        rename(columns={'w_ncount':'ncount', 'w_mae':'mae', 'w_rmse':'rmse', 'w_rsq':'rsq'})}, \
                    names=['Weighting']).reorder_levels(['Groups','Weighting','Testing', 'Row_Type']).sort_index()

                # return data for dataset as a whole

                # resolve pandas FutureWarning by explicitly excluding the grouping columns
                # from the operation, which will be the default behavior in future versions

                all = self.df.groupby(['Testing', 'Row_Type']).apply(
                    self._ncount_mae_rmse_rsq_unweighted_and_weighted,
                    include_groups=False
                )

                # reformat the weighted and unweighted data into separate rows

                all = concat({'Unweighted':all[['ncount', 'mae', 'rmse', 'rsq']], \
                    'weighted':all[['w_ncount', 'w_mae', 'w_rmse', 'w_rsq']].\
                        rename(columns={'w_ncount':'ncount', 'w_mae':'mae', 'w_rmse':'rmse', 'w_rsq':'rsq'})}, \
                        names=['Weighting']).\
                            reorder_levels(['Weighting','Testing', 'Row_Type']).sort_index()

                # combine dataframes
                self.errors = concat([concat({'*ALL':all}, names=['Groups']), grouped])
                #print(self.errors['mae'].keys())
                #print(self.errors['mae'][('*ALL', 'Unweighted', False, 'Energy')])

                #assert(False)
                self.errors.ncount = self.errors.ncount.astype(int)
                self.errors.index.rename(["Group", "Weighting", "Testing", "Subsystem", ], inplace=True)

                # format for markdown printing
                self.errors.index = self.errors.index.set_levels(
                    ['Testing' if e else 'Training' for e in self.errors.index.levels[2]],
                    level=2)


"""

    def error_analysis(self):
    
    

        pt = self.pt

        # -------- LOCAL SLICE OF SHARED ARRAY AND REGULARIZATION ROWS --------

        a = pt.shared_arrays['a'].array
        b = pt.shared_arrays['b'].array
        w = pt.shared_arrays['w'].array
        start_idx, end_idx = pt.fitsnap_dict["sub_a_indices"]
        reg_row_idx = pt.fitsnap_dict["reg_row_idx"]
        reg_col_idx = pt.fitsnap_dict["reg_col_idx"]
        reg_num_rows = end_idx - reg_row_idx + 1
        #pt.all_print(f"pt.fitsnap_dict {pt.fitsnap_dict}")
        pt.all_print(f"*** start_idx {start_idx} end_idx {end_idx} reg_row_idx {reg_row_idx} reg_col_idx {reg_col_idx} reg_num_rows {reg_num_rows}")

        # -------- FIXME: HEY SONNET PUT YOUR CODE HERE --------
        
        
        
