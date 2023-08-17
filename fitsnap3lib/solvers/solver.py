from fitsnap3lib.io.input import Config
from fitsnap3lib.parallel_tools import ParallelTools
import numpy as np
from pandas import DataFrame, Series, concat
import pickle

#pt = ParallelTools()
#config = Config()

class Solver:
    """
    This class declares the method to solve the machine learning problem, e.g. linear regression,
    nonlinear regression, etc.

    Attributes:
        fit: Numpy array containing coefficients of fit.
    """

    def __init__(self, name, pt, config, linear=True):
        self.config = config #Config()
        self.pt = pt #ParallelTools()
        self.name = name
        self.configs = None
        self.fit = None
        self.all_fits = None
        self.template_error = False
        self.errors = []
        self.weighted = 'Unweighted'
        self.residuals = None
        self.a = None
        self.b = None
        self.w = None

        self.df = None
        self.linear = linear
        self.cov = None
        self.fit_sam = None
        self._checks()

    def perform_fit(self):
        pass

    def fit_gather(self):
        # self.all_fits = pt.gather_to_head_node(self.fit)
        pass

    def _offset(self):
        num_types = self.config.sections["BISPECTRUM"].numtypes
        if num_types > 1:
            self.fit = self.fit.reshape(num_types, self.config.sections["BISPECTRUM"].ncoeff)
            offsets = np.zeros((num_types, 1))
            self.fit = np.concatenate([offsets, self.fit], axis=1)
            self.fit = self.fit.reshape((-1, 1))
        else:
            self.fit = np.insert(self.fit, 0, 0)

        if self.fit_sam is not None:

            if num_types > 1:
                offsets = np.zeros((num_types, 1))
                nsam, ncf = self.fit_sam.shape

                fit_sam = np.empty((nsam, ncf + num_types))
                for isam, fit in enumerate(self.fit_sam.reshape(nsam, num_types, config.sections["BISPECTRUM"].ncoeff)):
                    fit = np.concatenate([offsets, fit], axis=1)
                    fit_sam[isam, :] = fit.reshape((-1,))

                self.fit_sam = fit_sam + 0.0

            else:
                self.fit_sam = np.insert(self.fit_sam, 0, 0, axis=1)



    def _checks(self):
        assert not (self.config.sections['CALCULATOR'].linear and self.config.sections['CALCULATOR'].per_atom_energy and self.config.args.perform_fit)

    def _ncount_mae_rmse_rsq_unweighted_and_weighted(self, g):
        """
        Calculate errors given a dataframe. The name of this function denotes the quantities it 
        returns as a Pandas series.

        Args:
            g: Pandas dataframe

        Returns:
            A Pandas series of floats, although some quantities like nconfig are cast to int later.
        """
        res = g['truths'] - g['preds']
        mae = np.mean(abs(res))
        ssr = np.square(res).sum()
        nconfig = len(g['truths'])
        mse = ssr / nconfig
        rmse = np.sqrt(mse)
        rsq = 1 - ssr / np.sum(np.square(g['truths'] - (g['truths'] / nconfig).sum()))
        w_res = g['weights'] * (g['truths'] - g['preds'])
        w_mae = np.mean(abs(w_res))
        w_ssr = np.square(w_res).sum()
        w_nconfig = np.count_nonzero(g['weights'])
        w_mse = w_ssr / w_nconfig
        w_rmse = np.sqrt(w_mse)
        w_rsq = 1 - w_ssr / np.sum(np.square((g['weights'] * g['truths']) - (g['weights'] * g['truths'] / w_nconfig).sum()))
        return Series({'ncount':nconfig, 'mae':mae, 'rmse':rmse, 'rsq':rsq, 'w_ncount':w_nconfig, 'w_mae':w_mae, 'w_rmse':w_rmse, 'w_rsq':w_rsq})

    
    #@pt.rank_zero
    def error_analysis(self, a=None, b=None, w=None, fs_dict=None):
        """
        If linear fit: extracts and stores fitting data, such as descriptor values, truths, and predictions, into
        a Pandas dataframe.

        If nonlinear fit: evaluate NN on all configurations to get truth values for error calculation.

        The optional arguments are for calculating errors on a given set of inputs. For linear models these inputs are 
        A matrix, truth array, weights, and a fs dictionary which contains group info. Care must be taken to ensure that 
        these data structures are already processed and lined up properly.

        Args:
            a: Optional A matrix numpy array.
            b: Optional truth matrix numpy array.
            w: Optional weight matrix numpy array.
            fs_dict: Optional fs dictionary from a `fs.pt.fitsnap_dict`
        """

        # Reset errors to default empty list.

        self.errors = []

        # Only calculate errors on rank 0 to reduce unnecessary allocations and calculations.

        if self.pt._rank == 0:
            
            # Proceed with nonlinear error analysis, if doing a fit.
            # If doing a fit, then self.configs is not None.

            if not self.linear and self.configs is not None:
                import torch # Needed to declare dtype. TODO: Move this into NN evaluate function.
                mae_f = {} # Force MAE of each group, train and test.
                mae_e = {} # Test energy MAE of each group, train and test.
                rmse_f = {} # Test force RMSE of each group, train and test.
                rmse_e = {} # Test energy RMSE of each group, train and test.

                count_train = {} # Nested dictionary with number of configs and atoms per group for training data.
                count_test = {} # Nested dictionary with number of configs and atoms per group for testing data.
                for group in self.config.sections['GROUPS'].group_table:
                    mae_f[group] = {}
                    mae_e[group] = {}
                    rmse_f[group] = {}
                    rmse_e[group] = {}

                    mae_f[group]["train"],  mae_f[group]["test"] =  0.0, 0.0
                    mae_e[group]["train"],  mae_e[group]["test"] =  0.0, 0.0
                    rmse_f[group]["train"], rmse_f[group]["test"] = 0.0, 0.0
                    rmse_e[group]["train"], rmse_e[group]["test"] = 0.0, 0.0

                    count_test[group] = {}
                    count_test[group]["nconfigs"] = 0 # Total number test configs in group.
                    count_test[group]["natoms"] = 0 # Total number test atoms in group.
                    count_train[group] = {}
                    count_train[group]["nconfigs"] = 0 # Total number test configs in group.
                    count_train[group]["natoms"] = 0 # Total number test atoms in group.

                # Add dictionary keys for total data '*ALL'

                mae_f['*ALL'] = {}
                mae_e['*ALL'] = {}
                rmse_f['*ALL'] = {}
                rmse_e['*ALL'] = {}
                mae_f['*ALL']["train"],  mae_f['*ALL']["test"] =  0.0, 0.0
                mae_e['*ALL']["train"],  mae_e['*ALL']["test"] =  0.0, 0.0
                rmse_f['*ALL']["train"], rmse_f['*ALL']["test"] = 0.0, 0.0
                rmse_e['*ALL']["train"], rmse_e['*ALL']["test"] = 0.0, 0.0
                count_test['*ALL'] = {}
                count_test['*ALL']["nconfigs"] = 0 # Total number test configs in group.
                count_test['*ALL']["natoms"] = 0 # Total number test atoms in group.
                count_train['*ALL'] = {}
                count_train['*ALL']["nconfigs"] = 0 # Total number test configs in group.
                count_train['*ALL']["natoms"] = 0 # Total number test atoms in group.

                if 'EXTRAS' in self.config.sections:
                    if (self.config.sections["EXTRAS"].dump_peratom):
                        fha = open(self.config.sections["EXTRAS"].peratom_file, 'w')
                        line = f"Filename Group AtomID Type Fx_Truth Fy_Truth Fz_Truth Fx_Pred Fy_Pred Fz_Pred Testing_Bool"
                        fha.write(line + "\n")
                    if (self.config.sections["EXTRAS"].dump_perconfig):
                        fhc = open(self.config.sections["EXTRAS"].perconfig_file, 'w')
                        line = f"Filename Group Natoms Energy_Truth Energy_Pred Testing_Bool"
                        fhc.write(line + "\n")
                atom_indx = 0
                m = 0
                for idx, c in enumerate(self.configs):
                    (energies_model, forces_model) = self.evaluate_configs(config_idx=idx, \
                                                                           standardize_bool=False, \
                                                                           dtype=torch.float64)
                    e_pred = energies_model.detach().numpy()/c.natoms # Model per-atom energy.
                    # Custom networks need a further index.
                    if (self.config.sections["CALCULATOR"].calculator == "LAMMPSCUSTOM"):
                        e_pred = e_pred[0]

                    ae = abs(c.energy - e_pred)
                    se = (c.energy - e_pred)**2
                    
                    if (c.testing_bool):
                        mae_e[c.group]["test"] += ae
                        rmse_e[c.group]["test"] += se
                        count_test[c.group]["nconfigs"] += 1
                        mae_e['*ALL']["test"] += ae
                        rmse_e['*ALL']["test"] += se
                        count_test['*ALL']["nconfigs"] += 1
                    else:
                        mae_e[c.group]["train"] += ae
                        rmse_e[c.group]["train"] += se
                        count_train[c.group]["nconfigs"] += 1
                        mae_e['*ALL']["train"] += ae
                        rmse_e['*ALL']["train"] += se
                        count_train['*ALL']["nconfigs"] += 1

                    if (self.config.sections["EXTRAS"].dump_perconfig):
                        line = f"{c.filename} {c.group} {c.natoms} {c.energy} {e_pred} {c.testing_bool}\n"
                        fhc.write(line)

                    if (forces_model is not None):
                        f_pred = forces_model.detach().numpy()
                        # Custom calculator returns Nx3 force array but we need 3*N here.
                        if (self.config.sections["CALCULATOR"].calculator == "LAMMPSCUSTOM"):
                            f_pred = f_pred.flatten()
                        for i in range(c.natoms):
                            fx_truth = c.forces[3*i+0]
                            fy_truth = c.forces[3*i+1]
                            fz_truth = c.forces[3*i+2]
                            fx_pred = f_pred[3*i+0]
                            fy_pred = f_pred[3*i+1]
                            fz_pred = f_pred[3*i+2]

                            ae = abs(fx_truth - fx_pred) + \
                                abs(fy_truth - fy_pred) + \
                                abs(fz_truth - fz_pred)
                            se = ((fx_truth - fx_pred)**2 + \
                                  (fy_truth - fy_pred)**2 + \
                                  (fz_truth - fz_pred)**2)

                            if (c.testing_bool):
                                mae_f[c.group]["test"] += ae
                                rmse_f[c.group]["test"] += se
                                count_test[c.group]["natoms"] += 1
                                mae_f['*ALL']["test"] += ae
                                rmse_f['*ALL']["test"] += se
                                count_test['*ALL']["natoms"] += 1
                            else:
                                mae_f[c.group]["train"] += ae
                                rmse_f[c.group]["train"] += se
                                count_train[c.group]["natoms"] += 1
                                mae_f['*ALL']["train"] += ae
                                rmse_f['*ALL']["train"] += se
                                count_train['*ALL']["natoms"] += 1
                            
                            if (self.config.sections["EXTRAS"].dump_peratom):
                                line = f"{c.filename} {c.group} {i+1} {int(c.types[i]+1)} "
                                line += f"{fx_truth} {fy_truth} {fz_truth} "
                                line += f"{fx_pred} {fy_pred} {fz_pred} "
                                line += f"{c.testing_bool}"
                                fha.write(line + "\n")
                            atom_indx += 1
                    m += 1
                if (self.config.sections["EXTRAS"].dump_perconfig):
                    fhc.close()
                if (self.config.sections["EXTRAS"].dump_peratom):
                    fha.close()

                # Normalize to get average errors.

                # Force MAE.
                mae_f['*ALL']["test"]   /= 3*count_test['*ALL']["natoms"]  if count_test[group]["natoms"]   > 0 else np.nan
                mae_f['*ALL']["train"]  /= 3*count_train['*ALL']["natoms"] if count_train[group]["natoms"]  > 0 else np.nan
                # Force RMSE. 
                rmse_f['*ALL']["test"]  /= 3*count_test['*ALL']["natoms"]  if count_test[group]["natoms"]   > 0 else np.nan
                rmse_f['*ALL']["test"]   = np.sqrt(rmse_f['*ALL']["test"])
                rmse_f['*ALL']["train"] /= 3*count_train['*ALL']["natoms"] if count_train[group]["natoms"]  > 0 else np.nan
                rmse_f['*ALL']["train"]  = np.sqrt(rmse_f['*ALL']["train"])
                # Energy MAE.
                mae_e['*ALL']["test"]   /= count_test['*ALL']["nconfigs"]  if count_test[group]["nconfigs"]  > 0 else np.nan
                mae_e['*ALL']["train"]  /= count_train['*ALL']["nconfigs"] if count_train[group]["nconfigs"] > 0 else np.nan
                # Energy RMSE.
                rmse_e['*ALL']["test"]  /= count_test['*ALL']["nconfigs"]  if count_test[group]["nconfigs"]  > 0 else np.nan
                rmse_e['*ALL']["test"]   = np.sqrt(rmse_e['*ALL']["test"])
                rmse_e['*ALL']["train"] /= count_train['*ALL']["nconfigs"] if count_train[group]["nconfigs"] > 0 else np.nan
                rmse_e['*ALL']["train"]  = np.sqrt(rmse_e['*ALL']["train"])
                for group in self.config.sections['GROUPS'].group_table:
                    # Force MAE
                    mae_f[group]["test"]   /= 3*count_test[group]["natoms"]  if count_test[group]["natoms"]    > 0  else np.nan
                    mae_f[group]["train"]  /= 3*count_train[group]["natoms"] if count_train[group]["natoms"]   > 0  else np.nan
                    # Force RMSE
                    rmse_f[group]["test"]  /= 3*count_test[group]["natoms"]  if count_test[group]["natoms"]    > 0  else np.nan
                    rmse_f[group]["test"]   = np.sqrt(rmse_f[group]["test"])
                    rmse_f[group]["train"] /= 3*count_train[group]["natoms"] if count_train[group]["natoms"]   > 0  else np.nan
                    rmse_f[group]["train"]  = np.sqrt(rmse_f[group]["train"])
                    # Energy MAE
                    mae_e[group]["test"]   /= count_test[group]["nconfigs"]  if count_test[group]["nconfigs"]  > 0  else np.nan
                    mae_e[group]["train"]  /= count_train[group]["nconfigs"] if count_train[group]["nconfigs"] > 0  else np.nan
                    # Energy RMSE
                    rmse_e[group]["test"]  /= count_test[group]["nconfigs"]  if count_test[group]["nconfigs"]  > 0  else np.nan
                    rmse_e[group]["test"]   = np.sqrt(rmse_e[group]["test"])
                    rmse_e[group]["train"] /= count_train[group]["nconfigs"] if count_train[group]["nconfigs"] > 0  else np.nan
                    rmse_e[group]["train"]  = np.sqrt(rmse_e[group]["train"])

                self.errors = (mae_f, mae_e, rmse_f, rmse_e, count_train, count_test)

                # Write pickled list of configs.

                if self.config.sections["EXTRAS"].dump_configs:
                    configs_file = self.config.sections['EXTRAS'].configs_file
                    with open(configs_file, 'wb') as f:
                        pickle.dump(self.configs, f)

                return
            
            # If nonlinear and not doing a fit, just create configs.

            elif not self.linear and self.configs is None:
                
                # Create list of Configuration objects.
                
                self.create_datasets()
                
                # Save a pickled list of Configuration objects.

                if self.config.sections["EXTRAS"].dump_configs:
                    configs_file = self.config.sections['EXTRAS'].configs_file
                    with open(configs_file, 'wb') as f:
                        pickle.dump(self.configs, f)

                return

            # Proceed with linear error analysis.
            # Collect remaining arrays to write dataframe.

            if (a is None and b is None and w is None and fs_dict is None):
                a = self.pt.shared_arrays['a'].array
                b = self.pt.shared_arrays['b'].array
                w = self.pt.shared_arrays['w'].array
                fs_dict = self.pt.fitsnap_dict

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
            if self.fit is not None and not self.config.sections["SOLVER"].true_multinode:

                # Return data for each group.

                grouped = self.df.groupby(['Groups', \
                    'Testing', \
                    'Row_Type']).apply(self._ncount_mae_rmse_rsq_unweighted_and_weighted)

                # reformat the weighted and unweighted data into separate rows

                grouped = concat({'Unweighted':grouped[['ncount', 'mae', 'rmse', 'rsq']], \
                    'weighted':grouped[['w_ncount', 'w_mae', 'w_rmse', 'w_rsq']].\
                        rename(columns={'w_ncount':'ncount', 'w_mae':'mae', 'w_rmse':'rmse', 'w_rsq':'rsq'})}, \
                    names=['Weighting']).reorder_levels(['Groups','Weighting','Testing', 'Row_Type']).sort_index()

                # return data for dataset as a whole

                all = self.df.groupby(['Testing', 'Row_Type']).\
                    apply(self._ncount_mae_rmse_rsq_unweighted_and_weighted)

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

                self.errors.index = self.errors.index.set_levels(['Testing' if e else 'Training' \
                    for e in self.errors.index.levels[2]], \
                        level=2)
            
            # Adjust coefficients for bzeroflag.
            if self.fit is not None: 
                if (self.config.sections["CALCULATOR"].calculator == "LAMMPSSNAP" and \
                    self.config.sections["BISPECTRUM"].bzeroflag):
                    self._offset()

    def _template_error(self):
        pass

    def _compute_stdev(self, a, method="chol"):
        if method == "sam":
            assert(self.fit_sam is not None)
            pf_stdev = np.std(self.fit_sam @ a.T, axis=0)
        elif method == "chol":
            assert(self.cov is not None)
            chol = np.linalg.cholesky(self.cov)
            mat = a @ chol
            pf_stdev = np.linalg.norm(mat, axis=1)
        elif method == "choleye":
            assert(self.cov is not None)
            eigvals = np.linalg.eigvalsh(self.cov)
            chol = np.linalg.cholesky(self.cov+(abs(eigvals[0]) + 1e-14) * np.eye(self.cov.shape[0]))
            mat = a @ chol
            pf_stdev = np.linalg.norm(mat, axis=1)
        elif method == "svd":
            assert(self.cov is not None)
            u, s, vh = np.linalg.svd(self.cov, hermitian=True)
            mat = (a @ u) @ np.sqrt(np.diag(s))
            pf_stdev = np.linalg.norm(mat, axis=1)
        elif method == "loop":
            assert(self.cov is not None)
            tmp = np.dot(a, self.cov)
            pf_stdev = np.empty(a.shape[0])
            for ipt in range(a.shape[0]):
                pf_stdev[ipt] = np.sqrt(np.dot(tmp[ipt, :], a[ipt, :]))
        elif method == "fullcov":
            assert(self.cov is not None)
            pf_stdev = np.sqrt(np.diag((a @ self.cov) @ a.T))
        else:
            pf_stdev = np.zeros(a.shape[0])

        return pf_stdev
