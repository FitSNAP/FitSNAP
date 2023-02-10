from fitsnap3lib.io.input import Config
from fitsnap3lib.parallel_tools import ParallelTools
import numpy as np
from pandas import DataFrame, Series, concat

#pt = ParallelTools()
#config = Config()

class Solver:
    """
    This class declares the method to solve the machine learning problem, e.g. linear regression,
    nonlinear regression, etc.

    Attributes:
        fit: Numpy array containing coefficients of fit.
    """

    def __init__(self, name, linear=True):
        self.config = Config()
        self.pt = ParallelTools()
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
        Calculate errors given a dataframe.

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
    def error_analysis(self):
        """
        Extracts and stores fitting data, such as descriptor values, truths, and predictions, into
        a Pandas dataframe.
        """
        @self.pt.rank_zero
        def decorated_error_analysis():
            if not self.linear:
                self.pt.single_print("NN error analysis; using configuration class.")
                (energies_model, forces_model) = self.evaluate_configs(option=1, standardize_bool=False)
                print(len(energies_model))
                print(len(forces_model))
                #self.df = DataFrame(self.pt.shared_arrays['a'].array)
                #print(self.df)
                # make a list of per-atom quantities like type, i, forces, etc
                #fx_pred = [self.configs[m].forces[3*i+0] for i in range(self.configs[m].natoms) for m in range(len(self.configs))]
                fha = open("per-atom.dat", 'w')
                fhc = open("per-config.dat", 'w')
                atom_indx = 0
                m = 0
                for c in self.configs:
                    e_pred = energies_model[m].detach().numpy()
                    f_pred = forces_model[m].detach().numpy()
                    print(e_pred)
                    for i in range(c.natoms):
                        fx_truth = c.forces[3*i+0]
                        fy_truth = c.forces[3*i+1]
                        fz_truth = c.forces[3*i+2]
                        fx_pred = f_pred[3*i+0]
                        fy_pred = f_pred[3*i+1]
                        fz_pred = f_pred[3*i+2]
                        line = f"{fx_truth} {fy_truth} {fz_truth} "
                        line += f"{fx_pred} {fy_pred} {fz_pred}"
                        fha.write(line + "\n")
                        atom_indx += 1
                    m += 1
                fha.close()
                fhc.close()
                #print(self.configs)

                return

            # collect remaining arrays to write dataframe

            self.df = DataFrame(self.pt.shared_arrays['a'].array)
            self.df['truths'] = self.pt.shared_arrays['b'].array.tolist()
            if self.fit is not None:
                self.df['preds'] = self.pt.shared_arrays['a'].array @ self.fit
            self.df['weights'] = self.pt.shared_arrays['w'].array.tolist()
            for key in self.pt.fitsnap_dict.keys():
                if isinstance(self.pt.fitsnap_dict[key], list) and \
                    len(self.pt.fitsnap_dict[key]) == len(self.df.index):
                    self.df[key] = self.pt.fitsnap_dict[key]
            if self.config.sections["EXTRAS"].dump_dataframe:
                self.df.to_pickle(self.config.sections['EXTRAS'].dataframe_file)

            # proceed with error analysis if doing a fit

            if self.fit is not None:

                #return data for each group

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
                self.errors.ncount = self.errors.ncount.astype(int)
                self.errors.index.rename(["Group", "Weighting", "Testing", "Subsystem", ], inplace=True)

                # format for markdown printing

                self.errors.index = self.errors.index.set_levels(['Testing' if e else 'Training' \
                    for e in self.errors.index.levels[2]], \
                        level=2)
                
                if (self.config.sections["CALCULATOR"].calculator == "LAMMPSSNAP" and \
                    self.config.sections["BISPECTRUM"].bzeroflag):
                    self._offset()

        decorated_error_analysis()

    def _all_error(self):
        ## replaced by groupby().apply(ncount_mae_rmse_rsq_unweighted_and_weighted)
        # if self.config.sections["CALCULATOR"].energy:
        #     self._errors("*ALL", "Energy", (self.df['Row_Type'] == 'Energy'))
        # if self.config.sections["CALCULATOR"].force:
        #     self._errors("*ALL", "Force", (self.df['Row_Type'] == 'Force'))
        # if self.config.sections["CALCULATOR"].stress:
        #     self._errors("*ALL", "Stress", (self.df['Row_Type'] == 'Stress'))
        pass

    def _group_error(self):
        ## replaced by groupby().apply(ncount_mae_rmse_rsq_unweighted_and_weighted)
        # groups = set(self.pt.fitsnap_dict["Groups"])
        # if self.config.sections["CALCULATOR"].energy:
        #     energy_filter = self.df['Row_Type'] == 'Energy'
        # if self.config.sections["CALCULATOR"].force:
        #     force_filter = self.df['Row_Type'] == 'Force'
        # if self.config.sections["CALCULATOR"].stress:
        #     stress_filter = self.df['Row_Type'] == 'Stress'
        # for group in groups:
        #     group_filter = self.df['Groups'] == group
        #     if self.config.sections["CALCULATOR"].energy:
        #         self._errors(group, "Energy", group_filter & energy_filter)
        #     if self.config.sections["CALCULATOR"].force:
        #         self._errors(group, "Force", group_filter & force_filter)
        #     if self.config.sections["CALCULATOR"].stress:
        #         self._errors(group, "Stress", group_filter & stress_filter)
        pass
        
    def _config_error(self):
        # TODO: return normal functionality to detailed errors
        # configs = set(pt.fitsnap_dict["Configs"])
        # for this_config in configs:
        #     if config.sections["CALCULATOR"].energy:
        #         indices = (self.df['Configs'] == this_config) & (self.df['Row_Type'] == 'Energy')
        #         # self._errors(this_config, "Energy", indices)
        #     if config.sections["CALCULATOR"].force:
        #         indices = (self.df['Configs'] == this_config) & (self.df['Row_Type'] == 'Force')
        #         # self._errors(this_config, "Force", indices)
        #     if config.sections["CALCULATOR"].stress:
        #         indices = (self.df['Configs'] == this_config) & (self.df['Row_Type'] == 'Stress')
        #         # self._errors(this_config, "Stress", indices)
        pass


    def _errors(self, group, rtype, indices):
        ## replaced by groupby().apply(ncount_mae_rmse_rsq_unweighted_and_weighted)
        # this_true, this_pred = self.df['truths'][indices], self.df['preds'][indices]
        # #this_a = self.df.iloc[:, 0:self.pt.shared_arrays['a'].array.shape[1]].loc[indices].to_numpy()
        # if self.weighted == 'Weighted':
        #     w = self.pt.shared_arrays['w'].array[indices]
        #     this_true, this_pred = w * this_true, w * this_pred
        #     nconfig = np.count_nonzero(w)
        # else:
        #     nconfig = len(this_pred)
        # res = this_true - this_pred
        # mae = np.sum(np.abs(res) / nconfig)
        # ssr = np.square(res).sum()
        # mse = ssr / nconfig
        # rmse = np.sqrt(mse)
        # rsq = 1 - ssr / np.sum(np.square(this_true - (this_true / nconfig).sum()))
        # error_record = {
        #     "Group": group,
        #     "Weighting": self.weighted,
        #     "Subsystem": rtype,
        #     "ncount": nconfig,
        #     "mae": mae,
        #     "rmse": rmse,
        #     "rsq": rsq}
        # if self.residuals is not None:
        #     error_record["residual"] = res
        # self.errors.append(error_record)
        pass

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
