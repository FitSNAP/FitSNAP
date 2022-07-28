from ..io.input import config
from ..parallel_tools import pt
import numpy as np
from pandas import DataFrame



class Solver:

    def __init__(self, name, linear=True):
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
        num_types = config.sections["BISPECTRUM"].numtypes
        if num_types > 1:
            self.fit = self.fit.reshape(num_types, config.sections["BISPECTRUM"].ncoeff)
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
        assert not (self.linear and config.sections['CALCULATOR'].per_atom_energy and config.args.perform_fit)


    @pt.rank_zero
    def error_analysis(self):
        if not self.linear:
            pt.single_print("No Error Analysis for non-linear potentials")
            return
        self.df = DataFrame(pt.shared_arrays['a'].array)
        self.df['truths'] = pt.shared_arrays['b'].array.tolist()
        self.df['preds'] = pt.shared_arrays['a'].array @ self.fit
        self.df['weights'] = pt.shared_arrays['w'].array.tolist()
        for key in pt.fitsnap_dict.keys():
            if isinstance(pt.fitsnap_dict[key], list) and len(pt.fitsnap_dict[key]) == len(self.df.index):
                self.df[key] = pt.fitsnap_dict[key]
        if config.sections["EXTRAS"].dump_dataframe:
            self.df.to_pickle(config.sections['EXTRAS'].dataframe_file)
        for option in ["Unweighted", "Weighted"]:
            self.weighted = option
            self._all_error()
            self._group_error()
            if config.sections["SOLVER"].detailed_errors:
                self._config_error()

        if self.template_error is True:
            self._template_error()

        self.errors = DataFrame.from_records(self.errors)
        self.errors = self.errors.set_index(["Group", "Weighting", "Subsystem", ]).sort_index()

        if config.sections["CALCULATOR"].calculator == "LAMMPSSNAP" and config.sections["BISPECTRUM"].bzeroflag:
            self._offset()

    def _all_error(self):
        if config.sections["CALCULATOR"].energy:
            self._errors("*ALL", "Energy", (self.df['Row_Type'] == 'Energy'))
        if config.sections["CALCULATOR"].force:
            self._errors("*ALL", "Force", (self.df['Row_Type'] == 'Force'))
        if config.sections["CALCULATOR"].stress:
            self._errors("*ALL", "Stress", (self.df['Row_Type'] == 'Stress'))

    def _group_error(self):
        groups = set(pt.fitsnap_dict["Groups"])
        if config.sections["CALCULATOR"].energy:
            energy_filter = self.df['Row_Type'] == 'Energy'
        if config.sections["CALCULATOR"].force:
            force_filter = self.df['Row_Type'] == 'Force'
        if config.sections["CALCULATOR"].stress:
            stress_filter = self.df['Row_Type'] == 'Stress'
        for group in groups:
            group_filter = self.df['Groups'] == group
            if config.sections["CALCULATOR"].energy:
                self._errors(group, "Energy", group_filter & energy_filter)
            if config.sections["CALCULATOR"].force:
                self._errors(group, "Force", group_filter & force_filter)
            if config.sections["CALCULATOR"].stress:
                self._errors(group, "Stress", group_filter & stress_filter)

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
        this_true, this_pred = self.df['truths'][indices], self.df['preds'][indices]
        this_a = self.df.iloc[:, 0:pt.shared_arrays['a'].array.shape[1]].loc[indices].to_numpy()
        if self.fit_sam is not None:  ## should be correctly updated for current internal data format
            if self.cov is None:
                pf_std = np.std(self.fit_sam @ this_a.T, axis=0)
            else:
                # need to see which method is faster, both give the same results within numerical errors
                # Method 1
                chol = np.linalg.cholesky(self.cov)   #TODO: update for robustness - get an error when cov was backfilled with 0s (multi-element with different 2J max values)
                mat = this_a @ chol
                pf_std = np.linalg.norm(mat, axis=1)
                # Method 2
                # tmp = np.dot(a_err, self.cov)
                # pf_std = np.empty(a_err.shape[0])
                # for ipt in range(a_err.shape[0]):
                #     pf_std[ipt] = np.sqrt(np.dot(tmp[ipt, :], a_err[ipt, :]))
                # print(np.linalg.norm(pf_std-pf_std0), np.linalg.norm(pf_std0), np.linalg.norm(pf_std))
        else:
            pf_std = np.zeros(this_a.shape[0])
        if self.weighted == 'Weighted':
            w = pt.shared_arrays['w'].array[indices]
            this_true, this_pred = w * this_true, w * this_pred
            nconfig = np.count_nonzero(w)
        else:
            nconfig = len(this_pred)
        res = this_true - this_pred
        mae = np.sum(np.abs(res) / nconfig)
        ssr = np.square(res).sum()
        mse = ssr / nconfig
        rmse = np.sqrt(mse)
        rsq = 1 - ssr / np.sum(np.square(this_true - (this_true / nconfig).sum()))
        error_record = {
            "Group": group,
            "Weighting": self.weighted,
            "Subsystem": rtype,
            "ncount": nconfig,
            "mae": mae,
            "rmse": rmse,
            "rsq": rsq
        }

        if self.residuals is not None:
            error_record["residual"] = res
        self.errors.append(error_record)

        if config.sections["EXTRAS"].plot > 0:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import logging
            logging.getLogger('matplotlib').setLevel(logging.ERROR)
            mpl.rc('axes', linewidth=2, grid=True, labelsize=16)
            mpl.rc('figure', max_open_warning=500)

            plt.figure(figsize=(8,8))

            if config.sections["EXTRAS"].plot == 1:
                plt.plot(this_true, this_pred, 'ro', markersize=11, markeredgecolor='w')
            elif config.sections["EXTRAS"].plot > 1:
                plt.errorbar(this_true, this_pred, yerr=pf_std, fmt = 'ro', ecolor='r', elinewidth=2, markersize=11, markeredgecolor='w')

            xmin, xmax = min(np.min(this_true),np.min(this_pred)), max(np.max(this_true),np.max(this_pred))
            plt.plot([xmin, xmax], [xmin, xmax], 'k--', linewidth=1)
            #plt.xlim([xmin, xmax])
            #plt.ylim([xmin, xmax])
            plt.xlabel('DFT')
            plt.ylabel('SNAP')
            plt.title(group+'; '+rtype+'; '+self.weighted)
            plt.gcf().tight_layout()
            plt.savefig('dm_'+group+'_'+rtype+'_'+self.weighted+'.png')
            plt.clf()

            if config.sections["EXTRAS"].plot > 0:
                import matplotlib as mpl
                import matplotlib.pyplot as plt
                import logging
                logging.getLogger('matplotlib').setLevel(logging.ERROR)
                logging.getLogger('matplotlib.font_manager').disabled = True
                mpl.rc('axes', linewidth=2, grid=True, labelsize=16)
                mpl.rc('figure', max_open_warning=500)

                plt.figure(figsize=(8,8))

                if config.sections["EXTRAS"].plot == 1:
                    plt.plot(true, pred, 'ro', markersize=11, markeredgecolor='w')
                elif config.sections["EXTRAS"].plot > 1:
                    plt.errorbar(true, pred, yerr=pf_std, fmt = 'ro', ecolor='r', elinewidth=2, markersize=11, markeredgecolor='w')

                xmin, xmax = min(np.min(true),np.min(pred)), max(np.max(true),np.max(pred))
                plt.plot([xmin, xmax], [xmin, xmax], 'k--', linewidth=1)
                #plt.xlim([xmin, xmax])
                #plt.ylim([xmin, xmax])
                plt.xlabel('DFT')
                plt.ylabel('SNAP')
                plt.title(group+'; '+gtype+'; '+self.weighted)
                plt.gcf().tight_layout()
                plt.savefig('dm_'+group+'_'+gtype+'_'+self.weighted+'.png')
                plt.clf()

    def _template_error(self):
        pass
