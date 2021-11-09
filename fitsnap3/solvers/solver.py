from ..io.input import config
from ..parallel_tools import pt
import numpy as np
from pandas import DataFrame



class Solver:

    def __init__(self, name):
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
        self.cov = None
        self.fit_sam = None

    def perform_fit(self):
        pass

    def fit_gather(self):
        # self.all_fits = pt.allgather(self.fit)
        pass

    @pt.rank_zero
    def extras(self):
        length,width = 0,np.shape(pt.shared_arrays['a'].array)[1]
        if config.sections["CALCULATOR"].energy:
            a_e, b_e, w_e = self._make_abw(pt.shared_arrays['a'].energy_index, 1)
        else:
            a_e, b_e, w_e = np.zeros((length, width)),np.zeros((length,)),np.zeros((length,))
        if config.sections["CALCULATOR"].force:
            num_forces = np.array(pt.shared_arrays['a'].num_atoms)*3
            a_f, b_f, w_f = self._make_abw(pt.shared_arrays['a'].force_index, num_forces.tolist())
        else:
            a_f, b_f, w_f = np.zeros((length, width)),np.zeros((length,)),np.zeros((length,))
        if config.sections["CALCULATOR"].stress:
            a_s, b_s, w_s = self._make_abw(pt.shared_arrays['a'].stress_index, 6)
        else:
            a_s, b_s, w_s = np.zeros((length, width)),np.zeros((length,)),np.zeros((length,))
        if not config.sections["SOLVER"].detailed_errors:
            print(">>>Enable [SOLVER], detailed_errors = 1 to characterize the training/testing split of your output *.npy matricies")
        if config.sections["EXTRAS"].dump_a:
            # if config.sections["EXTRAS"].apply_transpose:
            #     np.save('Descriptors_Compact.npy', (np.concatenate((a_e,a_f,a_s),axis=0) @ np.concatenate((a_e,a_f,a_s),axis=0).T))
            # else:
            np.save('Descriptors.npy', np.concatenate([x for x in (a_e, a_f, a_s) if x.size > 0],axis=0))
        if config.sections["EXTRAS"].dump_b:
            np.save('Truth-Ref.npy', np.concatenate([x for x in (b_e, b_f, b_s) if x.size > 0],axis=0))
        if config.sections["EXTRAS"].dump_w:
            np.save('Weights.npy', np.concatenate([x for x in (w_e, w_f, w_s) if x.size > 0],axis=0))

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



    @pt.rank_zero
    def error_analysis(self):
        for option in ["Unweighted", "Weighted"]:
            self.weighted = option
            self._all_error()
            self._group_error()
            if config.sections["SOLVER"].detailed_errors:
                config_indicies,energy_list,force_list,stress_list = self._config_error()
#                from csv import writer
#                with open('detailed_config_list.dat', 'w') as f:
#                    writer = writer(f, delimiter=' ')
#                    writer.writerow(['FileID Usage N_Atoms FileName'])
#                    writer.writerows(config_indicies)

        if self.template_error is True:
            self._template_error()

        self.errors = DataFrame.from_records(self.errors)
        self.errors = self.errors.set_index(["Group", "Weighting", "Subsystem", ]).sort_index()

        if config.sections["CALCULATOR"].calculator == "LAMMPSSNAP" and config.sections["BISPECTRUM"].bzeroflag:
            self._offset()

    def _all_error(self):
        if config.sections["CALCULATOR"].energy:
            self._energy()
        if config.sections["CALCULATOR"].force:
            self._force()
        if config.sections["CALCULATOR"].stress:
            self._stress()
        #self._combined()

    def _energy(self):
        if config.sections["CALCULATOR"].energy:
            testing = -1 * pt.shared_arrays['configs_per_group'].testing
            a, b, w = self._make_abw(pt.shared_arrays['a'].energy_index, 1)
            config_indicies,energy_list,force_list,stress_list = self._config_error()
            self._errors([[0, testing]], ['*ALL'], "Energy", a, b, w)
            if config.sections["SOLVER"].detailed_errors and self.weighted == "Unweighted":
                from csv import writer
                true, pred = b, a @ self.fit
                ConfigType = ['Training'] * (np.shape(true)[0]-pt.shared_arrays['configs_per_group'].testing) + \
                                             ['Testing'] * (pt.shared_arrays['configs_per_group'].testing)
                with open('detailed_energy_errors.dat', 'w') as f:
                    writer = writer(f, delimiter=' ')
                    writer.writerow(['FileName Type True-Ref Predicted-Ref Difference(Pred-True)'])
                    writer.writerows(zip(energy_list,ConfigType,true, pred, pred-true))

            if testing != 0:
                self._errors([[testing, 0]], ['*ALL'], "Energy_testing", a, b, w)

    def _force(self):
        if config.sections["CALCULATOR"].force:
            num_forces = np.array(pt.shared_arrays['a'].num_atoms)*3
            if pt.shared_arrays['configs_per_group'].testing:
                testing = -1 * np.sum(num_forces[-pt.shared_arrays['configs_per_group'].testing:])
            else:
                testing = 0

            a, b, w = self._make_abw(pt.shared_arrays['a'].force_index, num_forces.tolist())
            config_indicies,energy_list,force_list,stress_list = self._config_error()
            if config.sections["SOLVER"].detailed_errors and self.weighted == "Unweighted":
                from csv import writer
                true, pred = b, a @ self.fit
                if pt.shared_arrays['configs_per_group'].testing:
                    ConfigType = ['Training'] * (
                                np.shape(true)[0] - np.sum(num_forces[-pt.shared_arrays['configs_per_group'].testing:])) + \
                                 ['Testing'] * (np.sum(num_forces[-pt.shared_arrays['configs_per_group'].testing:]))
                else:
                    ConfigType = ['Training'] * np.shape(true)[0]
                with open('detailed_force_errors.dat', 'w') as f:
                    writer = writer(f, delimiter=' ')
                    writer.writerow(['FileName Type True-Ref Predicted-Ref Difference(Pred-True)'])
                    writer.writerows(zip(force_list,ConfigType, true, pred, pred-true))

            self._errors([[0, testing]], ['*ALL'], "Force", a, b, w)
            if testing != 0:
                self._errors([[testing, 0]], ['*ALL'], "Force_testing", a, b, w)

    def _stress(self):
        if config.sections["CALCULATOR"].stress:
            testing = -6 * pt.shared_arrays['configs_per_group'].testing
            a, b, w = self._make_abw(pt.shared_arrays['a'].stress_index, 6)
            config_indicies,energy_list,force_list,stress_list = self._config_error()
            if config.sections["SOLVER"].detailed_errors and self.weighted == "Unweighted":
                from csv import writer
                true, pred = b, a @ self.fit
                ConfigType = ['Training'] * (np.shape(true)[0] - 6 * pt.shared_arrays['configs_per_group'].testing) + \
                                             ['Testing'] * (6 * pt.shared_arrays['configs_per_group'].testing)
                with open('detailed_stress_errors.dat', 'w') as f:
                    writer = writer(f, delimiter=' ')
                    writer.writerow(['FileName Type True-Ref Predicted-Ref Difference(Pred-True)'])
                    writer.writerows(zip(stress_list,ConfigType, true, pred, pred-true))

            self._errors([[0, testing]], ['*ALL'], "Stress", a, b, w)
            if testing != 0:
                self._errors([[testing, 0]], ['*ALL'], "Stress_testing", a, b, w)

    def _combined(self):
        self._errors([[0, pt.shared_arrays["configs_per_group"].testing_elements]], ["*ALL"], "Combined")
        if pt.shared_arrays["configs_per_group"].testing_elements != 0:
            self._errors([[pt.shared_arrays["configs_per_group"].testing_elements, 0]], ['*ALL'], "Combined_testing")

    @staticmethod
    def _make_abw(type_index, buffer):
        if isinstance(buffer, list):
            length = sum(buffer)
        else:
            length = len(type_index) * buffer
        width = np.shape(pt.shared_arrays['a'].array)[1]
        a = np.zeros((length, width))
        b = np.zeros((length,))
        w = np.zeros((length,))
        i = 0
        for j, value in enumerate(type_index):
            if isinstance(buffer, list):
                spacing = buffer[j]
            else:
                spacing = buffer
            a[i:i+spacing] = pt.shared_arrays['a'].array[value:value+spacing]
            b[i:i+spacing] = pt.shared_arrays['b'].array[value:value+spacing]
            w[i:i+spacing] = pt.shared_arrays['w'].array[value:value+spacing]
            i += spacing
        return a, b, w

    def _group_error(self):
        groups = []
        for group in pt.shared_arrays['configs_per_group'].list:
            group = group.split('/')[-1]
            groups.append(group)
        if config.sections["CALCULATOR"].energy:
            self._group_energy(groups)
        if config.sections["CALCULATOR"].force:
            self._group_force(groups)
        if config.sections["CALCULATOR"].stress:
            self._group_stress(groups)
#        self._group_combined(groups)

    def _group_energy(self, groups):
        group_index = pt.shared_arrays['a'].group_energy_index
        length = pt.shared_arrays['a'].group_energy_length
        index, a, b, w = self._make_group_abw(group_index, length, 1)
        self._errors(index, groups, "Energy", a, b, w)

    def _group_force(self, groups):
        group_index = pt.shared_arrays['a'].group_force_index
        length = pt.shared_arrays['a'].group_force_length
        index, a, b, w = self._make_group_abw(group_index, length, pt.shared_arrays['a'].num_atoms)
        self._errors(index, groups, "Force", a, b, w)

    def _group_stress(self, groups):
        group_index = pt.shared_arrays['a'].group_stress_index
        length = pt.shared_arrays['a'].group_stress_length*6
        index, a, b, w = self._make_group_abw(group_index, length, 6)
        self._errors(index, groups, "Stress", a, b, w)

    def _group_combined(self, groups):
        index = []
        group_index = pt.shared_arrays['a'].group_index
        for i in range(len(group_index) - 1):
            index.append([group_index[i], group_index[i+1]])
        #self._errors(index, groups, "Combined")

    @staticmethod
    def _make_group_abw(group_index, length, buffer):
        index = []
        width = np.shape(pt.shared_arrays['a'].array)[1]
        if isinstance(buffer, list):
            length = 3 * sum(buffer)
        a = np.zeros((length, width))
        b = np.zeros((length,))
        w = np.zeros((length,))
        i = 0
        j = 0
        for group in group_index:
            temp = [i]
            for value in group:
                if isinstance(buffer, list):
                    spacing = buffer[j]
                    j += 1
                    spacing *= 3
                else:
                    spacing = buffer
                a[i:i+spacing] = pt.shared_arrays['a'].array[value:value+spacing]
                b[i:i+spacing] = pt.shared_arrays['b'].array[value:value+spacing]
                w[i:i+spacing] = pt.shared_arrays['w'].array[value:value+spacing]
                i += spacing
            temp.append(i)
            index.append(temp)
        return index, a, b, w

    def _config_error(self):
        config_index = 0
        current_index = 0
        detailed_config_list = []
        energy_list = []
        force_list = []
        stress_list = []
        for i, num_atoms in enumerate(pt.shared_arrays['a'].num_atoms):
            this_config = pt.shared_arrays['number_of_atoms'].configs[i]
            if isinstance(this_config, str):
                this_config = this_config.split('/')[-2] + ':' + this_config.split('/')[-1]
                this_config = [this_config]
            elif isinstance(this_config, list):
                this_config = [this_config[-1].split('/')[-1] + ':' + str(this_config[0])]
            if config.sections["CALCULATOR"].energy:
                # current index is the index of the top of energy
                self._config_energy(this_config, current_index)
                current_index += 1
                energy_list += [this_config[0]]
            if config.sections["CALCULATOR"].force:
                # current index is the index of the top of force
                self._config_force(this_config, current_index, 3*num_atoms)
                current_index += 3*num_atoms
                force_list += 3*num_atoms*[this_config[0]]
            if config.sections["CALCULATOR"].stress:
                self._config_stress(this_config, current_index)
                current_index += 6
                stress_list += 6*[this_config[0]]
            self._config_combined(this_config, config_index, current_index)
            config_index = current_index
            if pt.shared_arrays['configs_per_group'].testing > 0 and \
                    i < (np.shape(pt.shared_arrays['a'].array)[0] - pt.shared_arrays['configs_per_group'].testing):
                ConfigType = 'Training'
            elif pt.shared_arrays['configs_per_group'].testing > 0 and \
                    i >= (np.shape(pt.shared_arrays['a'].array)[0] - pt.shared_arrays['configs_per_group'].testing):
                ConfigType = 'Testing'
            else:
                ConfigType = 'Training'

            detailed_config_list.append([i,ConfigType,num_atoms,this_config[0]])
        return detailed_config_list,energy_list,force_list,stress_list

    def _config_energy(self, this_config, current_index):
        index, a, b, w = self._make_config_abw(current_index, 1)
        #self._errors(index, this_config, "Energy", a, b, w)

    def _config_force(self, this_config, current_index, length):
        index, a, b, w = self._make_config_abw(current_index, length)
        #self._errors(index, this_config, "Force", a, b, w)

    def _config_stress(self, this_config, current_index):
        index, a, b, w = self._make_config_abw(current_index, 6)
        #self._errors(index, this_config, "Stress", a, b, w)

    def _config_combined(self, this_config, config_index, current_index):
        index, a, b, w = self._make_config_abw(config_index, current_index-config_index)
        #self._errors(index, this_config, "Combined", a, b, w)

    @staticmethod
    def _make_config_abw(i, buffer):
        index = None
        width = np.shape(pt.shared_arrays['a'].array)[1]
        a = np.zeros((buffer, width))
        b = np.zeros((buffer,))
        w = np.zeros((buffer,))
        a[:] = pt.shared_arrays['a'].array[i:i + buffer]
        b[:] = pt.shared_arrays['b'].array[i:i + buffer]
        w[:] = pt.shared_arrays['w'].array[i:i + buffer]

        return index, a, b, w

    def _errors(self, index, category, gtype, a=None, b=None, w=None):
        if a is None:
            a = pt.shared_arrays['a'].array
        if b is None:
            b = pt.shared_arrays['b'].array
        if w is None:
            w = pt.shared_arrays['w'].array
        for i, group in enumerate(category):
#            print(i,group,gtype)
            if index is None:
                a_err = a
                b_err = b
                w_err = w
            elif index[i][1] != 0:
                a_err = a[index[i][0]:index[i][1]]
                b_err = b[index[i][0]:index[i][1]]
                w_err = w[index[i][0]:index[i][1]]
            else:
                a_err = a[index[i][0]:]
                b_err = b[index[i][0]:]
                w_err = w[index[i][0]:]
            true, pred = b_err, a_err @ self.fit

            if self.fit_sam is not None:
                if self.cov is None:
                    pf_std = np.std(self.fit_sam @ a_err.T, axis=0)
                else:
                    # need to see which method is faster, both give the same results within numerical errors
                    # Method 1
                    chol = np.linalg.cholesky(self.cov)
                    mat = a_err @ chol
                    pf_std = np.linalg.norm(mat, axis=1)
                    # Method 2
                    # tmp = np.dot(a_err, self.cov)
                    # pf_std = np.empty(a_err.shape[0])
                    # for ipt in range(a_err.shape[0]):
                    #     pf_std[ipt] = np.sqrt(np.dot(tmp[ipt, :], a_err[ipt, :]))
                    # print(np.linalg.norm(pf_std-pf_std0), np.linalg.norm(pf_std0), np.linalg.norm(pf_std))
            else:
                pf_std = np.zeros(a_err.shape[0])

            if self.weighted == 'Weighted':
                true, pred, pf_std = w_err * true, w_err * pred, w_err * pf_std
                nconfig = np.count_nonzero(w_err)
            else:
                nconfig = len(pred)

            res = true - pred
            mae = np.sum(np.abs(res) / nconfig)
            # relative mae
            rres = ((true+1)-pred)/(true+1)
            rel_mae = np.sum(np.abs(rres) / nconfig)
            mean_dev = np.sum(np.abs(true - np.median(true)) / nconfig)
            ssr = np.square(res).sum()
            mse = ssr / nconfig
            rmse = np.sqrt(mse)
            rsq = 1 - ssr / np.sum(np.square(true - (true / nconfig).sum()))
            error_record = {
                "Group": group,
                "Weighting": self.weighted,
                "Subsystem": gtype,
                "ncount": nconfig,
                "mae": mae,
                "rmse": rmse,
                "rsq": rsq
            }
#                "rmae": mae / mean_dev,
#                "rrmse": rmse / np.std(true),
#                "ssr": ssr,

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

