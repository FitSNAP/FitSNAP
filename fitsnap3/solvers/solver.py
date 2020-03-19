from fitsnap3.io.input import config
from fitsnap3.parallel_tools import pt
import numpy as np
from pandas import DataFrame


class Solver:

    def __init__(self, name):
        self.name = name
        self.fit = None
        self.configs = None
        self.template_error = False
        self.errors = []
        self.weighted = 'Unweighted'
        self.residuals = None
        self.a = None
        self.b = None
        self.w = None

    def perform_fit(self):
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

    def error_analysis(self, data):
        for option in ["Unweighted", "Weighted"]:
            self.weighted = option
            self._all_error()
            self._group_error(data)

        if self.template_error is True:
            self._template_error()

        self.errors = DataFrame.from_records(self.errors)
        self.errors = self.errors.set_index(["Group", "Weighting", "Subsystem", ]).sort_index()

        if config.sections["CALCULATOR"].bzeroflag:
            self._offset()

    def _all_error(self):
        if config.sections["CALCULATOR"].energy:
            self._energy()
        if config.sections["CALCULATOR"].force:
            self._force()
        if config.sections["CALCULATOR"].stress:
            self._stress()
        self._combined()

    def _energy(self):
        testing = -1 * pt.shared_arrays['files_per_group'].testing
        a, b, w = self._make_abw(pt.shared_arrays['a'].energy_index, 1)
        self._errors([[0, testing]], ['*ALL'], "Energy", a, b, w)
        if testing != 0:
            self._errors([[testing, 0]], ['*ALL'], "Energy_testing", a, b, w)

    def _force(self):
        testing = -1 * pt.shared_arrays['files_per_group'].testing
        a, b, w = self._make_abw(pt.shared_arrays['a'].force_index, pt.shared_arrays['a'].num_atoms)
        self._errors([[0, testing]], ['*ALL'], "Force", a, b, w)
        if testing != 0:
            self._errors([[testing, 0]], ['*ALL'], "Force_testing", a, b, w)

    def _stress(self):
        testing = -1 * pt.shared_arrays['files_per_group'].testing
        a, b, w = self._make_abw(pt.shared_arrays['a'].stress_index, 6)
        self._errors([[0, testing]], ['*ALL'], "Stress", a, b, w)
        if testing != 0:
            self._errors([[testing, 0]], ['*ALL'], "Stress_testing", a, b, w)

    @staticmethod
    def _make_abw(type_index, buffer):
        if isinstance(buffer, list):
            length = sum(buffer)
            length *= 3
        else:
            length = len(type_index) * buffer
        width = np.shape(pt.shared_arrays['a'].array)[1]
        a = np.zeros((length, width))
        b = np.zeros((length,))
        w = np.zeros((length,))
        i = 0
        for j, value in enumerate(type_index):
            if isinstance(buffer, list):
                spacing = buffer[j]*3
            else:
                spacing = buffer
            a[i:i+spacing] = pt.shared_arrays['a'].array[value:value+spacing]
            b[i:i+spacing] = pt.shared_arrays['b'].array[value:value+spacing]
            w[i:i+spacing] = pt.shared_arrays['w'].array[value:value+spacing]
            i += spacing
        return a, b, w

    def _combined(self):
        testing = -1 * pt.shared_arrays['files_per_group'].testing
        self._errors([[0, testing]], ["*ALL"], "Combined")
        if testing != 0:
            self._errors([[testing, 0]], ['*ALL'], "Combined_testing")

    def _group_error(self, data):
        groups = []
        testing = len(data)-pt.shared_arrays['files_per_group'].testing
        for i, file in enumerate(data):
            if i < testing:
                groups.append(file["Group"])
            else:
                groups.append(file["Group"]+'_Testing')
        groups = sorted(set(groups))
        if config.sections["CALCULATOR"].energy:
            self._group_energy(groups)
        if config.sections["CALCULATOR"].force:
            self._group_force(groups)
        if config.sections["CALCULATOR"].stress:
            self._group_stress(groups)
        self._group_combined(groups)

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

    def _group_combined(self, groups):
        index = []
        group_index = pt.shared_arrays['a'].group_index
        for i in range(len(group_index) - 1):
            index.append([group_index[i], group_index[i+1]])
        self._errors(index, groups, "Combined")

    def _errors(self, index, category, gtype, a=None, b=None, w=None):
        if a is None:
            a = pt.shared_arrays['a'].array
        if b is None:
            b = pt.shared_arrays['b'].array
        if w is None:
            w = pt.shared_arrays['w'].array
        for i, group in enumerate(category):
            if index[i][1] != 0:
                a_err = a[index[i][0]:index[i][1]]
                b_err = b[index[i][0]:index[i][1]]
                w_err = w[index[i][0]:index[i][1]]
            else:
                a_err = a[index[i][0]:]
                b_err = b[index[i][0]:]
                w_err = w[index[i][0]:]
            true, pred = b_err, a_err @ self.fit
            if self.weighted == 'Weighted':
                true, pred = w_err * true, w_err * pred
                nconfig = np.count_nonzero(w_err)
            else:
                nconfig = len(pred)
            res = true - pred
            mae = np.sum(np.abs(res) / nconfig)
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
                "rmae": mae / mean_dev,
                "rmse": rmse,
                "rrmse": rmse / np.std(true),
                "ssr": ssr,
                "rsq": rsq
            }
            if self.residuals is not None:
                error_record["residual"] = res
            self.errors.append(error_record)

    def _template_error(self):
        pass
