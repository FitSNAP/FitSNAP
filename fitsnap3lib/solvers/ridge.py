from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.lib.ridge_solver.regressor import Local_Ridge
import numpy as np


#config = Config()
#pt = ParallelTools()

class RIDGE(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    def perform_fit(self):
        @self.pt.sub_rank_zero
        def decorated_perform_fit():

            training = [not elem for elem in self.pt.fitsnap_dict['Testing']]
            w = self.pt.shared_arrays['w'].array[training]
            aw, bw = w[:, np.newaxis] * self.pt.shared_arrays['a'].array[training], w * self.pt.shared_arrays['b'].array[training]
            if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
                bw = aw.T @ bw
                aw = aw.T @ aw
            alval = self.config.sections['RIDGE'].alpha
            #print (self.config.sections['RIDGE'].local_solver, type(self.config.sections['RIDGE'].local_solver))
            if not self.config.sections['RIDGE'].local_solver:
                try:
                    from sklearn.linear_model import Ridge
                    reg = Ridge(alpha = alval, fit_intercept = False)
                except ModuleNotFoundError:
                    self.pt.single_print('Cannot find sklearn module, using local ridge solver anyway')
                    reg = Local_Ridge(alpha = alval, fit_intercept = False)
            elif self.config.sections['RIDGE'].local_solver:
                reg = Local_Ridge(alpha = alval, fit_intercept = False)

            reg.fit(aw, bw)
            self.pt.single_print('printing fit: ', reg.coef_)
            self.fit = reg.coef_
            residues = np.matmul(aw,reg.coef_) - bw
        decorated_perform_fit()


    #@staticmethod
    def _dump_a():
        np.savez_compressed('a.npz', a= self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

