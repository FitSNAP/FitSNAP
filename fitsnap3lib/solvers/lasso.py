from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
import numpy as np


#config = Config()
#pt = ParallelTools()


try:
    from sklearn.linear_model import Lasso


    class LASSO(Solver):

        def __init__(self, name):
            super().__init__(name)
            self.pt = ParallelTools()
            self.config = Config()

        def perform_fit(self):
            @self.pt.sub_rank_zero
            def decorated_perform_fit():
                training = [not elem for elem in self.pt.fitsnap_dict['Testing']]
                w = self.pt.shared_arrays['w'].array[training]
                aw, bw = w[:, np.newaxis] * self.pt.shared_arrays['a'].array[training], w * self.pt.shared_arrays['b'].array[training]
                if self.config.sections['EXTRAS'].apply_transpose:
                    bw = aw.T @ bw
                    aw = aw.T @ aw
                alval = self.config.sections['LASSO'].alpha
                maxitr = self.config.sections['LASSO'].max_iter
                reg = Lasso(alpha=alval, fit_intercept=False, max_iter=maxitr)
                reg.fit(aw, bw)
                self.fit = reg.coef_
            decorated_perform_fit()

        #@staticmethod
        def _dump_a():
            np.savez_compressed('a.npz', a= self.pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = self.pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class LASSO(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'sklearn'")
