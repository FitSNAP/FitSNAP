from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import ParallelTools
from fitsnap3.io.input import Config
import numpy as np


config = Config()
pt = ParallelTools()


try:
    from sklearn.linear_model import Lasso


    class LASSO(Solver):

        def __init__(self, name):
            super().__init__(name)

        @pt.sub_rank_zero
        def perform_fit(self):
            if pt.shared_arrays['configs_per_group'].testing_elements != 0:
                testing = -1 * pt.shared_arrays['configs_per_group'].testing_elements
            else:
                testing = len(pt.shared_arrays['w'].array)
            w = pt.shared_arrays['w'].array[:testing]
            aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:testing], w * pt.shared_arrays['b'].array[:testing]
            if config.sections['EXTRAS'].apply_transpose:
                bw = aw.T @ bw
                aw = aw.T @ aw
            alval = config.sections['SOLVER'].alpha
            maxitr = config.sections['SOLVER'].max_iter
            reg = Lasso(alpha=alval, fit_intercept=False, max_iter=maxitr)
            reg.fit(aw, bw)
            self.fit = reg.coef_

        @staticmethod
        def _dump_a():
            np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class LASSO(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'sklearn'")
