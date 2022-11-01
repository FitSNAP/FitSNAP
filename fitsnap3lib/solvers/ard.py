from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
import numpy as np


config = Config()
pt = ParallelTools()


try:
    from sklearn.linear_model import ARDRegression


    class ARD(Solver):

        def __init__(self, name):
            super().__init__(name)
            self.pt = ParallelTools()
            self.config = Config()

        #@pt.sub_rank_zero
        def perform_fit(self):
            @self.pt.sub_rank_zero
            def decorated_perform_fit():
                training = [not elem for elem in pt.fitsnap_dict['Testing']]
                pt.single_print(training)
                w = pt.shared_arrays['w'].array[training]
                aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
                if config.sections['EXTRAS'].apply_transpose:
                    bw = aw.T@bw
                    aw = aw.T@aw
                # alval_small = config.sections['ARD'].alphasmall
                alval_big = config.sections['ARD'].alphabig
                lmbval_small = config.sections['ARD'].lambdasmall
                # lmbval_big = config.sections['ARD'].lambdabig
                thresh = config.sections['ARD'].threshold_lambda
                reg = ARDRegression(n_iter=300, tol=0.001, threshold_lambda=thresh, alpha_1=alval_big, alpha_2=alval_big,
                                    lambda_1=lmbval_small, lambda_2=lmbval_small, fit_intercept=False)
                reg.fit(aw, bw)
                self.fit = reg.coef_
                residues = reg.predict(aw) - bw
            decorated_perform_fit()
            for key in pt.fitsnap_dict.keys():
                pt.single_print(key,pt.fitsnap_dict[key])
            training = [not elem for elem in pt.fitsnap_dict['Testing']]
            w = pt.shared_arrays['w'].array[training]
            aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
            if config.sections['EXTRAS'].apply_transpose:
                bw = aw.T@bw
                aw = aw.T@aw
            alval_small = config.sections['ARD'].alphasmall
            alval_big = config.sections['ARD'].alphabig
            lmbval_small = config.sections['ARD'].lambdasmall
            lmbval_big = config.sections['ARD'].lambdabig
            thresh = config.sections['ARD'].threshold_lambda
            reg = ARDRegression(n_iter=300, tol=0.001, threshold_lambda=thresh, alpha_1=alval_big, alpha_2=alval_big,
                                lambda_1=lmbval_small, lambda_2=lmbval_small, fit_intercept=False)
            reg.fit(aw, bw)
            self.fit = reg.coef_
            residues = reg.predict(aw) - bw

        def _dump_a(self):
            np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class ARD(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'sklearn'")
