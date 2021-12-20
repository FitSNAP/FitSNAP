from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
from sklearn.linear_model import ARDRegression
import numpy as np


class ARD(Solver):

    def __init__(self, name):
        super().__init__(name)

    @pt.sub_rank_zero
    def perform_fit(self):
        if pt.shared_arrays['configs_per_group'].testing_elements != 0:
            testing = -1*pt.shared_arrays['configs_per_group'].testing_elements
        else:
            testing = len(pt.shared_arrays['w'].array)
        w = pt.shared_arrays['w'].array[:testing]
        aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:testing], w * pt.shared_arrays['b'].array[:testing]
#        Transpose method does not work with Quadratic SNAP (why?)
#        We need to revisit this preconditioning of the linear problem, we can make this a bit more elegant. 
#        Since this breaks some examples this will stay as a 'secret' feature. 
#        Need to chat with some mathy people on how we can profile A and find good preconditioners. 
#        Will help when we want to try gradient based linear solvers as well. 
        if config.sections['EXTRAS'].apply_transpose:
            bw = aw.T@bw
            aw = aw.T@aw
        alval_small = 1e-14
        alval_big = 1e-12
        reg = ARDRegression(n_iter=300, tol=0.001, alpha_1=alval_big, alpha_2=alval_big, lambda_1=alval_small, lambda_2=alval_small,fit_intercept=True)
        reg.fit(aw,bw)
        #I'm placing the inference-estimated noise in the "residues"
        #self.fit, residues = reg.coef_, reg.lambda_
        self.fit = reg.coef_
        residues = reg.predict(aw) - bw
        #self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)

    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
