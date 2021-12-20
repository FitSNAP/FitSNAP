from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
from sklearn.linear_model import Lasso
import numpy as np


class LASSO(Solver):

    def __init__(self, name):
        super().__init__(name)

    @pt.sub_rank_zero
    def perform_fit(self):
        if pt.shared_arrays['configs_per_group'].testing_elements != 0:
            testing = -1*pt.shared_arrays['configs_per_group'].testing_elements
        else:
            testing = len(pt.shared_arrays['w'].array)
        w = pt.shared_arrays['w'].array[:testing]
        #aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:testing], w * pt.shared_arrays['b'].array[:testing]
        a, b = pt.shared_arrays['a'].array[:testing],  pt.shared_arrays['b'].array[:testing]
#        Transpose method does not work with Quadratic SNAP (why?)
#        We need to revisit this preconditioning of the linear problem, we can make this a bit more elegant. 
#        Since this breaks some examples this will stay as a 'secret' feature. 
#        Need to chat with some mathy people on how we can profile A and find good preconditioners. 
#        Will help when we want to try gradient based linear solvers as well. 
        if config.sections['EXTRAS'].apply_transpose:
            b = a.T@b
            a = a.T@a
        alval_big = 1e-8
        reg = Lasso(alpha=alval_big,fit_intercept=True,max_iter=20000)
        # residues from fit?
        reg.fit(a,b,sample_weight=w)
        #I'm placing the inference-estimated noise in the "residues"
        self.fit = reg.coef_
        residues = reg.predict(a) - b
        #self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)

    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
