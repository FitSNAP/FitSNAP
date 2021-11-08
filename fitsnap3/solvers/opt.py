from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
from scipy.optimize import minimize
import numpy as np


def distance(x, aw, bw):
    return np.linalg.norm(np.dot(aw, x) - bw)


def distance_grad(x, aw, bw):
    # get analytical gradient:
    return np.dot(aw.T, np.dot(aw, x) - bw)


class OPT(Solver):

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

        param_ini = np.random.randn(aw.shape[1], )
        res = minimize(distance, param_ini, args=(aw, bw), method='BFGS', options={'gtol': 1e-13}, jac=distance_grad)
        self.fit = res.x

    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
