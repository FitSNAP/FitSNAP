from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import ParallelTools
from fitsnap3.io.input import Config
from scipy.linalg import lstsq
import numpy as np


config = Config()
pt = ParallelTools()


class SVD(Solver):

    def __init__(self, name):
        super().__init__(name)

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
        self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)

    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
