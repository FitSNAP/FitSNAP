from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import pt
from scipy.linalg import lstsq
import numpy as np


class SVD(Solver):

    def __init__(self, name):
        super().__init__(name)

    def perform_fit(self):
        if pt.shared_arrays['files_per_group'].testing != 0:
            testing = -1*pt.shared_arrays['files_per_group'].testing
        else:
            testing = len(pt.shared_arrays['w'].array)
        w = pt.shared_arrays['w'].array[:testing]
        aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:testing], w * pt.shared_arrays['b'].array[:testing]
        self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)
