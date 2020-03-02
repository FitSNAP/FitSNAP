from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import pt
from fitsnap3.io.input import config
from scipy.linalg import lstsq
import numpy as np


class SVD(Solver):

    def __init__(self, name):
        super().__init__(name)

    def perform_fit(self):
        w = pt.shared_arrays['w'].array
        aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array, w * pt.shared_arrays['b'].array
        self.fit, residues, rank, s = lstsq(aw, bw)
        if config.sections["MODEL"].bzeroflag:
            self._offset()
        return self.fit
