from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
from scipy.linalg import lstsq
import numpy as np
from sys import float_info as fi


config = Config()
pt = ParallelTools()


class SVD(Solver):

    def __init__(self, name):
        super().__init__(name)

    @pt.sub_rank_zero
    def perform_fit(self):
        training = [not elem for elem in pt.fitsnap_dict['Testing']]
        w = pt.shared_arrays['w'].array[training]
        aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
#       Look into gradient based linear solvers as well.
        if config.sections['EXTRAS'].apply_transpose:
            if np.linalg.cond(aw)**2 < 1 / fi.epsilon:
                bw = aw[:, :].T @ bw
                aw = aw[:, :].T @ aw
            else:
                print("The Matrix is ill-conditioned for the transpose trick")
        self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)

    @staticmethod
    def _dump_a():
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
