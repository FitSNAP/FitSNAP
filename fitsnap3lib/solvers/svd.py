from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
from scipy.linalg import lstsq
import numpy as np
from sys import float_info as fi


#config = Config()
#pt = ParallelTools()


class SVD(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        #self.config = Config()
        #self.pt = ParallelTools()

    #@pt.sub_rank_zero
    def perform_fit(self):
        @self.pt.sub_rank_zero
        def decorated_perform_fit():
            training = [not elem for elem in self.pt.fitsnap_dict['Testing']]
            w = self.pt.shared_arrays['w'].array[training]
            aw, bw = w[:, np.newaxis] * self.pt.shared_arrays['a'].array[training], w * self.pt.shared_arrays['b'].array[training]
    #       Look into gradient based linear solvers as well.
            if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
                if np.linalg.cond(aw)**2 < 1 / fi.epsilon:
                    bw = aw[:, :].T @ bw
                    aw = aw[:, :].T @ aw
                else:
                    print("The Matrix is ill-conditioned for the transpose trick")
            self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)
        decorated_perform_fit()

    def _dump_a(self):
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
