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

    def perform_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
        """
        Perform fit on a linear system. If no args are supplied, will use fitting data in `pt.shared_arrays`.

        Args:
            a (np.array): Optional "A" matrix.
            b (np.array): Optional Truth array.
            w (np.array): Optional Weight array.
            fs_dict (dict): Optional dictionary containing a `Testing` key of which A matrix rows should not be trained.
            trainall (bool): Optional boolean declaring whether to train on all samples in the A matrix.

        The fit is stored as a member `fs.solver.fit`.
        """
        pt = self.pt
        # Only fit on rank 0 to prevent unnecessary memory and work.
        if pt._rank == 0:

            if fs_dict is not None:
                training = [not elem for elem in fs_dict['Testing']]
            elif trainall:
                training = [True]*np.shape(a)[0]
            else:
                training = [not elem for elem in pt.fitsnap_dict['Testing']]

            if a is None and b is None and w is None:
                w = pt.shared_arrays['w'].array[training]
                aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
            else:
                aw, bw = w[:, np.newaxis] * a[training], w * b[training]

            if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
                if np.linalg.cond(aw)**2 < 1 / fi.epsilon:
                    bw = aw[:, :].T @ bw
                    aw = aw[:, :].T @ aw
                else:
                    print("The Matrix is ill-conditioned for the transpose trick")
            self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)

    def _dump_a(self):
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
