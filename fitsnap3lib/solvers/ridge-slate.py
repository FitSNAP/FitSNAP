from fitsnap3lib.solvers.solver import Solver
import numpy as np

class RidgeSlate(Solver):

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
