from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import pt
from fitsnap3.io.input import config
from scipy.linalg import lstsq
import numpy as np


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
        if config.sections['SOLVER'].apply_transpose:
            bw = aw.T@bw
            aw = aw.T@aw
        self.fit, residues, rank, s = lstsq(aw, bw, 1.0e-13)
