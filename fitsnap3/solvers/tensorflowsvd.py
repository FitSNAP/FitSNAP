from fitsnap3.solvers.solver import Solver
from fitsnap3.parallel_tools import pt
from fitsnap3.io.input import config
import numpy as np
try:
    import tensorflow as tf


    class TensorflowSVD(Solver):

        def __init__(self, name):
            super().__init__(name)

        @pt.single_timeit
        def perform_fit(self):
            w = pt.shared_arrays['w'].array
            aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array, w * pt.shared_arrays['b'].array
            bw = bw.reshape((len(bw), 1))
            self.fit = tf.linalg.lstsq(aw, bw)
            if config.sections["MODEL"].bzeroflag:
                self._offset()

except ModuleNotFoundError:

    class TensorflowSVD(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'tensorflow'")
