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
            if pt.shared_arrays['configs_per_group'].testing != 0:
                testing = -1 * pt.shared_arrays['configs_per_group'].testing
            else:
                testing = len(pt.shared_arrays['w'].array)
            w = pt.shared_arrays['w'].array[:testing]
            aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[:testing], w * pt.shared_arrays['b'].array[:testing]
            # NOTE: Transpose does not produce correct output
            # bw = aw.T @ bw.reshape
            # aw = aw.T @ aw
            # NOTE: Convert to tensor does not produce correct output
            # aw = tf.convert_to_tensor(aw, np.float32)
            # bw = tf.convert_to_tensor(bw, np.float32)
            # bw = tf.reshape(bw, [len(bw), 1])
            bw = bw.reshape((len(bw), 1))
            self.fit = tf.linalg.lstsq(aw, bw)
            self.fit = self.fit.numpy()
            if config.sections["CALCULATOR"].bzeroflag:
                self._offset()

except ModuleNotFoundError:

    class TensorflowSVD(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'tensorflow'")
