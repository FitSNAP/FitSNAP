from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
import numpy as np
try:
    import tensorflow as tf


    class TensorflowSVD(Solver):

        def __init__(self, name):
            super().__init__(name)

        @pt.single_timeit
        @pt.sub_rank_zero
        def perform_fit(self):
            training = [not elem for elem in pt.fitsnap_dict['Testing']]
            w = pt.shared_arrays['w'].array[training]
            aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
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
            if config.sections["CALCULATOR"].calculator == "LAMMPSSNAP" and config.sections["BISPECTRUM"].bzeroflag:
                self._offset()

except ModuleNotFoundError:

    class TensorflowSVD(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'tensorflow'")
