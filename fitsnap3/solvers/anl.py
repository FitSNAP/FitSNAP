from ..io.outputs.outputs import optional_open
from ..io.outputs.snap import _to_coeff_string
from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
import numpy as np
from sys import float_info as fi


class ANL(Solver):

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

        npt, nbas = aw.shape

        cov_nugget = config.sections["SOLVER"].cov_nugget
        invptp = np.linalg.inv(np.dot(aw.T, aw)+cov_nugget*np.diag(np.ones((nbas,))))

        invptp = invptp*0.5 + invptp.T*0.5  #forcing symmetry; can get numerically significant errors when A is ill-conditioned

        np.savetxt('invptp', invptp)
        self.fit = np.dot(invptp, np.dot(aw.T, bw))


        #bp = np.dot(bw, bw - np.dot(aw, self.fit))/2. #numerically unstable when A is ill-conditioned

        res = bw - np.dot(aw, self.fit)
        bp = np.dot(res, res)/2.
        ap = (npt - nbas)/2.

        sigmahat = bp/(ap-1.)
        # print("Datanoise stdev : ", np.sqrt(sigmahat))

        # True posterior covariance
        self.cov = sigmahat*invptp
        # Variational covariance
        # self.cov = sigmahat/np.diag(np.diag(np.dot(aw.T, aw)))

        nsam = config.sections["SOLVER"].nsam
        self.fit_sam = np.random.multivariate_normal(self.fit, self.cov, size=(nsam,))
        # self.fit_sam = self.fit + np.sqrt(np.diag(self.cov))*np.random.randn(nsam,nbas)
        np.save('covariance.npy', self.cov)


    @staticmethod
    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

