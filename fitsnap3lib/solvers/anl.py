from fitsnap3lib.io.outputs.outputs import optional_open
from fitsnap3lib.io.outputs.snap import _to_coeff_string
from fitsnap3lib.solvers.solver import Solver
import numpy as np
from sys import float_info as fi

class ANL(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name ,pt, config)

    #@pt.sub_rank_zero
    def perform_fit(self):
        @self.pt.sub_rank_zero
        def decorated_perform_fit():
            pt = self.pt   
            config = self.config
            training = [not elem for elem in pt.fitsnap_dict['Testing']]
            w = pt.shared_arrays['w'].array[training]
            aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
    #       TODO: See if the transpose trick works or is nonsense when feeding into the UQ algos (probably nonsense)
            if config.sections['EXTRAS'].apply_transpose:
                if np.linalg.cond(aw)**2 < 1 / fi.epsilon:
                    bw = aw[:, :].T @ bw
                    aw = aw[:, :].T @ aw
                else:
                    print("The Matrix is ill-conditioned for the transpose trick")

            npt, nbas = aw.shape

            cov_nugget = config.sections["SOLVER"].cov_nugget
            invptp = np.linalg.pinv(np.dot(aw.T, aw)+cov_nugget*np.diag(np.ones((nbas,)))) #pinv() instead of inv() is robust against columns of 0s and better matches svd.py fits
            invptp = invptp*0.5 + invptp.T*0.5  #forcing symmetry; can get numerically significant errors when A is ill-conditioned
            #np.savetxt('invptp', invptp)
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

            np.save('covariance.npy', self.cov)
            np.save('mean.npy', self.fit)

            nsam = config.sections["SOLVER"].nsam
            if nsam:
                self.fit_sam = np.random.multivariate_normal(self.fit, self.cov, size=(nsam,))
            # self.fit_sam = self.fit + np.sqrt(np.diag(self.cov))*np.random.randn(nsam,nbas)

        decorated_perform_fit()


    @staticmethod
    def _dump_a(self):
        pt = self.pt
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        pt = self.pt
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
