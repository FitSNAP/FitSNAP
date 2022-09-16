from fitsnap3lib.io.outputs.outputs import optional_open
from fitsnap3lib.io.outputs.snap import _to_coeff_string
from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
import numpy as np
from sys import float_info as fi

pt = ParallelTools()
config = Config()

class ANL(Solver):

    def __init__(self, name):
        super().__init__(name)

    @pt.sub_rank_zero
    def perform_fit(self):
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

        ## If multiple elements with different 2J max settings, there will be columns of all 0.
        ## Need to remove those to make the matrix invertible. Backfill in the columns after calculations.
        zero_column_list = []
        for i in range(aw.shape[1]):
            if not np.any(aw[:,i]):
                zero_column_list.append(i)
        aw = np.delete(aw, zero_column_list, 1)

        npt, nbas = aw.shape

        cov_nugget = config.sections["SOLVER"].cov_nugget
        invptp = np.linalg.inv(np.dot(aw.T, aw)+cov_nugget*np.diag(np.ones((nbas,))))
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

        ## Backfilling 0s for any removed 0 columns from the A matrix
        original_a_num_columns = aw.shape[1]+len(zero_column_list)
        original_a_filled_column_indices = [k for k in range(original_a_num_columns) if k not in zero_column_list]
        #original_a = np.zeros((aw.shape[0], original_a_num_columns), dtype=a.dtype)  ## making the correct size matrix
        #original_a[:, original_a_filled_column_indices] = aw  ## I don't believe the code actually needs this, so currently leaving out
        sized_cov = np.zeros((original_a_num_columns, original_a_num_columns), dtype=aw.dtype)
        sized_cov[np.array(original_a_filled_column_indices).reshape(-1,1), original_a_filled_column_indices] = self.cov
        self.cov = sized_cov

        sized_fit = np.zeros((original_a_num_columns), dtype = aw.dtype)
        sized_fit[original_a_filled_column_indices] = self.fit
        self.fit = sized_fit
        np.save('covariance.npy', self.cov)
        np.save('mean.npy', self.fit)

        nsam = config.sections["SOLVER"].nsam
        if nsam:
            self.fit_sam = np.random.multivariate_normal(self.fit, self.cov, size=(nsam,))
        # self.fit_sam = self.fit + np.sqrt(np.diag(self.cov))*np.random.randn(nsam,nbas)


    @staticmethod
    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

