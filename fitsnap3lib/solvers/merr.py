from fitsnap3lib.io.outputs.outputs import optional_open
from fitsnap3lib.io.outputs.snap import _to_coeff_string
from fitsnap3lib.solvers.solver import Solver
import numpy as np
from sys import float_info as fi

from .lreg import lreg_merr

class MERR(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    def perform_fit(self):
        @self.pt.sub_rank_zero
        def decorated_perform_fit():
            pt = self.pt
            config = self.config
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

            ## If multiple elements with different 2J max settings, there will be columns of all 0.
            ## Want to remove those because there's no reason to embed upon them. Backfill with 0s afterwards.
            zero_column_list = []
            for i in range(aw.shape[1]):
                if not np.any(aw[:,i]):
                    zero_column_list.append(i)
            aw = np.delete(aw, zero_column_list, 1)
                
            npt, nbas = aw.shape

            cov_nugget = config.sections["SOLVER"].cov_nugget
            invptp = np.linalg.pinv(np.dot(aw.T, aw)+cov_nugget*np.diag(np.ones((nbas,))))
            invptp = invptp*0.5 + invptp.T*0.5  #forcing symmetry; can get numerically significant errors when A is ill-conditioned
            cf = np.dot(invptp, np.dot(aw.T, bw))
            
            #bp = np.dot(bw, bw - np.dot(aw, cf))/2. numerically unstable when A is ill-conditioned
            res = bw - np.dot(aw, cf)
            bp = np.dot(res, res)/2.
            ap = (npt - nbas)/2.
            sigmahat = bp/(ap-1.)
            # print("Datanoise stdev : ", np.sqrt(sigmahat))
            
            merr_mult = config.sections["SOLVER"].merr_mult
            merr_method = config.sections["SOLVER"].merr_method
            merr_cfs_str = config.sections["SOLVER"].merr_cfs
            if merr_cfs_str == 'all':
                ind_embed = None
                print("Embedding model error in all coefficients")
            else:
                ind_embed = []
                for i in list(merr_cfs_str.split(" ")): # Sanity check
                    assert(int(i)<=nbas)
                    ind_embed.append(int(i))
                print("Embedding model error in coefficients: ", ind_embed)

            lreg = lreg_merr(ind_embed=ind_embed, datavar=sigmahat,
                             multiplicative=bool(merr_mult), merr_method=merr_method,
                             method='bfgs')

            lreg.fit(aw, bw)
            self.fit = lreg.cf.copy()
            self.cov = lreg.cf_cov.copy()
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

        decorated_perform_fit()

    def _dump_a(self):
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
