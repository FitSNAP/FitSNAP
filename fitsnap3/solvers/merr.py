from ..io.outputs.outputs import optional_open
from ..io.outputs.original import _to_coeff_string
from .solver import Solver
from ..parallel_tools import pt
from ..io.input import config
import numpy as np

from .lreg import lreg_merr

class MERR(Solver):

    def __init__(self, name):
        super().__init__(name)

    @pt.sub_rank_zero
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
        if config.sections['EXTRAS'].apply_transpose:
            bw = aw.T@bw
            aw = aw.T@aw

        npt, nbas = aw.shape

        cov_nugget = config.sections["SOLVER"].cov_nugget
        invptp = np.linalg.inv(np.dot(aw.T, aw)+cov_nugget*np.diag(np.ones((nbas,))))
        cf = np.dot(invptp, np.dot(aw.T, bw))
        bp = np.dot(bw - np.dot(aw, cf), bw - np.dot(aw, cf))/2.
        ap = (npt - nbas)/2.
        sigmahat = bp/(ap-1.)
        # print("Datanoise stdev : ", np.sqrt(sigmahat))

        lreg = lreg_merr(ind_embed=None, datavar=sigmahat,
                 multiplicative=False, merr_method='abc',
                 method='bfgs')

        lreg.fit(aw, bw)
        self.fit = lreg.cf.copy()
        self.cov = lreg.cf_cov.copy()

        nsam = config.sections["SOLVER"].nsam
        self.fit_sam = np.random.multivariate_normal(self.fit, self.cov, size=(nsam,))
        # self.fit_sam = self.fit + np.sqrt(np.diag(self.cov))*np.random.randn(nsam,nbas)




    def _dump_a(self):
        np.savez_compressed('a.npz', a=pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

