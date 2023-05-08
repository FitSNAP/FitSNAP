from fitsnap3lib.solvers.solver import Solver
import numpy as np


try:
    from sklearn.linear_model import ARDRegression


    class ARD(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)

        #@pt.sub_rank_zero
        def perform_fit(self):
            @self.pt.sub_rank_zero
            def decorated_perform_fit():
                training = [not elem for elem in self.pt.fitsnap_dict['Testing']]
                w = self.pt.shared_arrays['w'].array[training]
                aw, bw = w[:, np.newaxis] * self.pt.shared_arrays['a'].array[training], w * self.pt.shared_arrays['b'].array[training]

                if  self.config.sections['EXTRAS'].apply_transpose:
                    bw = aw.T@bw
                    aw = aw.T@aw

                ap = (1/(np.var(bw)))
                self.pt.single_print('inverse variance in training data: %f, logscale for threshold_lambda: %f' %(ap, (np.log10(ap))))

                alval_small = self.config.sections['ARD'].alphasmall
                alval_big = self.config.sections['ARD'].alphabig
                lmbval_small = self.config.sections['ARD'].lambdasmall
                lmbval_big = self.config.sections['ARD'].lambdabig
                thresh = self.config.sections['ARD'].threshold_lambda
                directmethod = self.config.sections['ARD'].directmethod
                scap = self.config.sections['ARD'].scap
                scai = self.config.sections['ARD'].scai
                logcut = self.config.sections['ARD'].logcut
                self.pt.single_print('automated threshold_lambda will be 10**(%f + %1.3f)' % (logcut , np.abs(np.log10(ap)) ) )
                if directmethod:
                    reg = ARDRegression(n_iter=1000, threshold_lambda=thresh, alpha_1=alval_big, alpha_2=alval_big,
                                        lambda_1=lmbval_small, lambda_2=lmbval_small, fit_intercept=False)
                elif not directmethod:
                    reg = ARDRegression(n_iter=1000,alpha_1=scap*ap, alpha_2=scap*ap, lambda_1=ap*scai,lambda_2=ap*scai,fit_intercept=False,threshold_lambda= 10**(int(np.abs(np.log10(ap)))+logcut) )
                else:
                    reg = ARDRegression(n_iter=1000,alpha_1=scap*ap, alpha_2=scap*ap, lambda_1=ap*scai,lambda_2=ap*scai,fit_intercept=False,threshold_lambda= 10**(int(np.abs(np.log10(ap)))+logcut) )

                reg.fit(aw, bw)
                self.fit = reg.coef_
            decorated_perform_fit()

    def _dump_a(self):
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class ARD(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'sklearn'")
