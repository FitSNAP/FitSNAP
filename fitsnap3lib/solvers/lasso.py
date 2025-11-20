from fitsnap3lib.solvers.solver import Solver
import numpy as np

norm_a = False

try:
    from sklearn.linear_model import Lasso
    if norm_a:
        from sklearn.preprocessing import MinMaxScaler

    class LASSO(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)

        #@pt.sub_rank_zero
        def perform_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
            pt = self.pt
            if pt._rank == 0:

                if fs_dict is not None:
                    training = [not elem for elem in fs_dict['Testing']]
                elif trainall:
                    training = [True]*np.shape(a)[0]
                else:
                    training = [not elem for elem in pt.fitsnap_dict['Testing']]
                training = [not elem for elem in pt.fitsnap_dict['Testing']]
                w = pt.shared_arrays['w'].array[training]
                if norm_a:
                    scaler = MinMaxScaler()
                    if a is None and b is None and w is None:
                        w = pt.shared_arrays['w'].array[training]
                        anorm = scaler.fit_transform(pt.shared_arrays['a'].array[training])
                        aw, bw = w[:, np.newaxis] * anorm, w * pt.shared_arrays['b'].array[training]
                    else:
                        aw, bw = w[:, np.newaxis] * a[training], w * b[training]
                else:
                    if a is None and b is None and w is None:
                        w = pt.shared_arrays['w'].array[training]
                        aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
                    else:
                        pt.single_print('lasso types',type(w),type(a),type(b))
                        if a is None:
                            a = pt.shared_arrays['a'].array
                        if b is None:
                            b = pt.shared_arrays['b'].array
                        aw, bw = w[:, np.newaxis] * a[training], w * b[training]
                if self.config.sections['EXTRAS'].apply_transpose:
                    bw = aw.T @ bw
                    aw = aw.T @ aw
                alval = self.config.sections['LASSO'].alpha
                maxitr = self.config.sections['LASSO'].max_iter
                reg = Lasso(alpha=alval, fit_intercept=False, max_iter=maxitr)
                reg.fit(aw, bw)
                
                if norm_a:
                    norm_fit = reg.coef_
                    orig_fit = norm_fit/(scaler.data_min_ - scaler.data_max_)
                    self.fit = orig_fit
                    #self.fit = norm_fit
                else:
                    self.fit = reg.coef_
            #decorated_perform_fit()

        #@staticmethod
        def _dump_a():
            np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = self.pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class LASSO(Solver):

        def __init__(self, name):
            super().__init__(name)
            raise ModuleNotFoundError("No module named 'sklearn'")
