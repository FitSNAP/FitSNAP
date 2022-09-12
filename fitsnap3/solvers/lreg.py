#!/usr/bin/env python

import functools
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.stats import multivariate_normal



from .mcmc import MCMC



class lreg(object):
    def __init__(self):
        self.cf = None
        self.cf_cov = None
        self.fitted = False
        return

    def fit(self, Amat, y):
        raise NotImplementedError


    def print_coefs(self):
        assert(self.fitted)
        print(self.cf)

        return


    def predict(self, Amat, cov=False):
        assert(self.fitted)
        ypred = Amat @ self.cf
        if cov:
            ss = Amat @ self.cf_cov
            ypred_cov = ss @ Amat.T + self.datavar*np.eye(Amat.shape[0])#this is overkill, right now we only care about variance
            # clean this, i.e. make sure base class defines self.datavar
            # Also, is this correct for multiplicative noise case??
        else:
            ypred_cov = None #np.zeros((Amat.shape[1], Amat.shape[1])) #None

        return ypred, ypred_cov



# Bare minimum least squares solution
# (note: scipy's lstsq uses svd under the hood)
class lsq(lreg):
    def __init__(self):
        super(lsq, self).__init__()

        return


    def fit(self, Amat, y):

        self.cf, residues, rank, s = lstsq(Amat, y, 1.0e-13)
        self.cf_cov = np.zeros((Amat.shape[1], Amat.shape[1]))
        self.fitted = True

        return



def logpost_emb(x, aw=None, bw=None, ind_sig=None, datavar=0.0, multiplicative=False, merr_method='abc'):
    assert(aw is not None and bw is not None)
    npt, nbas = aw.shape

    cfs = x[:nbas]
    sig_cfs = x[nbas:]
    # if(np.min(sig_cfs)<=0.0):
    #     return -1.e+80

    if ind_sig is None:
        ind_sig = range(nbas)

    if multiplicative:
        sig_cfs = np.abs(cfs[ind_sig]) * sig_cfs

    #print(sig_cfs.shape[0], len(ind_sig))
    assert(sig_cfs.shape[0] == len(ind_sig))
    ss = aw[:, ind_sig] * sig_cfs

    # #### FULL COVARIANCE
    if merr_method == 'full':
        cov = ss @ ss.T + datavar * np.eye(npt) #self.datavar is a small nugget was crucial for MCMC sanity!
        #return sgn*(multivariate_normal.logpdf(aw @ cfs, mean=bw, cov=np.diag(cov), allow_singular=True)-np.sum(np.log(np.abs(sig_cfs))))
        val = multivariate_normal.logpdf(aw @ cfs, mean=bw, cov=np.diag(cov), allow_singular=False)

    # #### IID
    elif merr_method == 'iid':
        err = aw @ cfs - bw
        stds = np.linalg.norm(ss, axis=1)
        stds = np.sqrt(stds**2+datavar)
        val = -0.5 * np.sum((err/stds)**2)
        val -= 0.5 * npt * np.log(2.*np.pi)
        val -= np.sum(np.log(stds))

    #### ABC
    elif merr_method == 'abc':
        abceps=0.1
        abcalpha=1.0
        err = aw @ cfs - bw
        stds = np.linalg.norm(ss, axis=1)
        stds = np.sqrt(stds**2+datavar)
        err2 = abcalpha*np.abs(err)-stds
        val = -0.5 * np.sum((err/abceps)**2)
        val = -0.5 * np.sum((err2/abceps)**2)
        val -= 0.5 * np.log(2.*np.pi)
        val -= np.log(abceps)

    else:
        print(f"Merr type {merr_method} unknown. Exiting.")
        sys.exit()

    #print(val)

    # Prior?
    #val -= np.sum(np.log(np.abs(sig_cfs)))

    return val



class lreg_merr(lreg):
    def __init__(self, ind_embed=None, datavar=0.0, multiplicative=False, merr_method='abc', method='bfgs'):
        super(lreg_merr, self).__init__()

        self.ind_embed = ind_embed
        self.datavar = datavar
        self.multiplicative = multiplicative
        self.merr_method = merr_method
        self.method = method
        return

    def fit(self, A, y):
        npts, nbas = A.shape
        assert(y.shape[0] == npts)

        if self.ind_embed is None:
            self.ind_embed = range(nbas)

        nbas_emb = len(self.ind_embed)

        logpost_params = {'aw': A, 'bw':y, 'ind_sig':self.ind_embed, 'datavar':self.datavar, 'multiplicative':self.multiplicative, 'merr_method':self.merr_method}

        params_ini = np.random.rand(nbas+nbas_emb)
        #params_ini[:nbas], residues, rank, s = lstsq(A, y, 1.0e-13)
        invptp = np.linalg.inv(np.dot(A.T, A)+1.e-6*np.diag(np.ones((nbas,))))
        params_ini[:nbas] = np.dot(invptp, np.dot(A.T, y))

        if self.method == 'mcmc':

            # res = minimize((lambda x, fcn, p: -fcn(x, **p)), params_ini, args=(logpost_emb,logpost_params), method='BFGS', options={'gtol': 1e-16})
            # print(res)
            # params_ini = res.x

            covini = 0.1 * np.ones((params_ini.shape[0], params_ini.shape[0]))
            nmcmc = 10000
            gamma = 0.5
            t0 = 100
            tadapt = 100
            calib_params = {'param_ini': params_ini, 'cov_ini': covini,
                            't0': t0, 'tadapt' : tadapt,
                            'gamma' : gamma, 'nmcmc' : nmcmc}
            calib = AMCMC()
            calib.setParams(**calib_params)
            #samples, cmode, pmode, acc_rate, acc_rate_all, pmode_all = amcmc([nmcmc, params_ini, gamma, t0, tadapt, covini], logpost_emb, A, y, ind_sig=ind_embed, sgn=1.0)
            calib_results = calib.run(logpost_emb, **logpost_params)
            samples, cmode, pmode, acc_rate = calib_results['chain'],  calib_results['mapparams'],calib_results['maxpost'], calib_results['accrate']

            np.savetxt('chn.txt', samples)
            np.savetxt('mapparam.txt', cmode)
            coeffs = cmode[:nbas]
            coefs_sig = cmode[nbas:]

        elif self.method == 'bfgs':
            #params_ini[nbas:] = np.random.rand(nbas_emb,)
            res = minimize((lambda x, fcn, p: -fcn(x, **p)), params_ini, args=(logpost_emb, logpost_params), method='BFGS', options={'gtol': 1e-3})
            #print(res)
            coeffs = res.x[:nbas]
            coefs_sig = res.x[nbas:]

        self.cf = coeffs
        coefs_sig_all = np.zeros((nbas,))
        if self.multiplicative:
            coefs_sig_all[self.ind_embed] = np.abs(self.cf[self.ind_embed]) * coefs_sig
        else:
            coefs_sig_all[self.ind_embed] = coefs_sig
        self.cf_cov = np.diag(coefs_sig_all**2)
        self.fitted = True

        return
