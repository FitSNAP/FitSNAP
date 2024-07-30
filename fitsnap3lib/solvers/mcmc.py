from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
import numpy as np
from scipy.linalg import lstsq
from sys import float_info as fi

#pt = ParallelTools()
#config = Config()


# Adaptive Markov chain Monte Carlo
def amcmc(inferpar, logpostFcn, aw, bw):
    nmcmc, cini, gamma, t0, tadapt, covini = inferpar  # inference parameters
    cdim = cini.shape[0]            # chain dimensionality
    print('chain dimensionality:', cdim)
    cov = np.zeros((cdim, cdim))   # covariance matrix
    samples = np.zeros((nmcmc, cdim))  # MCMC samples
    na = 0                        # counter for accepted steps
    sigcv = gamma * 2.4**2 / cdim
    samples[0] = cini                  # first step
    p1 = -logpostFcn(samples[0], aw, bw)  # NEGATIVE logposterior
    pmode = p1  # record MCMC 'mode', which is the current MAP value (maximum posterior)
    cmode = samples[0]  # MAP sample, new parameters
    acc_rate = 0.0  # Initial acceptance rate
    sample_change_steps = [0]
    unique_samples = samples[0].reshape((1,-1,))

    acc_rate_all=[]
    pmode_all=[]
    pmode_all.append(p1)
    # Loop over MCMC steps
    for k in range(nmcmc - 1):

        # Compute covariance matrix
        if k == 0:
            Xm = samples[0]
        else:
            Xm = (k * Xm + samples[k]) / (k + 1.0)
            rt = (k - 1.0) / k
            st = (k + 1.0) / k**2
            cov = rt * cov + st * np.dot(np.reshape(samples[k] - Xm, (cdim, 1)), np.reshape(samples[k] - Xm, (1, cdim)))
        if k == 0:
            propcov = covini
        else:
            if (k > t0) and (k % tadapt == 0):
                propcov = sigcv * (cov + 10**(-8) * np.identity(cdim))

        # Generate proposal candidate
        u = np.random.multivariate_normal(samples[k], propcov)
        p2 = -logpostFcn(u, aw, bw)
        #posterior ratio (target_PDF(proposed)/target_PDF(current))
        pr = np.exp(p1 - p2)
        # Accept...
        if np.random.random_sample() <= pr:
            samples[k + 1] = u
            na = na + 1  # Acceptance counter
            sample_change_steps.append(k+1)
            unique_samples = np.concatenate((unique_samples,u.reshape((1,-1,))))
            p1 = p2
            if p1 <= pmode:
                pmode = p1
                cmode = samples[k + 1]
                pmode_all.append(p1)
        # ... or reject
        else:
            samples[k + 1] = samples[k]

        acc_rate = float(na) / (k + 1)
        acc_rate_all.append(acc_rate)
        if((k + 2) % (nmcmc / 10) == 0) or k == nmcmc - 2:
            print('%d / %d completed, acceptance rate %lg' % (k + 2, nmcmc, acc_rate))
        steps_plus_end = np.concatenate((sample_change_steps, [nmcmc]))
        sample_weights = np.zeros(len(sample_change_steps))
        for i in range(len(steps_plus_end)-1):
            sample_weights[i] = steps_plus_end[i+1]-steps_plus_end[i]
    return samples, cmode, pmode, acc_rate, acc_rate_all, pmode_all, sample_weights, unique_samples

def log_norm_pdf(x, mu, sigma):
        s2 = sigma * sigma
        x_mu = x - mu
        norm_const = -0.5 * np.log(2 * np.pi * s2)
        return (norm_const - 0.5 * x_mu * x_mu / s2)

def logpost(x, aw, bw):
    lpostm = log_norm_pdf(aw@x,bw,sigma=0.1)
    return np.sum(lpostm)


class MCMC(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    def perform_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
        pt = self.pt
        # Only fit on rank 0 to prevent unnecessary memory and work.  
        if pt._rank == 0:
            
            if fs_dict is not None:
                training = [not elem for elem in fs_dict['Testing']]
            elif trainall:
                training = [True]*np.shape(a)[0]
            else:
                training = [not elem for elem in pt.fitsnap_dict['Testing']]

            if a is None and b is None and w is None:
                w = pt.shared_arrays['w'].array[training]
                aw, bw = w[:, np.newaxis] * pt.shared_arrays['a'].array[training], w * pt.shared_arrays['b'].array[training]
            else:
                aw, bw = w[:, np.newaxis] * a[training], w * b[training]

            if 'EXTRAS' in self.config.sections and self.config.sections['EXTRAS'].apply_transpose:
                if np.linalg.cond(aw)**2 < 1 / fi.epsilon:
                    bw = aw[:, :].T @ bw
                    aw = aw[:, :].T @ aw
                else:
                    print("The Matrix is ill-conditioned for the transpose trick")

            #MCMC parameters
            #param_ini = np.random.randn(aw.shape[1], )
            param_ini, residues, rank, s = lstsq(aw, bw, 1.0e-13)
            covini = np.zeros((aw.shape[1], aw.shape[1]))
            nmcmc = self.config.sections["SOLVER"].mcmc_num
            gamma = self.config.sections["SOLVER"].mcmc_gamma
            t0 = 100
            tadapt = 100
            samples, cmode, pmode, acc_rate, acc_rate_all, pmode_all, sample_weights, unique_samples = amcmc([nmcmc, param_ini, gamma, t0, tadapt, covini], logpost, aw, bw)
            self.fit = cmode
            nsam = self.config.sections["SOLVER"].nsam
            nevery = (nmcmc//2)//nsam
            self.fit_sam = samples[nmcmc//2:nmcmc:nevery, :][-nsam:, :]
            np.savetxt('chn.txt', samples)
            np.savetxt('chn_sam.txt', self.fit_sam)
            np.save('mean.npy', self.fit)
            np.save('unique_chn.npy', unique_samples)
            np.save('unique_chn_weights.npy', sample_weights)


    def _dump_a(self):
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
