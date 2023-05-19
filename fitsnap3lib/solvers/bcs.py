from fitsnap3lib.solvers.solver import Solver
from scipy.linalg import lstsq
import numpy as np


def bcs(A, y, sigma2=None, eta=1.e-8, adaptive=0, optimal=1, scale=0.1):
    #------------------------------------------------------------------
    # The BCS algorithm for the following paper:
    # "Bayesian Compressive Sesning" (Preprint, 2007). The algorithm
    # adopts from the fast RVM algorithm [Tipping & Faul, 2003].
    # Coded by: Shihao Ji, ECE, Duke University
    # last change: Jan. 2, 2007
    # You are suggested to use mt_CS.m for improved robustness
    #------------------------------------------------------------------
    # Input for BCS:
    #   A: projection matrix
    #   t:   CS measurements
    #   sigma2: initial noise variance
    #      If measurement noise exists and/or w is not truely sparse,
    #             then sigma2 = std(t)^2/1e2 (suggested)
    #      If no measurement noise and w is truely sparse,
    #             then sigma2 = std(t)^2/1e6 (suggested)
    #      This term is in fact not updated in the implementation to allow
    #      the fast algorithm. For this reason, you are recommended to use
    #      mt_CS.m, in which the noise variance is marginalized.
    #   eta: threshold for stopping the algorithm (suggested value: 1e-8)
    # Input for Adaptive CS:
    #   adaptive: generate basis for adpative CS? (default: 0)
    #   optimal: use the rigorous implementation of adaptive CS? (default: 1)
    #   scale: diagonal loading parameter (default: 0.1)
    # Output:
    #   weights:  sparse weights
    #   used:     the positions of sparse weights
    #   sigma2:   re-estimated noise variance
    #   errbars:  one standard deviation around the sparse weights
    #   basis:    if adaptive==1, then basis = the next projection vector
    #

    nugget = 1.e-16

    if sigma2 is None:
        sigma2 = max(np.std(y)**2/100., nugget)

    # find initial alpha
    N, M = A.shape
    Aty = np.dot(A.T, y)  # (M,)
    A2 = np.sum(A**2, axis=0) # (M,)
    ratio = (Aty**2 + nugget * np.ones_like(Aty)) / (A2 + nugget * np.ones_like(A2)) # (M,)


    index = [np.argmax(ratio)] # vector of dynamic size K with values = 0..M-1
    maxr = ratio[index] # (K,)

    alpha = A2[index] / (maxr - sigma2) # (K,)

    # compute initial mu, Sig, S, Q
    Asl = A[:, index] # (N,K)
    Hessian = alpha + np.dot(Asl.T, Asl) / sigma2 # (K,K)
    Sig = 1. / (Hessian + nugget * np.ones_like(Hessian)) # (K,K)
    mu = np.zeros((1,))
    mu[0] = Sig[0,0] * Aty[index[0]] / sigma2 # (K,)

    left = np.dot(A.T, Asl) / sigma2 # (M,K)
    S = A2 / sigma2 - Sig[0, 0] * left[:, 0]**2 # (M,)
    Q = Aty / sigma2 - Sig[0, 0] * Aty[index[0]] / sigma2 * left[:, 0] # (M,)


    itermax = 10
    mlhist = np.empty(itermax)

    for count in range(itermax):  # careful with index below
        ss = S.copy()
        qq = Q.copy()
        ss[index] = alpha * S[index] / (alpha - S[index])
        qq[index] = alpha * Q[index] / (alpha - S[index])
        theta = qq**2 - ss # (M,)

        # choose the next alpha that maximizes marginal likelihood
        ml = -np.inf * np.ones(M)
        ig0 = np.where(theta > 0)[0] # vector of values 0..M-1 of size L<=M

        # index for re-estimate ire=ig0[foo]=index[which]
        ire, foo, which = np.intersect1d(ig0, index, return_indices=True)
        if len(ire) > 0:
            alpha_ = ss[ire]**2 / theta[ire] + nugget

            delta = (alpha[which] - alpha_) / (alpha_ * alpha[which])
            ml[ire] = Q[ire]**2 * delta / (S[ire] * delta + 1) - np.log(1 + S[ire] * delta)

        # index for adding
        iad = np.setdiff1d(ig0, ire)
        if len(iad) > 0:
            ml[iad] = (Q[iad]**2 - S[iad]) / S[iad] + np.log(S[iad] / (Q[iad]**2))

        is0 = np.setdiff1d(np.arange(M), ig0)

        # index for deleting
        ide, foo, which = np.intersect1d(is0, index, return_indices=True)
        if len(ide) > 0:
            ml[ide] = Q[ide]**2 / (S[ide] - alpha[which]) - np.log(1. - S[ide] / alpha[which])


        idx = np.argmax(ml)  #TODO check single value?
        mlhist[count] = ml[idx]

        # check if terminates?
        if count > 1 and \
           abs(mlhist[count] - mlhist[count - 1]) < abs(mlhist[count] - mlhist[0]) * eta:
            break

        # update alphas
        which = np.where(index == idx)[0] # TODO assert length 1?
        if theta[idx] > 0:
            if len(which) > 0:            # re-estimate
                alpha_ = ss[idx]**2 / theta[idx] + nugget
                Sigii = Sig[which[0], which[0]]
                mui = mu[which[0]]
                Sigi = Sig[:, which[0]]  # (K,)

                delta = alpha_ - alpha[which[0]]
                ki = delta / (1. + Sigii * delta)
                mu = mu - ki * mui * Sigi  # (K,)
                Sig = Sig - ki * np.dot(Sigi.reshape(-1, 1), Sigi.reshape(1, -1))
                comm = np.dot(A.T, np.dot(Asl, Sigi) / sigma2)  # (M,)

                S = S + ki * comm**2 # (M,)
                Q = Q + ki * mui * comm # (M,)
                #
                alpha[which] = alpha_
            else:            # adding
                alpha_ = ss[idx]**2 / theta[idx] + nugget
                Ai = A[:, idx]  # (N,)
                Sigii = 1. / (alpha_ + S[idx])
                mui = Sigii * Q[idx]

                comm1 = np.dot(Sig, np.dot(Asl.T, Ai)) / sigma2 # (K,)

                ei = Ai - np.dot(Asl, comm1) # (N,)
                off = -Sigii * comm1 #( K,)
                #print(Sig.shape, Sigii.shape, off.shape)
                Sig = np.block([[
                               Sig + Sigii * np.dot(comm1.reshape(-1, 1),
                                                    comm1.reshape(1, -1)),
                               off.reshape(-1, 1)],
                               [off.reshape(1, -1),
                               Sigii]])

                mu = np.append(mu - mui * comm1, mui)
                comm2 = np.dot(A.T, ei) / sigma2 #(M,)
                S = S - Sigii * comm2**2
                Q = Q - mui * comm2
                #
                index = np.append(index, idx)
                alpha = np.append(alpha, alpha_)
                Asl = np.hstack((Asl, Ai.reshape(-1, 1))) # (N, K++)

        else:
            if len(which) > 0 and len(index) > 1:            # deleting
                Sigii = Sig[which[0], which[0]]
                mui = mu[which[0]]
                Sigi = Sig[:, which[0]] # (K,)

                Sig -= np.dot(Sigi.reshape(-1, 1), Sigi.reshape(1, -1)) / Sigii
                Sig = np.delete(Sig, which[0], 0)
                Sig = np.delete(Sig, which[0], 1)

                mu = mu - (mui / Sigii) * Sigi # (K,)
                mu = np.delete(mu, which[0])
                comm = np.dot(A.T, np.dot(Asl, Sigi)) / sigma2 # (M,)
                S = S + (comm**2 / Sigii) # (M,)
                Q = Q + (mui / Sigii) * comm # (M,)
                #
                index = np.delete(index, which[0])
                alpha = np.delete(alpha, which[0])
                Asl = np.delete(Asl, which[0], 1)
            if len(which) > 0 and len(index) == 1:
                break
    weights = mu
    used = index
    # re-estimated sigma2
    sigma2 = np.sum((y - np.dot(Asl, mu))**2) / (N - len(index) +
                                                 np.dot(alpha.reshape(1, -1),
                                                        np.diag(Sig)))

    assert(np.linalg.det(Sig)>-nugget)
    Sig[Sig<0]=0.0
    errbars = np.sqrt(np.diag(Sig))

    # generate a basis for adaptive CS?
    basis = None
    if adaptive:
        if optimal:
            D, V = np.linalg.eig(Sig)
            idx = np.argmax(D) # TODO is it a single number?
            basis = V[:, idx]
        else:
            temp = np.dot(Asl.T, Asl) / sigma2
            Sig_inv = temp + scale * np.mean(np.diag(temp)) * np.eye(len(used))
            D, V = np.linalg.eig(Sig_inv)
            idx = np.argmin(D)
            basis = V[:, idx]

    return weights, errbars, used, sigma2, basis, Sig # the last Sig was recently added

class BCS(Solver):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)

    @pt.sub_rank_zero
    def perform_fit(self):
        pt = self.pt
        config = self.config
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

        fit_, errbars, used, sigma2, basis, Sig = bcs(aw, bw, eta=1.e-18)

        # KS: this is temporary until we figure out how to handle sparse regression a-la Lasso
        self.fit = np.zeros((aw.shape[1],))
        self.fit[used] = fit_
        self.cov = np.diag(self.fit**2)
        print("AAA ", used, aw.shape, used.shape, fit_.shape, Sig.shape)
        self.cov[np.ix_(used, used)] = Sig
        nsam = config.sections["SOLVER"].nsam
        self.fit_sam = np.random.multivariate_normal(self.fit, self.cov, size=(nsam,))
        # self.fit_sam = self.fit + np.sqrt(np.diag(self.cov))*np.random.randn(nsam,nbas)

    def _dump_a(self):
        np.savez_compressed('a.npz', a=self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)
