import pickle
import pathlib
import numpy as np
from scipy import special
from fitsnap3lib.parallel_tools import ParallelTools
from datetime import date


#pt = ParallelTools()


# multiplicative factor for expansion coefficients (for conversion of units from native testing to LAMMPS ML-PACE)
multfac = 1.


def intermediates(l1, l2):
    return range(abs(l1 - l2), l1 + l2 + 1)


def wigner_3j(j1, m1, j2, m2, j3, m3):
    # uses relation between Clebsch-Gordann coefficients and W-3j symbols to evaluate W-3j
    # VERIFIED - wolframalpha.com
    cg = clebsch_gordan(j1, m1, j2, m2, j3, -m3)

    num = (-1)**(j1-j2-m3)
    denom = ((2*j3) + 1)**(1/2)

    return cg*(num/denom)


class CouplingCoeffs:

    def __init__(self, nus=None, lmax=14, wflag=False):
        # coupling = {str(rank): {} for rank in ranks}
        self.pt = ParallelTools()
        self.coupling = {}
        self.lib_path = str(pathlib.Path(__file__).parent.resolve())
        self.lmax = lmax
        self.w3j = {}
        self.clebsch_gordan = {}
        self.find_w3j()

        if not wflag:
            self.pt.single_print('using default generalized Wigner 3j couplings')
            self.pt.single_print('generalized CG couplings have been removed')

        for nu in nus:
            nu.ccs(self.w3j)
            self.coupling[str(nu.rank)] = nu.nl_dict

        self.nus = nus
        self.lmax = lmax

        # write a coupling coefficient file that is compatible with py_ACE
        import json
        with open('{}/ccs.json'.format(self.lib_path), 'w') as writejson:
            json.dump(self.coupling, writejson, indent=2)

    def find_cg(self):
        try:
            with open('%s/Clebsch_Gordan.pickle' % self.lib_path, 'rb') as handle:
                self.pt.single_print("Opening Clebsch Gordan Pickle")
                self.clebsch_gordan = pickle.load(handle)
                self.pt.single_print("Finished Opening Clebsch Gordan Pickle")
        except FileNotFoundError:
            self.pt.single_print("Generating your first pickled library of CG coefficients. This will take a few moments...")
            if self.pt.get_rank() == 0:
                self.init_coupling(clebsch_gordan, self.clebsch_gordan)
                with open('%s/Clebsch_Gordan.pickle' % self.lib_path, 'wb') as handle:
                    pickle.dump(self.clebsch_gordan, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.pt.all_barrier()
            with open('%s/Clebsch_Gordan.pickle' % self.lib_path, 'rb') as handle:
                self.clebsch_gordan = pickle.load(handle)

    def find_w3j(self):
        try:
            with open('%s/Wigner_3j.pickle' % self.lib_path, 'rb') as handle:
                self.pt.single_print("Opening Wigner Pickle")
                self.w3j = pickle.load(handle)
                self.pt.single_print("Finished Opening Wigner Pickle")
        except FileNotFoundError:
            self.pt.single_print(
                "Generating your first pickled library of Wigner 3j coefficients. This will take a few moments...")
            if self.pt.get_rank() == 0:
                self.init_coupling(wigner_3j, self.w3j)
                with open('%s/Wigner_3j.pickle' % self.lib_path, 'wb') as handle:
                    pickle.dump(self.w3j, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.pt.all_barrier()
            with open('%s/Wigner_3j.pickle' % self.lib_path, 'rb') as handle:
                self.w3j = pickle.load(handle)

    def init_coupling(self, func, var):
        # returns dictionary of all cg coefficients to be used at a given value of lmax
        for l1 in range(self.lmax + 1):
            for l2 in range(self.lmax + 1):
                # for l3 in range(abs(l1-l2),l1+l2+1):
                for l3 in range(self.lmax + 1):
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            for m3 in range(-l3, l3 + 1):
                                key = '%d,%d,%d,%d,%d,%d' % (l1, m1, l2, m2, l3, m3)
                                var[key] = func(l1, m1, l2, m2, l3, m3)

    #@pt.rank_zero
    def write_pot(self, filname, element, nradbase, rcut, exp_lambda):
        @self.pt.rank_zero
        def decorated_write_pot():
            # TODO: CLEAN THIS FUNCTION UP (WAY TOO MUCH REDUNDANCY)
            tol = 1.e-5
            ranks = [nu.rank for nu in self.nus]
            nls = []
            for nu in self.nus:
                nls.extend(nu.nl)
            nradmax = max([nu.nmax for nu in self.nus])
            lmax = max([nu.lmax for nu in self.nus])
            # initialize an array of dummy ctilde coefficients (must be initialized to 1 for potential fitting)
            coeff_arr = np.ones(len(nls) + 1)
            coeffs = {nl: coeff for nl, coeff in zip(nls, coeff_arr[1:])}
            today = date.today()
            dt = today.strftime("%y-%m-%d")
            write_str = "# DATE: %s UNITS: metal CONTRIBUTOR: James Goff <jmgoff@sandia.gov> CITATION: py_PACE\n\n" % dt + \
                        "nelements=1\n" + \
                        "elements: %s\n\n" % element + \
                        "lmax=%d \n\n" % lmax + \
                        "2 FS parameters:  1.000000 1.000000\n" + \
                        "core energy-cutoff parameters: 100000.000000000000000000 250.000000000000000000\n" + \
                        "E0:%8.32f\n\n" % 0. + \
                        "radbasename=ChebExpCos\n" + \
                        "nradbase=%d\n" % nradbase + \
                        "nradmax=%d\n" % nradmax + \
                        "cutoffmax=%2.10f\n" % (rcut + tol) + \
                        "deltaSplineBins=0.001000\n" + \
                        "core repulsion parameters: 0.000000000000000000 1.000000000000000000\n" + \
                        "radparameter= %2.10f\n" % exp_lambda + \
                        "cutoff= %2.10f\n" % rcut + \
                        "dcut= 0.010000000000000000\n"

            # radial basis function expansion coefficients
            # saved in n,l,k shape
            # defaults to orthogonal delta function [g(n,k)] basis of drautz 2019
            crad = np.zeros((nradmax, lmax + 1, nradbase))
            for n in range(nradmax):
                for l in range(lmax + 1):
                    crad[n][l] = np.array([1. if k == n else 0. for k in range(nradbase)])

            cnew = np.zeros((nradbase, nradmax, lmax + 1))
            for n in range(1, nradmax + 1):
                for l in range(lmax + 1):
                    for k in range(1, nradbase + 1):
                        cnew[k - 1][n - 1][l] = crad[n - 1][l][k - 1]

            crd = """crad= """
            for k in range(nradbase):
                for row in cnew[k]:
                    tmp = ' '.join(str(b) for b in row)
                    tmp = tmp + '\n'
                    crd = crd + tmp
            crd = crd + '\n'

            ms, ccs, max_num_m = get_m_cc(self.coupling, ranks)
            maxrank = max(ranks)
            write_str2 = "rankmax=%d\n" % maxrank + \
                         "ndensitymax=1\n\n" + \
                         "num_c_tilde_max=%d\n" % len(nls) + \
                         "num_ms_combinations_max=%d\n" % max_num_m

            rank1 = "total_basis_size_rank1: %d\n" % len(ms[1].keys())

            # ----write rank 1s----
            for key in ms[1].keys():
                ctilde = "ctilde_basis_func: rank=1 ndens=1 mu0=0 mu=( 0 )\n" + \
                         "n=( %s )\n" % key.split(',')[0] + \
                         "l=( 0 )\n" + \
                         "num_ms=1\n" + \
                         "< 0 >:  %8.24f\n" % (coeffs[key] * multfac)
                rank1 = rank1 + ctilde
            rankplus = "total_basis_size: %d\n" % np.sum([len(ms[i].keys()) for i in range(2, maxrank + 1)])
            for rank in range(2, maxrank + 1):
                for key in ms[rank].keys():
                    try:
                        c = coeffs[key]
                    except KeyError:
                        print('Warning! no coefficient for %s' % key, 'using c_%s=0' % key)
                        c = 0
                    nstrlst = [' %d '] * rank
                    lstrlst = [' %d '] * rank
                    mustrlst = [' %d '] * rank
                    nstr = ''.join(n for n in nstrlst)
                    lstr = ''.join(l for l in lstrlst)
                    mustr = ''.join(l for l in mustrlst)
                    ns, ls = get_n_l(key, **{'rank': rank})
                    nstr = nstr % tuple(ns)
                    lstr = lstr % tuple(ls)
                    mustr = mustr % tuple([0] * rank)
                    num_ms = len(ms[rank][key])
                    ctilde = "ctilde_basis_func: rank=%d ndens=1 mu0=0 mu=( %s )\n" % (rank, mustr) + \
                             "n=(%s)\n" % nstr + \
                             "l=(%s)\n" % lstr + \
                             "num_ms=%d\n" % num_ms

                    for ind, m in enumerate(ms[rank][key]):
                        if type(m) == str:
                            m = [int(kz) for kz in m.split(',')]
                        mstr = ''.join(l for l in lstrlst)
                        mkeystr = ','.join(y for y in ['%d'] * rank)
                        mkeystr = mkeystr % tuple(m)
                        mstr = mstr % tuple(m)
                        m_add = '<%s>:  %8.24f\n' % (mstr, (c * multfac) * ccs[rank][key][ind])
                        ctilde = ctilde + m_add
                    ctilde = ctilde
                    rankplus = rankplus + ctilde

            with open('%s' % filname, 'w', encoding='utf8') as writeout:
                writeout.write(write_str)
                writeout.write(crd)
                writeout.write(write_str2)
                writeout.write(rank1)
                writeout.write(rankplus)

        decorated_write_pot()

                
def clebsch_gordan(j1, m1, j2, m2, j3, m3):
    # Clebsch-gordan coefficient calculator based on eqs. 4-5 of:
    # https://hal.inria.fr/hal-01851097/document
    # and christoph ortner's julia code ACE.jl

    # VERIFIED: test non-zero indices in Wolfram using format ClebschGordan[{j1,m1},{j2,m2},{j3,m3}]
    # rules:
    rule1 = np.abs(j1-j2) <= j3
    rule2 = j3 <= j1+j2
    rule3 = m3 == m1 + m2
    rule4 = np.abs(m3) <= j3

    # rules assumed by input
    # assert np.abs(m1) <= j1, 'm1 must be \in {-j1,j1}'
    # assert np.abs(m2) <= j2, 'm2 must be \in {-j2,j2}'

    if rule1 and rule2 and rule3 and rule4:
        # attempting binomial representation
        N1 = (2*j3) + 1 
        N2 = special.factorial(j1 + m1, exact=True) \
             * special.factorial(j1 - m1, exact=True) \
             * special.factorial(j2 + m2, exact=True) \
             * special.factorial(j2 - m2, exact=True) \
             * special.factorial(j3 + m3, exact=True) \
             * special.factorial(j3 - m3, exact=True)

        N3 = special.factorial(j1 + j2 - j3, exact=True) \
             * special.factorial(j1 - j2 + j3, exact=True) \
             * special.factorial(-j1 + j2 + j3, exact=True) \
             * special.factorial(j1 + j2 + j3 + 1, exact=True)

        N = (N1*N2)/N3

        G = 0.

        # k conditions (see eq.5 of https://hal.inria.fr/hal-01851097/document)
        # k  >= 0
        # k <= j1 - m1
        # k <= j2 + m2

        for k in range(0, min([j1-m1, j2+m2]) + 1):
            G1 = (-1)**k
            G2 = special.comb(j1 + j2 - j3, k, exact=True)
            G3 = special.comb(j1 - j2 + j3, j1 - m1 - k, exact=True)
            G4 = special.comb(-j1 + j2 + j3, j2 + m2 - k, exact=True)
            G += G1*G2*G3*G4
        return (N**(1/2))*G 

    else:
        return 0.


def get_m_cc(d, ranks):
    len_ms = []
    m_dict = {rank: None for rank in ranks}
    cc_dict = {rank: None for rank in ranks}
    for rank in ranks:
        rnk = str(rank)
        keys = d[rnk].keys()
        ms_dict = {key: None for key in keys}
        ccs_dict = {key: None for key in keys}

        for key in ms_dict.keys():
            ms_dict[key] = list(d[rnk][key].keys())
            len_ms.append(len(ms_dict[key]))
            ccs_dict[key] = list(d[rnk][key].values())
        m_dict[rank] = ms_dict
        cc_dict[rank] = ccs_dict
    max_num_ms = max(len_ms)
    return m_dict, cc_dict, max_num_ms


def get_nu_rank(nu):
    nu_splt = [int(k) for k in nu.split(',')]
    if len(nu_splt) == 3:
        if nu_splt[1] == 0 and nu_splt[2] == 0:
            return 1
        elif nu_splt[1] != 0 or nu_splt[2] != 0:
            return 2
    elif len(nu_splt) > 3:
        return int(len(nu_splt)/2)


def get_n_l(nu, **kwargs):
    try:
        rank = kwargs['rank']
    except KeyError:
        rank = get_nu_rank(nu)
    if rank == 1:
        nusplt = [int(k) for k in nu.split(',')]
        n = [nusplt[0]]
        l = [0]
    elif rank == 2:
        nusplt = [int(k) for k in nu.split(',')]
        n = [nusplt[0], nusplt[1]]
        # same l for each rank A in a rank 2 invariant       a
        l = [nusplt[-1], nusplt[-1]]
    elif rank > 2:
        nusplt = [int(k) for k in nu.split(',')]
        n = nusplt[:rank]
        l = nusplt[rank:]

    return n, l
