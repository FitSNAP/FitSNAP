from fitsnap3lib.solvers.solver import Solver
import numpy as np


try:
    from sklearn.linear_model import Lasso,PassiveAggressiveRegressor,ARDRegression,OrthogonalMatchingPursuit,OrthogonalMatchingPursuit,OrthogonalMatchingPursuit,OrthogonalMatchingPursuit
    from fitsnap3lib.lib.sym_ACE.pa_gen import *
    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import  *
    from fitsnap3lib.lib.sym_ACE.wigner_couple import *
    from fitsnap3lib.lib.sym_ACE.clebsch_couple import *

    def sub_sort(nus_unsort):
        nus = nus_unsort.copy()
        mu0s = []
        mus =[]
        ns = []
        ls = []
        for nu in nus_unsort:
            mu0ii,muii,nii,lii = get_mu_n_l(nu)
            mu0s.append(mu0ii)
            mus.append(tuple(muii))
            ns.append(tuple(nii))
            ls.append(tuple(lii))
        nus.sort(key = lambda x : mus[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : ns[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : ls[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
        nus.sort(key = lambda x : len(x),reverse = False)
        nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
        byattyp = srt_by_attyp(nus)
        return byattyp

    class SEQ_ACE_LASSO(Solver):

        def __init__(self, name, pt, config):
            super().__init__(name, pt, config)

        def return_reg(self,alval,enforce_pos,irank=0):
            reg = Lasso(alpha = alval,max_iter=self.config.sections['SEQ_ACE_LASSO'].max_iter, positive=bool(enforce_pos), fit_intercept = False, tol=1.e-1)
            #reg = ARDRegression(max_iter=self.config.sections['SEQ_ACE_LASSO'].max_iter, fit_intercept = False)
            #reg = OrthogonalMatchingPursuit( n_nonzero_coefs=int(alval*self.rank_lens[irank]),fit_intercept = False)
            #reg = PassiveAggressiveRegressor(C= alval,max_iter=self.config.sections['SEQ_ACE_LASSO'].max_iter, fit_intercept = False, tol=1.e-1)
            return reg

        def perform_fit(self, a=None, b=None, w=None, fs_dict=None, trainall=False):
            """
            Perform fit on a linear system. If no args are supplied, will use fitting data in `pt.shared_arrays`.

            Args:
                a (np.array): Optional "A" matrix.
                b (np.array): Optional Truth array.
                w (np.array): Optional Weight array.
                fs_dict (dict): Optional dictionary containing a `Testing` key of which A matrix rows should not be trained.
                trainall (bool): Optional boolean declaring whether to train on all samples in the A matrix.

            The fit is stored as a member `fs.solver.fit`.
            """
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
                    bw = aw.T @ bw
                    aw = aw.T @ aw
                assert 'ACE' in list(self.config.sections.keys()), "sequential lasso fits are only implemented with ACE"
                these_ranked = self.config.sections['ACE'].ranked_chem_nus
                these_lens = [len(k) for k in these_ranked]
                self.rank_lens = these_lens
                if self.config.sections['ACE'].bzeroflag:
                    shape_cond = np.sum(these_lens) == aw.shape[1]
                elif not self.config.sections['ACE'].bzeroflag:
                    shape_cond = np.sum(these_lens) + len(self.config.sections['ACE'].types) == aw.shape[1]
                assert shape_cond, "must have number of descriptors equal number of columns in A matrix"
                alvals = self.config.sections['SEQ_ACE_LASSO'].alpha
                if len(self.config.sections['SEQ_ACE_LASSO'].enforce_pos_rank) > 1:
                    enforce_pos_rank = [ int(ki) for ki in self.config.sections['SEQ_ACE_LASSO'].enforce_pos_rank]
                else:
                    enforce_pos_rank = [ int(ki) for ki in self.config.sections['SEQ_ACE_LASSO'].enforce_pos_rank] * len(these_lens)
                assert len(alvals) == len(self.config.sections['ACE'].ranks) or len(alvals)==1, "either supply an alpha for each ACE descriptor rank or supply one alpha to be applied to all ranks"
                these_chem_lens = [int(len(k)/len(self.config.sections['ACE'].types)) for k in these_ranked]
                per_chem_ranked = [sub_sort(subnus) for subnus in these_ranked]
                #aw_cut1 = aw[:,: these_chem_lens[0]]
                if self.config.sections['ACE'].bzeroflag:
                    fit = np.zeros(np.sum(these_lens))
                elif not self.config.sections['ACE'].bzeroflag:
                    fit = np.zeros(np.sum(these_lens) + len(self.config.sections['ACE'].types))
                chem_offset_ranked = np.zeros((len(self.config.sections['ACE'].ranks),len(self.config.sections['ACE'].types)),dtype=np.int64)
                for irank,rank in enumerate(self.config.sections['ACE'].ranks):
                    rank = int(rank)
                    alval = float(alvals[irank])
                    if irank == 0:
                        rankoffset = 0
                    else:
                        rankoffset = int(np.sum(these_chem_lens[:irank]))
                    stacked_chems_per_rank = []
                    offsets = []
                    testoffsets = []
                    iranklst = list(range(irank))
                    iranklst.reverse()
                    lower_rank_inds = [i for i in iranklst if i != irank]
                    for ichem in range(len(self.config.sections['ACE'].types)):
                        ichemlst = list(range(ichem))
                        lower_chem_inds = [i for i in ichemlst if i != ichem]
                        chemoffset = ichem * these_chem_lens[irank]
                        #testoffset = ichem * int(np.sum(these_chem_lens))
                        testoffset = (ichem * int(np.sum(these_chem_lens))) + rankoffset
                        if not self.config.sections['ACE'].bzeroflag and irank == 0:
                            testoffset+=1 #account for 0th order "descriptor" column
                        #for ilchem in lower_chem_inds:
                        #    chemoffset+= 
                        #if rank ==1:
                        #    chemoffset = ichem * int(np.sum(these_chem_lens))
                        if rank > 1:
                            for ilrank in lower_rank_inds:
                                for ilchem in lower_chem_inds:
                                    chemoffset += chem_offset_ranked[ilrank][ilchem]
                                #chemoffset += chem_offset_ranked[ilrank][ichem]
                        #NOTE
                        #aw_cut1 = aw[:, chemoffset:chemoffset+these_chem_lens[irank]]
                        aw_cut1 = aw[:, testoffset:testoffset+these_chem_lens[irank]]
                        stacked_chems_per_rank.append(aw_cut1)
                        offsets.append((chemoffset,chemoffset+these_chem_lens[irank]))
                        testoffsets.append((testoffset,testoffset+these_chem_lens[irank]))
                        chem_offset_ranked[irank][ichem] = chemoffset
                    aw_per_rank = np.hstack(stacked_chems_per_rank)
                    reg_per_rank = self.return_reg(alval,enforce_pos_rank[irank])
                    #reg_per_rank = self.return_reg(alval,enforce_pos_rank[irank],irank)
                    reg_per_rank.fit(aw_per_rank, bw - aw @ fit)
                    coeff_per_rank = reg_per_rank.coef_
                    for ichem in range(len(self.config.sections['ACE'].types)):
                        suboff = int(chem_offset_ranked[irank][ichem])
                        chemoffset = ichem * these_chem_lens[irank]
                        chem_coeff = coeff_per_rank[chemoffset:chemoffset+these_chem_lens[irank]]
                        these_offsets = offsets[ichem]
                        tst_off = testoffsets[ichem]
                        ##update global fit
                        #fit[suboff:suboff+these_chem_lens[irank]] = chem_coeff
                        #fit[these_offsets[0]:these_offsets[1]] = chem_coeff
                        fit[tst_off[0]:tst_off[1]] = chem_coeff
                        
                        
                # self.pt.single_print('printing fit: ', reg.coef_)
                self.fit = fit
                residues = np.matmul(aw,fit) - bw

        def _dump_a(self):
            np.savez_compressed('a.npz', a= self.pt.shared_arrays['a'].array)

        def _dump_x(self):
            np.savez_compressed('x.npz', x=self.fit)

        def _dump_b(self):
            b = self.pt.shared_arrays['a'].array @ self.fit
            np.savez_compressed('b.npz', b=b)

except ModuleNotFoundError:

    class SEQ_ACE_LASSO(Solver):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            raise ModuleNotFoundError("Missing sympy or pyyaml modules.")
