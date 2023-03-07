"""
Definitions of functions to build pickled libraries of coupling coefficients and Wigner 3j 
coefficients. 

Unclear if this is nice to have in a class; will keep this and others as collection of functions 
for now.
"""

import numpy as np
from fitsnap3lib.lib.sym_ACE.coupling_coeffs import *
from fitsnap3lib.lib.sym_ACE.tree_method import *
from fitsnap3lib.lib.sym_ACE.gen_labels import *
from sympy.combinatorics import Permutation
from functools import partial


def get_coupling(ldict, Wigner_3j, L_R=0,**kwargs):
    M_Rs = list(range(-L_R,L_R+1))
    #generic coupling for any L_R - support must be added to call 
    ranks = list(ldict.keys())
    coupling = {M_R : {rank:{} for rank in ranks} for M_R in M_Rs}

    #weights are only necessary when generating linear combinations of wigner coefficients
    weights = None
    for M_R in M_Rs:
        for rank in ranks:
            rnk = rank
            ls_per_rnk = generate_l_LR(range(ldict[rank]+1),rank,L_R,M_R)
            for lstr in ls_per_rnk:
                l = [int(k) for k in lstr.split(',')]
                if rank ==1:
                    decomped = rank_1_tree(l,Wigner_3j, L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==2:
                    decomped = rank_2_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==3:
                    decomped = rank_3_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==4:
                    decomped = rank_4_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==5:
                    decomped = rank_5_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==6:
                    decomped = rank_6_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==7:
                    decomped = rank_7_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank ==8:
                    decomped = rank_8_tree(l,Wigner_3j,L_R,M_R)
                    coupling[M_R][rnk][lstr] = decomped
                elif rank > 8:
                    raise ValueError("Cannot generate couplings for rank %d. symmetric L_R couplings up to rank 6 have been implemented" % rank)
    return coupling,weights



def local_coupling(l,L_R=0,**kwargs):
    M_Rs = list(range(-L_R,L_R+1))
    #generic coupling for any L_R - support must be added to call 
    ranks = [len(l)]
    coupling = {M_R : {rank:{} for rank in ranks} for M_R in M_Rs}

    #weights are only necessary when generating linear combinations of wigner coefficients
    weights = None
    for M_R in M_Rs:
        rank = len(l)
        rnk = rank
        lstr = ','.join([str(li) for li in l])
        if type(l) == str:
            l = [int(k) for k in l.split(',')]
        if rank ==1:
            decomped = rank_1_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==2:
            decomped = rank_2_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==3:
            decomped = rank_3_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==4:
            decomped = rank_4_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==5:
            decomped = rank_5_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==6:
            decomped = rank_6_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==7:
            decomped = rank_7_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank ==8:
            decomped = rank_8_tree(l,L_R,M_R)
            coupling[M_R][rnk][lstr] = decomped
        elif rank > 8:
            raise ValueError("Cannot generate couplings for rank %d. symmetric L_R couplings up to rank 6 have been implemented" % rank)
    return coupling,weights

def generate_ccs():
    """
    Define and return coupling coefficients, called by AcePot initializer in potential.py
    """
    Wigner_3j = get_w3j_and_cg() # from coupling_coeffs.py

    import time
    t1 =time.time()
    try:
        with open('%s/global_ccs_LR_0_MR_0.pickle' % lib_path, 'rb') as handle:
            global_ccs = pickle.load(handle)
    except FileNotFoundError:
        global_ccs_all,global_weights = get_coupling(lmax_dict_G, Wigner_3j)
        global_ccs = global_ccs_all[0]
        with open('%s/global_ccs_LR_0_MR_0.pickle' % lib_path, 'wb') as handle:
            pickle.dump(global_ccs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if gen_LR1:
        try:
            with open('%s/global_ccs_LR_1.pickle' % lib_path, 'rb') as handle:
                global_ccs_l1 = pickle.load(handle)
        except FileNotFoundError:
            global_ccs_l1,global_weights = get_coupling(lmax_dict_G, Wigner_3j, L_R=1)
            with open('%s/global_ccs_LR_1.pickle' % lib_path, 'wb') as handle:
                pickle.dump(global_ccs_l1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # t2 = time.time()
    #print ('Time to generate/load global generalized couplings:', (t2-t1)*1000, 'ms', '\n', 'global lmax per descriptor rank:', lmax_dict_G)

    return global_ccs
