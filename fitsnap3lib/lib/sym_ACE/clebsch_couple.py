import numpy as np
from fitsnap3lib.lib.sym_ACE.clebsch_tree import *
from sympy.combinatorics import Permutation
from functools import partial


def get_cg_coupling(ldict,L_R=0,use_permutations=True):
	M_Rs = list(range(-L_R,L_R+1))
	#generic coupling for any L_R - support must be added to call 
	ranks = list(ldict.keys())
	coupling = {M_R : {rank:{} for rank in ranks} for M_R in M_Rs}

	for M_R in M_Rs:
		for rank in ranks:
			rnk = rank
			ls_per_rnk = generate_l_LR(range(ldict[rank]+1),rank,L_R,M_R,use_permutations = use_permutations)
			for lstr in ls_per_rnk:
				l = [int(k) for k in lstr.split(',')]
				if rank ==1:
					decomped = rank_1_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==2:
					decomped = rank_2_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==3:
					decomped = rank_3_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==4:
					decomped = rank_4_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==5:
					decomped = rank_5_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==6:
					decomped = rank_6_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==7:
					decomped = rank_7_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank ==8:
					decomped = rank_8_cg_tree(l,L_R,M_R)
					coupling[M_R][rnk][lstr] = decomped
				elif rank > 8:
					raise ValueError("Cannot generate couplings for rank %d. symmetric L_R couplings up to rank 8 have been implemented" % rank)
	return coupling

