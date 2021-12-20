from gen_labels import *
from internal_testing.opt_funcs import *
from wigner_couple import *
import cProfile


def __main__():
	ranks = range(1,9)
	rc=7.5
	nradbase=16
	nradmax_dict = {1:nradbase,2:5,3:4,4:2,5:1,6:1,7:1,8:1}
	lmax_dict = {1:2,2:6,3:4,4:3,5:2,6:2,7:1,8:1}
	lmbda=5.
	ranked_nus = {rank:generate_nl(rank,nradmax_dict[rank],lmax_dict[rank]) for rank in ranks}
	print ('ROTATIONIAL INVARIANCE TESTS')
	print ('nu random rotation residual MAE/(atom position)')
	for rank in ranks:
		if rank >=4:
			for nu in ranked_nus[rank]:
				n,l = get_n_l(nu)
				if rank ==4:
					ccs =  rank_4(l)
				if rank ==5:
					ccs =  rank_5(l)
				if rank ==6:
					ccs =  rank_6(l)
				if rank ==7:
					ccs =  rank_7(l)
				if rank ==8:
					ccs =  rank_8(l)
				res = residual(n,l,rank,7.5,16,4,4,5.0,ccs)
				print(nu,res)

if __name__ == '__main__':
	cProfile.run('__main__()', sort='cumulative')

