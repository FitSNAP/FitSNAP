from internal_testing.site_basis import *
from internal_testing.B_sums import *
import numpy as np

# objective functions for optimizing intermediate weights

def residual(n,l,rank,rc,nradbase,nradmax,lmax,lmbda,ccs):
	try:
		positions = np.load('positions.npy')
	except FileNotFoundError:
		positions = np.random.rand(1000,3)
		np.save('positions.npy',positions)
	positions0 = np.multiply(10,positions[:int(len(positions)/2)])
	positions1 = np.multiply(-10,positions[int(len(positions)/2):])

	# random rotation vector
	r = R.from_rotvec(np.pi/2 * np.random.rand(3))
	pos_rot = np.matmul(r.as_matrix(),positions.T)
	r_arr0 = np.array([np.linalg.norm(p) for p in positions])
	rb =radial_basis(r_arr0, rc, nradbase, nradmax, lmax, lmbda)
	ab = angular_basis(positions,lmax)
	Abasis=A_basis(rb,ab)
	
	pos_rot = pos_rot.T

	r_arr_rot = np.array([np.linalg.norm(p) for p in pos_rot])
	rb_r = radial_basis(r_arr_rot,rc,nradbase,nradmax,lmax,lmbda)
	ab_r = angular_basis(pos_rot,lmax)
	Abasis_r = A_basis(rb_r,ab_r)

	keylst = ['%d']*2*rank
	key = ','.join(k for k in keylst) % tuple(n +l)
	if rank ==4:
		l1,l2,l3,l4 = l[0],l[1],l[2],l[3]
		f1 = rank_4_invariant({'A':Abasis,  'n':n,'l':l,'ccs':ccs})
		f2 = rank_4_invariant({'A':Abasis_r,'n':n,'l':l,'ccs':ccs})

	elif rank ==5:
		l1,l2,l3,l4,l5 = l[0],l[1],l[2],l[3],l[4]
		f1 = rank_5_invariant({'A':Abasis,'n':n,'l':l,'ccs':ccs})
		f2 = rank_5_invariant({'A':Abasis_r,'n':n,'l':l,'ccs':ccs})

	elif rank ==6:
		l1,l2,l3,l4,l5,l6 = l[0],l[1],l[2],l[3],l[4],l[5]
		f1 = rank_6_invariant({'A':Abasis,'n':n,'l':l,'ccs':ccs})
		f2 = rank_6_invariant({'A':Abasis_r,'n':n,'l':l,'ccs':ccs})
	elif rank ==7:
		l1,l2,l3,l4,l5,l6,l7 = l[0],l[1],l[2],l[3],l[4],l[5],l[6]
		f1 = rank_7_invariant({'A':Abasis,'n':n,'l':l,'ccs':ccs})
		f2 = rank_7_invariant({'A':Abasis_r,'n':n,'l':l,'ccs':ccs})
	elif rank ==8:
		l1,l2,l3,l4,l5,l6,l7,l8 = l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7]
		f1 = rank_8_invariant({'A':Abasis,'n':n,'l':l,'ccs':ccs})
		f2 = rank_8_invariant({'A':Abasis_r,'n':n,'l':l,'ccs':ccs})
	else:
		raise ValueError("residuals for rank %d descriptors are not implemented" % rank)

	d = np.abs(((f1[key]-f2[key])/(len(positions)**rank)))
	return d

