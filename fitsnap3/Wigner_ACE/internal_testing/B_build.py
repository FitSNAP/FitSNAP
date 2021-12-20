import numpy as np
import scipy
from multiprocessing import pool,cpu_count
from gen_labels import *
from internal_testing.site_basis import *
from internal_testing.convert_configs import *
from internal_testing.B_sums import *
from coupling_coeffs import *
from scipy import special
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from ase.neighborlist import primitive_neighbor_list
from ase import Atoms,Atom
from ase.io import read,write


#GLOBAL VARIABLES
multiproc = False # flag to use python multiprocessing to evaluate invariants (can be fast, but often slows down 
		  #  code with wait time
ncpus = int(cpu_count()) # number of cpus used for python multiprocessing

def simple_invariant(Abasis,rank,**kwargs):
	#try to load generalized cg coefficients
	cc_flag = False
	try:
		ext_ccs = kwargs['ccs']
		cc_flag = True
	except KeyError:
		pass
	#-----------------------------------------------------
	# basis functions
	#-----------------------------------------------------
	#single bond (2-body) basis functions
	if rank ==1:
		B= {}
		for n in range(1,Abasis.rb.nradmax+1):
			B['%d,0,0'%n] = 0.
			B['%d,0,0_ms'%n] = ['0']
		for n in range(1,Abasis.rb.nradmax+1):
			B['%d,0,0'%n] += np.sum( Abasis.A_1(n) )

	# two bond (3-body) basis functions
	# note that the wigner 3j symbol for (l1,m1,l2,m2,0,0) can be simplified.
	# wigner_3j function is not called for basis functions of this rank
	elif rank ==2:
		if not cc_flag:
			nls = generate_nl(rank,Abasis.rb.nradmax,Abasis.ab.lmax)
		elif cc_flag:
			nls = ext_ccs.keys()
		B = {nl:0. for nl in nls}
		for nl in nls:
			B[nl + '_ms'] = None
		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2 = nl_splt[0],nl_splt[1]
			l = nl_splt[-1]
			B['%d,%d,%d' % (n1,n2,l)] = 0.
			B['%d,%d,%d_ms' % (n1,n2,l)] = None

		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2 = nl_splt[0],nl_splt[1]
			l = nl_splt[-1]
			func,mlst = rank_2_invariant(Abasis,n1,n2,l)
			B['%d,%d,%d'%(n1,n2,l)] = func
			B['%d,%d,%d'%(n1,n2,l) + '_ms'] = sorted(list(set(mlst)))

	elif rank == 3:
		if not cc_flag:
			nls = generate_nl(rank,Abasis.rb.nradmax,Abasis.ab.lmax)
		elif cc_flag:
			nls = ext_ccs.keys()
		B = {nl:0. for nl in nls}
		for nl in nls:
			B[nl + '_ms'] = None
		for nl in nls:
			B[nl + '_cc'] = None
		args = []
		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2,n3 = nl_splt[0],nl_splt[1],nl_splt[2]
			l1,l2,l3 = nl_splt[3],nl_splt[4],nl_splt[5]
			if cc_flag:
				ccs = ext_ccs[nl]
			elif not cc_flag:
				ccs = rank_3_ccs([n1,n2,n3],[l1,l2,l3])
			args.append({'A':Abasis,'ccs': ccs,'n':[n1,n2,n3],'l':[l1,l2,l3]})
		if multiproc:
			pl = pool.Pool(processes=int(ncpus))
			results = pl.map(rank_3_invariant,args)
		else:
			results = map(rank_3_invariant,args)
		for result in results:
			for key,value in result.items():
				B[key] = value
		
	elif rank == 4:
		if not cc_flag:
			nls = generate_nl(rank,Abasis.rb.nradmax,Abasis.ab.lmax)
		elif cc_flag:
			nls = ext_ccs.keys()
		B = {nl:0. for nl in nls}
		for nl in nls:
			B[nl + '_ms'] = None
		for nl in nls:
			B[nl + '_cc'] = None
		args = []
		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2,n3,n4 = nl_splt[0],nl_splt[1],nl_splt[2],nl_splt[3]
			l1,l2,l3,l4 = nl_splt[4],nl_splt[5],nl_splt[6],nl_splt[7]
			if cc_flag:
				ccs = ext_ccs[nl]
			elif not cc_flag:
				ccs = rank_4_ccs([n1,n2,n3,n4],[l1,l2,l3,l4])
			args.append({'A':Abasis,'ccs':ccs,'n':[n1,n2,n3,n4],'l':[l1,l2,l3,l4]})
		if multiproc:
			pl = pool.Pool(processes=int(ncpus))
			results = pl.map(rank_4_invariant,args)
		else:
			results = map(rank_4_invariant,args)
		for result in results:
			for key,value in result.items():
				B[key] = value
	elif rank == 5:
		if not cc_flag:
			nls = generate_nl(rank,Abasis.rb.nradmax,Abasis.ab.lmax)
		elif cc_flag:
			nls = ext_ccs.keys()
		B = {nl:0. for nl in nls}
		for nl in nls:
			B[nl + '_ms'] = None
		for nl in nls:
			B[nl + '_cc'] = None
		args = []
		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2,n3,n4,n5 = nl_splt[0],nl_splt[1],nl_splt[2],nl_splt[3],nl_splt[4]
			l1,l2,l3,l4,l5 = nl_splt[5],nl_splt[6],nl_splt[7],nl_splt[8],nl_splt[9]
			if cc_flag:
				ccs = ext_ccs[nl]
			elif not cc_flag:
				ccs = rank_5_ccs([n1,n2,n3,n4,n5],[l1,l2,l3,l4,l5])
			args.append({'A':Abasis,'ccs':ccs,'n':[n1,n2,n3,n4,n5],'l':[l1,l2,l3,l4,l5]})
		if multiproc:
			pl = pool.Pool(processes=int(ncpus))
			results = pl.map(rank_5_invariant,args)
		else:
			results = map(rank_5_invariant,args)
		for result in results:
			for key,value in result.items():
				B[key] = value

	elif rank == 6:
		if not cc_flag:
			nls = generate_nl(rank,Abasis.rb.nradmax,Abasis.ab.lmax)
		elif cc_flag:
			nls = ext_ccs.keys()
		B = {nl:0. for nl in nls}
		for nl in nls:
			B[nl + '_ms'] = None
		for nl in nls:
			B[nl + '_cc'] = None
		args = []
		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2,n3,n4,n5,n6 = nl_splt[0],nl_splt[1],nl_splt[2],nl_splt[3],nl_splt[4],nl_splt[5]
			l1,l2,l3,l4,l5,l6 = nl_splt[6],nl_splt[7],nl_splt[8],nl_splt[9],nl_splt[10],nl_splt[11]
			if cc_flag:
				ccs = ext_ccs[nl]
			elif not cc_flag:
				ccs = rank_6_ccs([n1,n2,n3,n4,n5,n6],[l1,l2,l3,l4,l5,l6])
			args.append({'A':Abasis,'ccs':ccs,'n':[n1,n2,n3,n4,n5,n6],'l':[l1,l2,l3,l4,l5,l6]})
		if multiproc:
			pl = pool.Pool(processes=int(ncpus))
			results = pl.map(rank_6_invariant,args)
		else:
			results = map(rank_6_invariant,args)
		for result in results:
			for key,value in result.items():
				B[key] = value

	elif rank == 7:
		if not cc_flag:
			nls = generate_nl(rank,Abasis.rb.nradmax,Abasis.ab.lmax)
		elif cc_flag:
			nls = ext_ccs.keys()
		B = {nl:0. for nl in nls}
		for nl in nls:
			B[nl + '_ms'] = None
		for nl in nls:
			B[nl + '_cc'] = None
		args = []
		for nl in nls:
			nl_splt = [int(k) for k in nl.split(',')]
			n1,n2,n3,n4,n5,n6,n7 = nl_splt[0],nl_splt[1],nl_splt[2],nl_splt[3],nl_splt[4],nl_splt[5],nl_splt[6]
			l1,l2,l3,l4,l5,l6,l7 = nl_splt[7],nl_splt[8],nl_splt[9],nl_splt[10],nl_splt[11],nl_splt[12],nl_splt[13]
			if cc_flag:
				ccs = ext_ccs[nl]
			elif not cc_flag:
				ccs = rank_7_ccs([n1,n2,n3,n4,n5,n6,n7],[l1,l2,l3,l4,l5,l6,l7])
			args.append({'A':Abasis,'ccs':ccs,'n':[n1,n2,n3,n4,n5,n6,n7],'l':[l1,l2,l3,l4,l5,l6,l7]})
		if multiproc:
			pl = pool.Pool(processes=int(ncpus))
			results = pl.map(rank_7_invariant,args)
		else:
			results = map(rank_7_invariant,args)
		for result in results:
			for key,value in result.items():
				B[key] = value
	return B


def get_descriptors(args,**kwargs):
	i_d = args['id']
	atoms = args['atoms']
	print (i_d,atoms)
	rc = args['rc']
	nradbase=args['nradbase']
	nradmax_dict =args['nradmax']
	lmax_dict = args['lmax']
	lmbda=args['lmbda']
	ranks=args['rank']
	nus=args['nus']
	tol = 0.1

	#Build the neighbor list for the atoms (including periodic images if applicable)
	nl = primitive_neighbor_list('ijdD',pbc=atoms.pbc,positions=atoms.positions ,cell=atoms.get_cell(),cutoff=rc+tol)
	atinds = [atom.index for atom in atoms]
	at_neighs = { i: [] for i in nl[0]}
	at_dists = {i:[] for i in nl[0]}
	for i,j in zip(nl[0],nl[1]):
		at_neighs[i].append(j)
	for i,j in zip(nl[0],nl[-1]):
		at_dists[i].append(j)
	#if a transformation matrix is supplied, transform the neighbor positions accordingly
	# (Note that this is used to check for rotation, inversion, and/or reflection invariance)
	try:
		transform = kwargs['transformation']
		for i in [atom.index for atom in atoms]:
			positions = at_dists[i]
			pos_rot = np.matmul(transform.as_matrix(),positions.T)
			at_dists[i] =pos_rot
	except KeyError:
		at_dists = at_dists

	#if a different basis is desired, use it to define the radial functions
	bflag=False
	try:
		basis = kwargs['basis']
		bflag = True
	except KeyError:
		pass
	#if coupling coefficients are provided, load them
	ccflag = False
	try:
		ranked_ccs = kwargs['ccs']
		ccflag = True
	except KeyError:
		print ('no generalized couplings provided... will calculate on the fly')
		pass
	#if radial function expansion coefficients are provided, load them
	cradflag = False
	try:
		crad=kwargs['crad']
		cradflag=True
	except KeyError:
		pass
	B_by_atom = {atom.index: {nu:0 for nu in nus} for atom in atoms}
	B_coupling = {rank: {nu+'_cc':None for nu in nus if get_nu_rank(nu) == rank} for rank in ranks }
	B_ms = {rank: {nu+'_ms':None for nu in nus if get_nu_rank(nu) == rank} for rank in ranks }
	for rank in ranks:
		for atind in at_dists.keys():#atinds:
			r = np.array([np.linalg.norm(p) for p in at_dists[atind]])
			rb = radial_basis(r_arr = r, rc=rc, nradbase=nradbase, nradmax=nradmax_dict[rank], lmax=lmax_dict[rank], lmbda=lmbda)
			if rank ==1:
				rb = radial_basis(r_arr = r, rc=rc, nradbase=nradbase, nradmax=nradbase, lmax=lmax_dict[rank], lmbda=lmbda)
			if bflag:
				if cradflag:
					if rank !=1:
						rb.set_basis(basis,**{'crad':crad})
				elif not cradflag:
					rb.set_basis(basis)
			elif not bflag:
				if cradflag:
					basis = rb.basis
					if rank !=1:
						rb.set_basis(basis,**{'crad':crad})
				else:
					pass
			ab = angular_basis(at_dists[atind],lmax=lmax_dict[rank])
			Abasis = A_basis(rb,ab)
			if ccflag:
				try:
					B_dict = simple_invariant(Abasis,rank,**{'ccs':ranked_ccs[rank]})
				except KeyError:
					B_dict = simple_invariant(Abasis,rank,**{'ccs':ranked_ccs[str(rank)]})
				
			elif not ccflag:
				B_dict = simple_invariant(Abasis,rank)

			for key in B_dict.keys():
				if 'cc' not in key:
					B_by_atom[atind][key]=B_dict[key]
				elif 'cc' in key:
					ms = []
					cc = []
					for mstr in B_dict[key].keys():
						cc.append(B_dict[key][mstr])
						ms.append(mstr)
					B_coupling[key] = cc
					B_ms[key] = cc
	return B_by_atom,B_coupling,args,B_ms

#wrapper for mapping functions
def get_descriptors_wrapper(arg):
	args, kwargs = arg
	return get_descriptors(args, **kwargs)
