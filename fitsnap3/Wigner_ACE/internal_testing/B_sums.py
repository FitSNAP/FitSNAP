import numpy as np
import scipy
from multiprocessing import pool,cpu_count
from gen_labels import *
from internal_testing.site_basis import *
from internal_testing.convert_configs import *
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


#FUNCTIONS

def rank_2_invariant(Abasis,n1,n2,l):
	func = 0.
	ccs = {}
	for m in range(-l,l+1):
		A1 = Abasis.A(n1,l,m)
		A2 = Abasis.A(n2,l,-m)
		prd =((-1)**m) *A1*A2
		func += prd
		ccs['%d,%d' % (-m,m)]= (-1)**m
	return func,ccs


def rank_3_invariant(args):
	Abasis = args['A']
	ccs = args['ccs']
	ms = []
	n1,n2,n3 = args['n'][0] , args['n'][1], args['n'][2]
	l1,l2,l3 = args['l'][0] , args['l'][1], args['l'][2]

	func = 0.
	key = '%d,%d,%d,%d,%d,%d' % (n1,n2,n3,l1,l2,l3)
	coupling_coeffs = {}
	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				mflag = (m3+ m1+m2) ==0
				if mflag:
					A1 =Abasis.A(n1,l1,m1)
					A2 =Abasis.A(n2,l2,m2)
					A3 =Abasis.A(n3,l3,m3)
					mstr = '%d,%d,%d' %(m1,m2,m3) 
					cg1 = ccs[mstr] 
					prd = cg1*A1*A2*A3
					func += prd
					mstr = '%d,%d,%d' %(m1,m2,m3) 

					coupling_coeffs[mstr] = cg1
	return {key:func,key+'_cc':coupling_coeffs}

def rank_4_invariant(args,**kwargs):
	Abasis = args['A']
	ccs = args['ccs']
	ms = []
	n1,n2,n3,n4 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3]
	l1,l2,l3,l4 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3]
	l12s = get_intermediates('%d,%d' % (l1,l2))
	key = '%d,%d,%d,%d,%d,%d,%d,%d' % (n1,n2,n3,n4,l1,l2,l3,l4)

	full_func = 0
	full_couplings = {}
	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				for m4 in range(-l4,l4+1):
					mflag = (m1+m2+m3+m4)==0
					if mflag:
						A1 = Abasis.A(n1,l1,m1)
						A2 = Abasis.A(n2,l2,m2)
						A3 = Abasis.A(n3,l3,m3)
						A4 = Abasis.A(n4,l4,m4)
						mstr='%d,%d,%d,%d' %(m1,m2,m3,m4) 
						try:
							prd = ccs[mstr] *A1*A2*A3*A4
						except KeyError:
							prd = 0.
						full_func += prd
						#full_couplings[key+ ',' + mstr] = ccs[mstr]

	return {key:full_func, key+'_cc':full_couplings}

def phiprd_only_rank_4(args):
	phi = args['phi']
	n1,n2,n3,n4 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3]
	l1,l2,l3,l4 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3]
	m1,m2,m3,m4 = args['m'][0] , args['m'][1], args['m'][2], args['m'][3]
	r1,r2,r3,r4 = args['rs'][0] , args['rs'][1], args['rs'][2], args['rs'][3]
	A1 = phi.phi(n1,l1,m1,r1)
	A2 = phi.phi(n2,l2,m2,r2)
	A3 = phi.phi(n3,l3,m3,r3)
	A4 = phi.phi(n4,l4,m4,r4)
	return A1*A2*A3*A4

def rank_4_phiprd(args,**kwargs):
	import itertools
	phi = args['phi']
	ccs = args['ccs']
	ms = []
	n1,n2,n3,n4 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3]
	l1,l2,l3,l4 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3]
	r1,r2,r3,r4 = args['rs'][0] , args['rs'][1], args['rs'][2], args['rs'][3]
	l12s = get_intermediates('%d,%d' % (l1,l2))
	key = '%d,%d,%d,%d,%d,%d,%d,%d' % (n1,n2,n3,n4,l1,l2,l3,l4)
	vecs = [i for i in itertools.permutations(range(4)) ]

	full_func = 0
	full_couplings = {}
	#for vec in vecs:
	for vec in [vecs[0]]:
	#for vec in [args['l']]:
		ltmp =  [args['l'][i] for i in vec]
		l1,l2,l3,l4 = ltmp[0],ltmp[1],ltmp[2],ltmp[3]
		#ccs_tmp = rank_4(ltmp)
		for m1 in range(-l1,l1+1):
			for m2 in range(-l2,l2+1):
				for m3 in range(-l3,l3+1):
					for m4 in range(-l4,l4+1):
						mflag = (m1+m2+m3+m4)==0
						if mflag:
							A1 = phi.phi(n1,l1,m1,r1)
							A1c = ((-1)**m1)*phi.phi(n1,l1,-m1,r1)
							A2 = phi.phi(n2,l2,m2,r2)
							A2c = ((-1)**m2)*phi.phi(n2,l2,-m2,r2)
							A3 = phi.phi(n3,l3,m3,r3)
							A3c = ((-1)**m3)*phi.phi(n3,l3,-m3,r3)
							A4 = phi.phi(n4,l4,m4,r4)
							A4c = ((-1)**m4)*phi.phi(n4,l4,-m4,r4)
							mstr='%d,%d,%d,%d' %(m1,m2,m3,m4)
							try:
								#prd = ccs[mstr] * (A1*A2*A3*A4 - A1c*A2c*A3c*A4c)
								#prd = ccs_tmp[mstr] * ( A1c*A2c*A3c*A4c)
								prd = ccs[mstr] * (A1*A2*A3*A4)
							except KeyError:
								prd = 0.
							full_func += prd
							#full_couplings[key+ ',' + mstr] = ccs[mstr]

	return full_func

def rank_5_invariant(args,**kwargs):
	Abasis = args['A']
	ccs = args['ccs']
	ms = []
	n1,n2,n3,n4,n5 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3], args['n'][4]
	l1,l2,l3,l4,l1234 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3], args['l'][4] 
	func = 0.
	key = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (n1,n2,n3,n4,n5,l1,l2,l3,l4,l1234)
	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				for m4 in range(-l4,l4+1):
					for m1234 in range(-l1234,l1234+1):
						if m1+m2+m3+m4+m1234==0:

							A1 = Abasis.A(n1,l1,m1)
							A2 = Abasis.A(n2,l2,m2)
							A3 = Abasis.A(n3,l3,m3)
							A4 = Abasis.A(n4,l4,m4)
							A5 = Abasis.A(n5,l1234,m1234)
							cc = ccs['%d,%d,%d,%d,%d' %(m1,m2,m3,m4,m1234)]
							prd = cc * A1*A2*A3*A4*A5
							func += prd
	return {key:func,key + '_cc': ccs}


def rank_6_invariant(args,**kwargs):
	Abasis = args['A']
	ccs = args['ccs']
	ms = []
	n1,n2,n3,n4,n5,n6 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3], args['n'][4], args['n'][5]
	l1,l2,l3,l4,l5,l12345 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3], args['l'][4], args['l'][5]
	func = 0.
	key = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (n1,n2,n3,n4,n5,n6,l1,l2,l3,l4,l5,l12345)

	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				for m4 in range(-l4,l4+1):
					for m5 in range(-l5,l5+1):
						for m12345 in range(-l12345,l12345+1):
							if m1+m2+m3+m4+m5+m12345==0:

								A1 = Abasis.A(n1,l1,m1)
								A2 = Abasis.A(n2,l2,m2)
								A3 = Abasis.A(n3,l3,m3)
								A4 = Abasis.A(n4,l4,m4)
								A5 = Abasis.A(n5,l5,m5)
								A6 = Abasis.A(n6,l12345,m12345)
								cc = ccs['%d,%d,%d,%d,%d,%d' %(m1,m2,m3,m4,m5,m12345)]
								prd = cc * A1*A2*A3*A4*A5*A6
								func += prd
	return {key:func,key + '_cc': ccs}


def rank_7_invariant(args,**kwargs):
	Abasis = args['A']
	ccs = args['ccs']
	ms = []
	n1,n2,n3,n4,n5,n6,n7 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3], args['n'][4], args['n'][5], args['n'][6]
	l1,l2,l3,l4,l5,l6,l123456 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3], args['l'][4], args['l'][5], args['l'][6]
	func = 0.
	key = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (n1,n2,n3,n4,n5,n6,n7,l1,l2,l3,l4,l5,l6,l123456)
	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				for m4 in range(-l4,l4+1):
					for m5 in range(-l5,l5+1):
						for m6 in range(-l6,l6+1):
							for m123456 in range(-l123456,l123456+1):
								if (m1+m2+m3+m4+m5+m6+m123456)==0:

									A1 = Abasis.A(n1,l1,m1)
									A2 = Abasis.A(n2,l2,m2)
									A3 = Abasis.A(n3,l3,m3)
									A4 = Abasis.A(n4,l4,m4)
									A5 = Abasis.A(n5,l5,m5)
									A6 = Abasis.A(n6,l6,m6)
									A7 = Abasis.A(n7,l123456,m123456)
									cc = ccs['%d,%d,%d,%d,%d,%d,%d' %(m1,m2,m3,m4,m5,m6,m123456)]
									prd = cc * A1*A2*A3*A4*A5*A6*A7
									func += prd
	return {key:func,key + '_cc': ccs}

def rank_8_invariant(args,**kwargs):
	Abasis = args['A']
	ccs = args['ccs']
	ms = []
	n1,n2,n3,n4,n5,n6,n7,n8 = args['n'][0] , args['n'][1], args['n'][2], args['n'][3], args['n'][4], args['n'][5], args['n'][6], args['n'][7]
	l1,l2,l3,l4,l5,l6,l7,l1234567 = args['l'][0] , args['l'][1], args['l'][2], args['l'][3], args['l'][4], args['l'][5], args['l'][6], args['l'][7]
	func = 0.
	key = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (n1,n2,n3,n4,n5,n6,n7,n8,l1,l2,l3,l4,l5,l6,l7,l1234567)
	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				for m4 in range(-l4,l4+1):
					for m5 in range(-l5,l5+1):
						for m6 in range(-l6,l6+1):
							for m7 in range(-l7,l7+1):
								for m1234567 in range(-l1234567,l1234567+1):
									if (m1+m2+m3+m4+m5+m6+m7+m1234567)==0:

										A1 = Abasis.A(n1,l1,m1)
										A2 = Abasis.A(n2,l2,m2)
										A3 = Abasis.A(n3,l3,m3)
										A4 = Abasis.A(n4,l4,m4)
										A5 = Abasis.A(n5,l5,m5)
										A6 = Abasis.A(n6,l6,m6)
										A7 = Abasis.A(n7,l7,m7)
										A8 = Abasis.A(n8,l1234567,m1234567)
										cc = ccs['%d,%d,%d,%d,%d,%d,%d,%d' %(m1,m2,m3,m4,m5,m6,m7,m1234567)]
										prd = cc * A1*A2*A3*A4*A5*A6*A7*A8
										func += prd
	return {key:func,key + '_cc': ccs}
