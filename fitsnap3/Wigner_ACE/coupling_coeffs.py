from scipy import special
import pickle
import numpy as np
from .gen_labels import *
cglib = False

# TODO get a more elegant solution to the pickled library locations
import os
import sys
pkg_name = 'FitSNAP-1'
pkg_paths = [ p for p in sys.path if pkg_name in p and p.split('/')[-1] == pkg_name]
assert len(pkg_paths) >=1, "package %s not found in PYTHONPATH, add it to your path and check the name of your package" % pkg_name
lib_path = pkg_paths[0] + '/lib'


def uni_norm_2d(vec):
	#normalizes a 2d array with variable shape per index
	#e.g.
	vec_f=[item for sublist in vec for item in sublist]
	vecsum = np.sum([(i)**2 for i in vec_f])
	vsqr = np.sqrt(vecsum)

	
	vec_norm = [sublist/vsqr for sublist in vec]
	return vec_norm

def uni_norm_3d(vec,filled=False):
	#normalizes a 3d array with variable shape per index
	def reduce_1d(lst):
		return [item for sublist in lst for item in sublist]
	
	vec_2d = [reduce_1d(d3) for d3 in vec]
	if filled:
		vec_2d = [reduce_1d(d2) for d2 in vec_2d]
	vec_f=[(item**2) for sublist in vec_2d for item in sublist]
	
	vecsum = np.sum(vec_f)
	vsqr = np.sqrt(vecsum)

	vec_norm = [[[v/vsqr for v in sublist1] for sublist1 in sublist] for sublist in vec]
	return vec_norm

def uni_norm_4d(vec):
	#normalizes a 4d array with variable shape per index
	def reduce_1d(lst):
		return [item for sublist in lst for item in sublist]
	vec_3d = [reduce_1d(d4) for d4 in vec]	
	vec_2d = [reduce_1d(d3) for d3 in vec_3d]
	vec_f=[(item**2) for sublist in vec_2d for item in sublist]
	vecsum = np.sum(vec_f)
	vsqr = np.sqrt(vecsum)
	vec_norm = [[[[v/vsqr for v in sublist1] for sublist1 in sublist2] for sublist2 in sublist] for sublist in vec]
	return vec_norm


def Clebsch_gordan(j1,m1,j2,m2,j3,m3):
	# Clebsch-gordan coefficient calculator based on eqs. 4-5 of:
	# https://hal.inria.fr/hal-01851097/document
	# and christoph ortner's julia code ACE.jl

	#VERIFIED: test non-zero indices in Wolfram using format ClebschGordan[{j1,m1},{j2,m2},{j3,m3}]
	#rules:
	rule1 = np.abs(j1-j2) <= j3
	rule2 = j3 <= j1+j2
	rule3 = m3 == m1 + m2
	rule4 = np.abs(m3) <= j3

	#rules assumed by input
	#assert np.abs(m1) <= j1, 'm1 must be \in {-j1,j1}'
	#assert np.abs(m2) <= j2, 'm2 must be \in {-j2,j2}'

	if rule1 and rule2 and rule3 and rule4:
		#attempting binomial representation
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

		N = (N1*N2)/(N3)


		G = 0.

		#k conditions (see eq.5 of https://hal.inria.fr/hal-01851097/document)
		# k  >= 0
		# k <= j1 - m1
		# k <= j2 + m2

		for k in range(0, min([j1-m1, j2+m2]) + 1  ):
			G1 = (-1)**k
			G2 = special.comb(j1 + j2 - j3, k,exact=True)
			G3 = special.comb(j1 - j2 + j3, j1 - m1 - k,exact=True)
			G4 = special.comb(-j1 +j2 + j3, j2 + m2 - k,exact=True)
			G += G1*G2*G3*G4
		return (N**(1/2))*G 


	else:
		return 0.

def clebsch_gordan(l1,m1,l2,m2,l3,m3):
	# try to load c library for calculating cg coefficients
	if cglib:
		return lib.Clebsch_Gordan(l1,m1,l2,m2,l3,m3)
	else:
		return Clebsch_gordan(l1,m1,l2,m2,l3,m3)

def wigner_3j(j1,m1,j2,m2,j3,m3):
	# uses relation between Clebsch-Gordann coefficients and W-3j symbols to evaluate W-3j
	#VERIFIED - wolframalpha.com
	cg = clebsch_gordan(j1,m1,j2,m2,j3,-m3)

	num = (-1)**(j1-j2-m3)
	denom = ((2*j3) +1)**(1/2)

	return cg*(num/denom)


def init_clebsch_gordan(lmax):
	#returns dictionary of all cg coefficients to be used at a given value of lmax
	cg = {}
	for l1 in range(lmax+1):
		for l2 in range(lmax+1):
			#for l3 in range(abs(l1-l2),l1+l2+1):
			for l3 in range(lmax+1):
				for m1 in range(-l1,l1+1):
					for m2 in range(-l2,l2+1):
						for m3 in range(-l3,l3+1):
							key = '%d,%d,%d,%d,%d,%d' % (l1,m1,l2,m2,l3,m3)
							cg[key] = clebsch_gordan(l1,m1,l2,m2,l3,m3)
	return cg


def init_wigner_3j(lmax):
	#returns dictionary of all cg coefficients to be used at a given value of lmax
	cg = {}
	for l1 in range(lmax+1):
		for l2 in range(lmax+1):
			#for l3 in range(abs(l1-l2),l1+l2+1):
			for l3 in range(lmax+1):
				for m1 in range(-l1,l1+1):
					for m2 in range(-l2,l2+1):
						for m3 in range(-l3,l3+1):
							key = '%d,%d,%d,%d,%d,%d' % (l1,m1,l2,m2,l3,m3)
							cg[key] = wigner_3j(l1,m1,l2,m2,l3,m3)
	return cg



# store a large dictionary of clebsch gordan coefficients
try:
	with open('%s/Clebsch_Gordan.pickle' %lib_path, 'rb') as handle:
		Clebsch_Gordan = pickle.load(handle)
except FileNotFoundError:
	print ("Generating your first pickled library of CG coefficients. This will take a few moments...")
	Clebsch_Gordan = init_clebsch_gordan(14)
	with open('%s/Clebsch_Gordan.pickle' %lib_path, 'wb') as handle:
		pickle.dump(Clebsch_Gordan, handle, protocol=pickle.HIGHEST_PROTOCOL)
# do the same thing for the wigner_3j symbols
try:
	with open('%s/Wigner_3j.pickle' % lib_path, 'rb') as handle:
		Wigner_3j = pickle.load(handle)
except FileNotFoundError:
	print ("Generating your first pickled library of Wigner 3j coefficients. This will take a few moments...")
	Wigner_3j = init_wigner_3j(14)
	with open('%s/Wigner_3j.pickle' % lib_path, 'wb') as handle:
		pickle.dump(Wigner_3j, handle, protocol=pickle.HIGHEST_PROTOCOL)

def rank_1_ccs():
	return {'0':1.}

def rank_2_ccs(n,l):
	if len(l) ==2:
		l = l[0]
	coupling_coeff = {}
	for m in range(-l,l+1):
		coupling_coeff['%d,%d'%(m,-m)] = ((-1)**m) 
	return coupling_coeff

def rank_3_ccs(n,l):
	l1,l2,l3 = l[0],l[1],l[2]
	coupling_coeff = {}
	for m1 in range(-l1,l1+1):
		for m2 in range(-l2,l2+1):
			for m3 in range(-l3,l3+1):
				mflag = (m1+m2+m3)==0
				if mflag:
					coupling_coeff['%d,%d,%d'% (m1,m2,m3)] = Wigner_3j['%d,%d,%d,%d,%d,%d'%(l1,m1,l2,m2,l3,m3)]
	return coupling_coeff

