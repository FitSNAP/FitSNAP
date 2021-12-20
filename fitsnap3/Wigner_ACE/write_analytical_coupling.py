import numpy as np
from datetime import date
from .gen_labels import *
from .coupling_coeffs import *
from .wigner_couple import *
import json

#multiplicative factor for expansion coefficients (for conversion of units from native testing to LAMMPS ML-PACE)
multfac=1.

def get_m_cc(d,ranks):
	len_ms = []
	m_dict = {rank:None for rank in ranks}
	cc_dict  = {rank:None for rank in ranks}
	for rank in ranks:
		rnk = str(rank)
		keys = d[rnk].keys()
		ms_dict = {key:None for key in keys}
		ccs_dict = {key:None for key in keys}

		for key in ms_dict.keys():
			ms_dict[key] = list(d[rnk][key].keys())
			len_ms.append(len(ms_dict[key]))
			ccs_dict[key] = list(d[rnk][key].values())
		m_dict[rank] = ms_dict
		cc_dict[rank] = ccs_dict
	max_num_ms  = max(len_ms)
	return m_dict,cc_dict,max_num_ms
		

def write_pot(filname,element,ranks,lmax,nradbase,nradmax,rcut,exp_lambda, nus,coupling,coeffs,E_0):
	tol = 1.e-5
	today = date.today()
	dt = today.strftime("%y-%m-%d")
	write_str ="""# DATE: %s UNITS: metal CONTRIBUTOR: James Goff <jmgoff@sandia.gov> CITATION: py_PACE

nelements=1
elements: %s

lmax=%d

2 FS parameters:  1.000000 1.000000
core energy-cutoff parameters: 100000.000000000000000000 250.000000000000000000
E0:%8.32f

radbasename=ChebExpCos
nradbase=%d
nradmax=%d
cutoffmax=%2.10f
deltaSplineBins=0.001000
core repulsion parameters: 0.000000000000000000 1.000000000000000000
radparameter= %2.10f
cutoff= %2.10f
dcut= 0.010000000000000000\n""" % (dt,element,lmax,E_0,nradbase,nradmax,rcut+tol, exp_lambda,rcut)

	#radial basis function expansion coefficients
	#saved in n,l,k shape
	# defaults to orthogonal delta function [g(n,k)] basis of drautz 2019
	crad = np.zeros((nradmax,lmax+1,nradbase))
	for n in range(nradmax):
		for l in range(lmax+1):
			crad[n][l] = np.array([1. if k==n else 0. for k in range(nradbase)]) 

	cnew = np.zeros((nradbase,nradmax,lmax+1))
	for n in range(1,nradmax+1):
		for l in range(lmax+1):
			for k in range(1,nradbase+1):
				cnew[k-1][n-1][l] = crad[n-1][l][k-1]

	crd = """crad= """
	for k in range(nradbase):
		for row in cnew[k]:
			tmp = ' '.join(str(b) for b in row)
			tmp = tmp + '\n'
			crd = crd+tmp
	crd = crd + '\n'

	ms,ccs,max_num_m = get_m_cc(coupling,ranks)
	maxrank = max(ranks)
	write_str2 = """rankmax=%d
ndensitymax=1

num_c_tilde_max=%d
num_ms_combinations_max=%d\n""" % (maxrank,len(nus),max_num_m) 

	rank1 = """total_basis_size_rank1: %d\n""" % len(ms[1].keys())

	#----write rank 1s----
	for key in ms[1].keys():
		ctilde = """ctilde_basis_func: rank=1 ndens=1 mu0=0 mu=( 0 )
n=( %s )
l=( 0 )
num_ms=1
< 0 >:  %8.24f\n""" % (key.split(',')[0],coeffs[key]*multfac)
		rank1 = rank1+ctilde
	rankplus = """total_basis_size: %d\n""" % np.sum([len(ms[i].keys()) for i in range(2,maxrank+1)])
	for rank in range(2,maxrank+1):
		for key in ms[rank].keys():
			#print (key)
			try:
				c = coeffs[key]
			except KeyError:
				print ('Warning! no coefficient for %s' %key, 'using c_%s=0' %key)
				c=0
			nstrlst = [' %d ']*rank
			lstrlst = [' %d ']*rank
			mustrlst = [' %d '] *rank
			nstr = ''.join(n for n in nstrlst)
			lstr = ''.join(l for l in lstrlst)
			mustr = ''.join(l for l in mustrlst)
			ns,ls = get_n_l(key,**{'rank':rank})
			nstr = nstr % tuple(ns)
			lstr = lstr % tuple(ls)
			mustr = mustr % tuple([0] * rank)
			num_ms = len(ms[rank][key])
			ctilde = """ctilde_basis_func: rank=%d ndens=1 mu0=0 mu=( %s )
n=(%s)
l=(%s)
num_ms=%d\n"""%(rank,mustr,nstr,lstr,num_ms)
			for ind,m in enumerate(ms[rank][key]):
				if type(m) ==str:
					m = [int(kz) for kz in m.split(',')]
				mstr = ''.join(l for l in lstrlst)
				mkeystr= ','.join(y for y in ['%d']*rank)
				mkeystr= mkeystr % tuple(m)
				mstr = mstr % tuple(m)
				m_add = '<%s>:  %8.24f\n' % (mstr,(c*multfac)*ccs[rank][key][ind])
				ctilde = ctilde+m_add
			ctilde = ctilde
			rankplus = rankplus +ctilde
	
	with open('%s.ace' %filname,'w',encoding='utf8') as writeout:
		writeout.write(write_str)
		writeout.write(crd)
		writeout.write(write_str2)
		writeout.write(rank1)
		writeout.write(rankplus)

