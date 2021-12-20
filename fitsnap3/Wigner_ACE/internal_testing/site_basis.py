import numpy as np
import math
import scipy
from scipy import special
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# asmuth and polar angles (spherical polar units)
def get_spher_pol(r_vecs,r_arr):
	#This will still throw a runtime divide by zero encounter warning, but it will still return
	#  a 0 when lim(y/x)-> 0 and pi/2 when lim(y/x)-> inf
	#carts = [np.divide(r_vec,r) for r_vec,r in zip(r_vecs,r_arr)  ]
	#asmuths = [np.arctan(np.nan_to_num(cart[1]/cart[0])) for cart in carts]
	#polars = [np.arccos(cart[2]/np.linalg.norm(cart)) for ind,cart in enumerate(carts)]
	rhats = [np.divide(r_vec,r) for r_vec,r in zip(r_vecs,r_arr)  ]
	polars = [math.atan2(np.sqrt((cart[0]**2) + (cart[1]**2)),cart[2]) for cart in rhats]
	asmuths = [math.atan2(cart[1],cart[0]) for cart in rhats]

	return np.array([asmuths,polars])

def cut_func_general(r_scale,func):
	mult = []
	for r in r_scale:
		if r <= 1:
			mult.append(1.)
		else:
			mult.append(0.)
	return mult *func


class radial_basis:
	def __init__(self,
			r_arr, #np.array - atomic separations in neighbor list
			rc, #float - cutoff distance
			nradbase, #int-max k index in 
			nradmax, #int-max n index in Rnl
			lmax, #int-max l index in Rnl
			lmbda): #float - exponential factor in g(x)
		self.r_arr = r_arr
		self.rc = rc
		self.nradbase = nradbase
		self.nradmax = nradmax
		self.lmax = lmax
		self.lmbda = lmbda
		self.basis = 'ChebExpCos'
		#initialize g_ks
		self.init_func()
		#initialize rnls
		self.init_R_nl()
		return None


	def init_func(self):
		g = {}
		for n in range(1,self.nradbase+1):
			g[n] = self.G(n)
		self.gks = g
		return None

	def g(self,n):
		return self.gks[n]

	def x_scale(self):
		r_scale = self.r_arr/ self.rc
		numerator = np.exp(-self.lmbda *(r_scale -1)) - 1
		denominator = np.exp(self.lmbda)-1
		x = 1 - (2 *(numerator/denominator))
		return x

	def cut_func(self,func):
		r_scale = self.r_arr/self.rc
		rsc = []
		for r in r_scale:
			if r <= 1:
				rsc.append(1.)
			else:
				rsc.append(0.)
		rsc = np.array(rsc)
	
		return rsc*func
	
	def G(self,k):
		r_scale = self.r_arr/self.rc
		if k == 0:
			func = np.ones(len(self.r_arr))

		if k == 1:
			func = (1/2)*np.add(1,np.cos(np.pi*r_scale))

		if k > 1:
			x = self.x_scale()
			cheb1 = special.eval_chebyt(k-1,x)
			func = (1/4)*(1 -cheb1) *np.add(1,np.cos(np.pi*r_scale))

		return self.cut_func(func)


	def R_nl(self,n,l,crad):
		rnl = np.zeros(len(self.r_arr))
		gs = { i: None for i in range(self.nradbase +1)[1:] }
		for i in range(self.nradbase +1)[1:]:
			f = self.g(i)
			gs[i] = f
			# for c_nlk, the n and k indices start from base 0
			#   though, in all other equations, they start from 1
			#   the c_nlk should be set up so that n=1 corresponds to crad[n=0][lind][kind]
			c_nlk = crad[n-1][l][i-1] 
			rnl = np.add(rnl, f*c_nlk)
		return rnl

	def init_R_nl(self,**kwargs):
		try:
			basis = kwargs['basis']
		except KeyError:
			# default to chebychev exponential cosine basis from drautz 2019
			# NOTE this basis is non-orthogonal!!!
			basis = self.basis
		r_nls = {n:{l: None for l in range(self.lmax +1)} for n in range(1,self.nradmax+1) }
		if basis == 'genLag':
			r_scale = self.r_arr/ self.rc
			for n in range(1,self.nradmax+1):
				for l in range(self.lmax+1):
					r_nls[n][l] = scipy.special.eval_genlaguerre(n,l,r_scale)
		elif basis == 'ChebExpCos':
			try:
				crad = kwargs['crad']
			except KeyError:
				#crad = np.zeros((self.nradmax,self.lmax+1,self.nradbase))
				crad = np.zeros((self.nradbase,self.lmax+1,self.nradbase))
				for nind in range(self.nradbase):
				#for nind in range(self.nradmax):
					for lind in range(self.lmax+1):
						crad[nind][lind] = np.array([1. if k ==nind else 0. for k in range(self.nradbase)])
			for n in range(1,self.nradmax+1):
				for l in range(self.lmax+1):
					r_nls[n][l] = self.R_nl(n,l,crad)
		self.r_nls = r_nls
		#self.basis = basis
		return None

	def set_basis(self,basis,**kwargs):
		try:
			crad = kwargs['crad']
			kwd = {'basis':basis,'crad':crad}
		except KeyError:
			kwd = {'basis':basis}
		self.basis = basis
		self.init_R_nl(**kwd)
		return None

	def r_nl(self,n,l):
		return self.r_nls[n][l]

class angular_basis:
	def __init__(self,
			r_vecs, #np.array - array of arrays of positions in cartesian
			lmax): #int-max l index in Ylm
		self.r_vecs = r_vecs
		self.r_arr = [np.linalg.norm(r) for r in self.r_vecs]
		self.r_hats = get_spher_pol(self.r_vecs,self.r_arr)
		self.lmax = lmax
		self.Ylm()
		return None

	def Ylm(self):
		lm_matrix = { l: {m:None for m in range(-l,l+1)} for l in range(self.lmax+1)}
		for l in range(self.lmax+1):
			for m in range(-l,l+1):
				func = special.sph_harm(m,l, self.r_hats[0],self.r_hats[1])
				lm_matrix[l][m] = func
		self.lm_matrix = lm_matrix
		return None
		
	
	def ylm (self,l,m):
		func = self.lm_matrix[l][m]
		return func

class phi_basis:
	def __init__(self,
			radial,
			angular):
		self.rb = radial
		self.ab = angular
		#self.prefac = 1.
		#prefactor for phi basis
		self.prefac = 1/np.sqrt(4*np.pi)
		return None
	def phi(self,n,l,m,r_ind):
		#lpm = scipy.special.factorial(l+m,exact=True)
		#lsm = scipy.special.factorial(l-m,exact=True)
		#p2 = np.sqrt(1/(2*l +1)) * np.sqrt( lpm /lsm  ) 
		phi = self.prefac*self.ab.ylm(l,m)*self.rb.r_nl(n,l)
		return phi[r_ind]
	def phi_1(self,n,r_ind):
		#NOTE energies only match with those in lammps if the rank1 prefactors are 1.
		if self.rb.basis == 'genLag':
			phi = self.rb.r_nl(n,0)
		elif self.rb.basis == 'ChebExpCos':
			phi = self.rb.g(n)
		return phi[r_ind]
class A_basis:
	def __init__(self,
			radial,
			angular):
		self.rb = radial
		self.ab = angular
		#self.prefac = 1.
		#prefactor for phi basis
		self.prefac = 1/np.sqrt(4*np.pi)
		return None
	def A(self,n,l,m):
		#lpm = scipy.special.factorial(l+m,exact=True)
		#lsm = scipy.special.factorial(l-m,exact=True)
		#p2 = np.sqrt(1/(2*l +1)) * np.sqrt( lpm /lsm  ) 
		phi = self.prefac*self.ab.ylm(l,m)*self.rb.r_nl(n,l)
		return np.sum(phi)
	def A_1(self,n):
		#NOTE energies only match with those in lammps if the rank1 prefactors are 1.
		if self.rb.basis == 'genLag':
			phi = self.rb.r_nl(n,0)
		elif self.rb.basis == 'ChebExpCos':
			phi = self.rb.g(n)
		return np.sum(phi)
