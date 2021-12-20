from .sections import Section
from itertools import combinations_with_replacement
import numpy as np
from ...parallel_tools import pt
from ...Wigner_ACE.gen_labels import *
from ...Wigner_ACE.write_analytical_coupling import *

class Ace(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        #copy allowed keys section & write file to disk under new io/sections/ace 
        allowedkeys = ['numTypes','ranks','lmax','nmax','nmaxbase','rcutfac','lambda','type','bzeroflag','wigner_flag']
        for value_name in config['ACE']:
            if value_name in allowedkeys: continue
            else:
                raise RuntimeError(">>> Found unmatched variable in ACE section of input: ",value_name)
                #pt.single_print(">>> Found unmatched variable in ACE section of input: ",value_name)
        self.lmbda = self.get_value("ACE","lambda",'5.0',"float")
        self.numtypes = self.get_value("ACE", "numTypes", "1", "int")
        self.ranks = self.get_value("ACE","ranks","3").split()
        self.lmax = self.get_value("ACE", "lmax", "2").split()
        self.nmax = self.get_value("ACE", "nmax", "2").split() 
        self.nmaxbase = self.get_value("ACE", "nmaxbase", "16","int")
        self.rcutfac = self.get_value("ACE", "rcutfac", "7.5", "float")
        self.types = self.get_value("ACE", "type", "H").split()
#        self.wj = []
#        self.radelem = []
#        self.types = []
        self.type_mapping = {}
#        for i in range(self.numtypes):
#            self.wj.append(self.get_value("ACE", "wj{}".format(i + 1), "1.0", "float"))
#        for i in range(self.numtypes):
#            self.radelem.append(self.get_value("ACE", "radelem{}".format(i + 1), "0.5", "float"))
#        for i in range(self.numtypes):
#            self.types.append(self.get_value("ACE", "type{}".format(i + 1), "H"))
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1

        self.bzeroflag = self.get_value("ACE", "bzeroflag", "0", "bool")
        self.wigner_flag = self.get_value("ACE", "wigner_flag", "1", "bool")

        self._generate_b_list()
        self._write_couple()
        #self._reset_chemflag()
        self.delete()

    def _generate_b_list(self):
        self.blist = []
        self.blank2J = []
        prefac = 1.0
        i = 0
        for atype in range(self.numtypes):
            ranked_nus = [generate_nl(int(rnk),int(self.nmax[ind]),int(self.lmax[ind])) for ind,rnk in enumerate(self.ranks)]
            nus = [item for sublist in ranked_nus for item in sublist]
            for nu in nus:
                i += 1
                self.blist.append([i] + [int(k) for k in nu.split(',')])
                self.blank2J.append([prefac])
        self.ncoeff = int(len(self.blist))
        if not self.bzeroflag:
            self.blank2J = np.reshape(self.blank2J, (self.numtypes, int(len(self.blist)/self.numtypes)))
            onehot_atoms = np.ones((self.numtypes, 1))
            self.blank2J = np.concatenate((onehot_atoms, self.blank2J), axis=1)
            self.blank2J = np.reshape(self.blank2J, (len(self.blist) + self.numtypes))
        else:
            self.blank2J = np.reshape(self.blank2J, len(self.blist))

    def _write_couple(self):
        ranked_nus = [generate_nl(int(rnk),int(self.nmax[ind]),int(self.lmax[ind])) for ind,rnk in enumerate(self.ranks)]
        nus = [item for sublist in ranked_nus for item in sublist]
        if True:#not os.path.isfile('coupling_coefficients.ace'):
            if not self.wigner_flag:
                coupling,weights = get_coupling(nus,[int(rnk) for rnk in self.ranks])
            elif self.wigner_flag:
                coupling,weights = get_coupling(nus,[int(rnk) for rnk in self.ranks])
                #coupling,weights = get_coupling(nus,[int(rnk) for rnk in self.ranks],**{'wigner_flag':True})
            #initialize an array of dummy ctilde coefficients (must be initialized to 1 for potential fitting)
            coeff_arr = np.ones(len(nus)+1)
            #write a coupling coefficient file that is compatible with py_ACE
            import json
            with open('ccs.json','w') as writejson:
                json.dump(coupling,writejson,indent=2)
            coeffs = {nu:coeff for nu,coeff in zip(nus,coeff_arr[1:])}
            #TODO figure out where E0 comes from in Fitsnap
            l_lst = [int(li) for li in self.lmax]
            n_lst = [int(ni) for ni in self.nmax]
            rnk_lst = [int(rnk) for rnk in self.ranks]
            write_pot('coupling_coefficients',self.types[0],rnk_lst,lmax=max(l_lst),nradbase=self.nmaxbase,nradmax=max(n_lst), rcut=self.rcutfac, exp_lambda=self.lmbda, nus=nus,coupling=coupling,coeffs=coeffs,E_0=0.)
        #else:
        #    print ('USING EXISTING coupling_coefficients.ace -this file will need to be deleted if potential parameters are changed in the FitSNAP infile!')
