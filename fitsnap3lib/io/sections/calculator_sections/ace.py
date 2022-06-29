import numpy as np
import itertools
from rpi_lib import *
from yamlpace_tools.potential import  *

from fitsnap3lib.io.sections.sections import Section
from fitsnap3lib.parallel_tools import ParallelTools

pt = ParallelTools()

class Ace(Section):

    def __init__(self, name, config, args):
        raise Exception("ACE calculator not working yet.")
        super().__init__(name, config, args)
        
        allowedkeys = ['numTypes', 'ranks', 'lmax', 'nmax', 'mumax', 'nmaxbase', 'rcutfac', 'lambda', 'type', 'bzeroflag', 'erefs', 'rcinner','drcinner','RPI_heuristic']
        for value_name in config['ACE']:
            if value_name in allowedkeys: continue
            else:
                raise RuntimeError(">>> Found unmatched variable in ACE section of input: ",value_name)
        self.numtypes = self.get_value("ACE", "numTypes", "1", "int")
        self.ranks = self.get_value("ACE","ranks","3").split()
        self.lmax = self.get_value("ACE", "lmax", "2").split()
        self.nmax = self.get_value("ACE", "nmax", "2").split() 
        self.mumax = self.get_value("ACE","mumax", "1")
        self.nmaxbase = self.get_value("ACE", "nmaxbase", "16","int")
        self.rcutfac = self.get_value("ACE", "rcutfac", "7.5").split()
        self.lmbda = self.get_value("ACE","lambda",'5.0').split()
        self.rcinner = self.get_value("ACE","rcinner",'0.0').split()
        self.drcinner = self.get_value("ACE","drcinner",'0.01').split()
        self.types = self.get_value("ACE", "type", "H").split()
        self.erefs = self.get_value("ACE", "erefs", "0.0").split() 
        self.bikflag = self.get_value("ACE", "bikflag", "0", "bool")
        self.RPI_heuristic = self.get_value("ACE" , "RPI_heuristic" , 'root_SO3_span')
        self.type_mapping = {}
        for i, atom_type in enumerate(self.types):
            self.type_mapping[atom_type] = i+1

        self.bzeroflag = self.get_value("ACE", "bzeroflag", "0", "bool")
        self.wigner_flag = self.get_value("ACE", "wigner_flag", "1", "bool")

        if self.bikflag:
            self._assert_dependency('bikflag', "CALCULATOR", "per_atom_energy", True)

        self._generate_b_list()
        self._write_couple()
        Section.num_desc = len(self.blist)
        self.delete()

    def _generate_b_list(self):
        self.blist = []
        self.blank2J = []
        prefac = 1.0
        i = 0

        if self.RPI_heuristic == 'lexicographic':
            ranked_chem_nus = [generate_nl(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), int(self.mumax)) for ind,rnk in enumerate(self.ranks)]
        else:
            ranked_chem_nus = [descriptor_labels_YSG(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), int(self.mumax) ) for ind,rnk in enumerate(self.ranks)]
        flatnus = [item for sublist in ranked_chem_nus for item in sublist]
        pt.single_print("Total ACE descriptors",len(flatnus))
        pt.single_print("ACE descriptors",flatnus)
        byattyp = srt_by_attyp(flatnus)

        for atype in range(self.numtypes):
            #self.blist.append( [atype] + [0] )
            nus = byattyp[str(atype)]
            for nu in nus:
                i += 1
                mu0,mu,n,l,L = get_mu_n_l(nu,return_L=True)
                if L != None:
                    flat_nu = [mu0] + mu + n + l + L
                else:
                    flat_nu = [mu0] + mu + n + l
                self.blist.append([i] + flat_nu)
                self.blank2J.append([prefac])
        self.ncoeff = int(len(self.blist)/self.numtypes)
        if not self.bzeroflag:
            self.blank2J = np.reshape(self.blank2J, (self.numtypes, int(len(self.blist)/self.numtypes)))
            #self.blank2J = np.reshape(self.blank2J, (self.numtypes, self.ncoeff)))
            onehot_atoms = np.ones((self.numtypes, 1))
            self.blank2J = np.concatenate((onehot_atoms, self.blank2J), axis=1)
            self.blank2J = np.reshape(self.blank2J, (len(self.blist) + self.numtypes))
        else:
            self.blank2J = np.reshape(self.blank2J, len(self.blist))

    def _write_couple(self):
        if self.bzeroflag:
            assert len(self.types) ==  len(self.erefs), "must provide reference energy for each atom type"
            reference_ens = [float(e0) for e0 in self.erefs]
        elif not self.bzeroflag:
            reference_ens = [0.0] * len(self.types)
        if not os.path.isfile('coupling_coefficients.yace'):
            bondinds=range(len(self.types))
            bonds = [b for b in itertools.product(bondinds,bondinds)]
            bondstrs = ['[%d, %d]' % b for b in bonds]
            pt.single_print("Bonds",bondstrs)
            assert len(self.lmbda) == len(bondstrs), "must provide rc, lambda, for each BOND type" 
            assert len(self.rcutfac) == len(bondstrs), "must provide rc, lambda, for each BOND type" 
            if len(self.lmbda) == 1:
                lmbdavals = self.lmbda
                rcvals = self.rcutfac
                rcinnervals = self.rcinner
                drcinnervals = self.drcinner
            if len(self.lmbda) > 1:
                lmbdavals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.lmbda)}
                rcvals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.rcutfac)}
                rcinnervals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.rcinner)}
                drcinnervals = {bondstr:lmb for bondstr,lmb in zip(bondstrs,self.drcinner)}
                

            apot = AcePot(self.types, reference_ens, [int(k) for k in self.ranks], [int(k) for k in self.nmax],  [int(k) for k in self.lmax], self.nmaxbase, rcvals, lmbdavals, rcinnervals, drcinnervals, self.RPI_heuristic)
            apot.write_pot('coupling_coefficients')

        else:
            pt.single_print('USING EXISTING coupling_coefficients.yace -this file will need to be deleted if potential parameters are changed in the FitSNAP infile!')
