import numpy as np
import itertools
#from fitsnap3lib.lib.sym_ACE.rpi_lib import *
#from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import  *
from fitsnap3lib.io.sections.sections import Section

try:

    from fitsnap3lib.lib.sym_ACE.rpi_lib import *
    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import  *

    class Ace(Section):

        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            
            allowedkeys = ['numTypes', 'ranks', 'lmax', 'nmax', 'mumax', 'nmaxbase', 'rcutfac', 'lambda', 
                          'type', 'bzeroflag', 'erefs', 'rcinner', 'drcinner', 'RPI_heuristic', 'lmin', 
                          'bikflag', 'dgradflag']
            for value_name in config['ACE']:
                if value_name in allowedkeys: continue
                else:
                    raise RuntimeError(">>> Found unmatched variable in ACE section of input: ",value_name)
            self.numtypes = self.get_value("ACE", "numTypes", "1", "int")
            self.ranks = self.get_value("ACE","ranks","3").split()
            self.lmin = self.get_value("ACE", "lmin", "0").split() 
            self.lmax = self.get_value("ACE", "lmax", "2").split()
            self.nmax = self.get_value("ACE", "nmax", "2").split() 
            self.mumax = self.get_value("ACE","mumax", "1")
            self.nmaxbase = self.get_value("ACE", "nmaxbase", "16","int")
            self.rcutfac = self.get_value("ACE", "rcutfac", "4.5").split()
            self.lmbda = self.get_value("ACE","lambda",'1.35').split()
            self.rcinner = self.get_value("ACE","rcinner",'0.0').split()
            self.drcinner = self.get_value("ACE","drcinner",'0.01').split()
            self.types = self.get_value("ACE", "type", "H").split()
            self.erefs = self.get_value("ACE", "erefs", "0.0").split() 
            self.bikflag = self.get_value("ACE", "bikflag", "0", "bool")
            self.dgradflag = self.get_value("ACE", "dgradflag", "0", "bool")
            self.RPI_heuristic = self.get_value("ACE" , "RPI_heuristic" , 'root_SO3_span')
            self.type_mapping = {}
            for i, atom_type in enumerate(self.types):
                self.type_mapping[atom_type] = i+1

            self.bzeroflag = self.get_value("ACE", "bzeroflag", "0", "bool")
            self.wigner_flag = self.get_value("ACE", "wigner_flag", "1", "bool")

            #if self.bikflag:
            #    self._assert_dependency('bikflag', "CALCULATOR", "per_atom_energy", True)
            self.lmax_dct = {int(rnk):int(lmx) for rnk,lmx in zip(self.ranks,self.lmax)}
            if self.RPI_heuristic != 'root_SO3_span':
                self.pt.single_print('WARNING: do not change RPI flags unless you know what you are doing!')
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
                if type(self.lmin) == list:
                    if len(self.lmin) == 1:
                        self.lmin = self.lmin * len(self.ranks)
                    ranked_chem_nus = [descriptor_labels_YSG(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), int(self.mumax),lmin = int(self.lmin[ind]) ) for ind,rnk in enumerate(self.ranks)]
                else:
                    ranked_chem_nus = [descriptor_labels_YSG(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), int(self.mumax),lmin = int(self.lmin) ) for ind,rnk in enumerate(self.ranks)]

            highranks = [int(r) for r in self.ranks if int(r) >= 4]
            warnflag = any([ self.lmax_dct[rank] >= 3 for ind,rank in enumerate(highranks)])
            if warnflag:
                self.pt.single_print('WARNING: lmax of %d will generate descriptors that cannot be entered into LAMMPS_PACE - try a lower lmax for ranks >= 4' % warnflag[0])
            nus_unsort = [item for sublist in ranked_chem_nus for item in sublist]
            nus = nus_unsort.copy()
            mu0s = []
            mus =[]
            ns = []
            ls = []
            for nu in nus_unsort:
                mu0ii,muii,nii,lii = get_mu_n_l(nu)
                mu0s.append(mu0ii)
                mus.append(tuple(muii))
                ns.append(tuple(nii))
                ls.append(tuple(lii))
            nus.sort(key = lambda x : mus[nus_unsort.index(x)],reverse = False)
            nus.sort(key = lambda x : ns[nus_unsort.index(x)],reverse = False)
            nus.sort(key = lambda x : ls[nus_unsort.index(x)],reverse = False)
            nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
            nus.sort(key = lambda x : len(x),reverse = False)
            nus.sort(key = lambda x : mu0s[nus_unsort.index(x)],reverse = False)
            byattyp = srt_by_attyp(nus)

            for atype in range(self.numtypes):
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
            bondinds=range(len(self.types))
            bonds = [b for b in itertools.product(bondinds,bondinds)]
            bondstrs = ['[%d, %d]' % b for b in bonds]
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
                    

            apot = AcePot(self.types, reference_ens, [int(k) for k in self.ranks], [int(k) for k in self.nmax],  [int(k) for k in self.lmax], self.nmaxbase, rcvals, lmbdavals, rcinnervals, drcinnervals, [int(k) for k in self.lmin], self.RPI_heuristic)
            apot.write_pot('coupling_coefficients')

except ModuleNotFoundError:

    class Ace(Section):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            raise ModuleNotFoundError("Missing sympy or pyyaml modules.")
