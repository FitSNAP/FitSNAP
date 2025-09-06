import numpy as np
import itertools
import pickle
import json
#from fitsnap3lib.lib.sym_ACE.rpi_lib import *
#from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import  *
from fitsnap3lib.io.sections.sections import Section

try:

    from fitsnap3lib.lib.sym_ACE.pa_gen import *
    from fitsnap3lib.lib.sym_ACE.yamlpace_tools.potential import  *
    from fitsnap3lib.lib.sym_ACE.wigner_couple import *
    from fitsnap3lib.lib.sym_ACE.clebsch_couple import *

    class Ace(Section):

        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            
            allowedkeys = ['numTypes', 'ranks', 'lmax', 'nmax', 'mumax', 'nmaxbase', 'rcutfac', 'lambda', 
                          'type', 'bzeroflag', 'erefs', 'rcinner', 'drcinner', 'RPI_heuristic', 'lmin', 
                          'bikflag', 'dgradflag','wigner_flag','b_basis','manuallabs']
            for value_name in config['ACE']:
                if value_name in allowedkeys: continue
                else:
                    raise RuntimeError(">>> Found unmatched variable in ACE section of input: ",value_name)
            self.numtypes = self.get_value("ACE", "numTypes", "1", "int")
            self.ranks = self.get_value("ACE","ranks","3").split()
            self.lmin = self.get_value("ACE", "lmin", "0").split() 
            self.lmax = self.get_value("ACE", "lmax", "2").split()
            self.nmax = self.get_value("ACE", "nmax", "2").split() 
            #self.mumax = self.get_value("ACE","mumax", "1")
            self.nmaxbase = self.get_value("ACE", "nmaxbase", "16","int")
            self.rcutfac = self.get_value("ACE", "rcutfac", "4.5").split()
            self.lmbda = self.get_value("ACE","lambda",'1.35').split()
            self.rcinner = self.get_value("ACE","rcinner",'0.0').split()
            self.drcinner = self.get_value("ACE","drcinner",'0.01').split()
            self.types = self.get_value("ACE", "type", "H").split()
            self.mumax = len(self.types)
            #self.erefs = self.get_value("ACE", "erefs", "0.0").split() 
            self.erefs = [0.0] * len(self.types)
            self.bikflag = self.get_value("ACE", "bikflag", "0", "bool")
            self.dgradflag = self.get_value("ACE", "dgradflag", "0", "bool")
            self.b_basis = self.get_value("ACE" , "b_basis" , "pa_tabulated") 
            self.manuallabs = self.get_value("ACE", "manuallabs", 'None')
            self.type_mapping = {}
            for i, atom_type in enumerate(self.types):
                self.type_mapping[atom_type] = i+1

            self.bzeroflag = self.get_value("ACE", "bzeroflag", "0", "bool")
            self.wigner_flag = self.get_value("ACE", "wigner_flag", "1", "bool")

            #if self.bikflag:
            #    self._assert_dependency('bikflag', "CALCULATOR", "per_atom_energy", True)
            self.lmax_dct = {int(rnk):int(lmx) for rnk,lmx in zip(self.ranks,self.lmax)}
            if self.b_basis != 'pa_tabulated':
                self.pt.single_print('WARNING: Only change ACE basis flags if you know what you are doing!')
            self._generate_b_list()

            # Print detailed statistics about ACE basis and design matrix
            # self._print_ace_statistics()

            self._write_couple()
            Section.num_desc = len(self.blist)
                        
            self.delete()
        
        def _print_ace_statistics(self):
            """Print detailed statistics about the ACE basis and design matrix."""
            @self.pt.rank_zero
            def print_stats():
                print("\n" + "="*80, flush=True)
                print("ACE BASIS AND DESIGN MATRIX STATISTICS", flush=True)
                print("="*80, flush=True)
                
                # Basic configuration
                print("\n[ACE Configuration]", flush=True)
                print(f"  Number of atom types: {self.numtypes}", flush=True)
                print(f"  Atom types: {' '.join(self.types)}", flush=True)
                print(f"  Ranks: {' '.join(self.ranks)}", flush=True)
                print(f"  lmin values: {' '.join(str(l) for l in self.lmin)}", flush=True)
                print(f"  lmax values: {' '.join(str(l) for l in self.lmax)}", flush=True)
                print(f"  nmax values: {' '.join(str(n) for n in self.nmax)}", flush=True)
                print(f"  nmaxbase: {self.nmaxbase}", flush=True)
                print(f"  Basis type: {self.b_basis}", flush=True)
                print(f"  Include B0 (bzeroflag): {self.bzeroflag}", flush=True)
                print(f"  Wigner coupling: {self.wigner_flag}", flush=True)
                print(f"  Per-atom basis (bikflag): {self.bikflag}", flush=True)
                print(f"  Compute gradients (dgradflag): {self.dgradflag}", flush=True)
                
                # Radial parameters
                print("\n[Radial Parameters]", flush=True)
                print(f"  rcutfac: {' '.join(str(r) for r in self.rcutfac)}", flush=True)
                print(f"  lambda: {' '.join(str(l) for l in self.lmbda)}", flush=True)
                print(f"  rcinner: {' '.join(str(r) for r in self.rcinner)}", flush=True)
                print(f"  drcinner: {' '.join(str(d) for d in self.drcinner)}", flush=True)
                
                # Basis statistics
                print("\n[Basis Function Statistics]", flush=True)
                print(f"  Total number of basis functions: {len(self.blist)}", flush=True)
                print(f"  Number of basis functions per atom type: {self.ncoeff}", flush=True)
                
                # Count basis functions by rank
                rank_counts = {}
                for nu in self.nus:
                    rank = len(nu)
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1
                
                print("\n  Basis functions by rank:", flush=True)
                for rank in sorted(rank_counts.keys()):
                    count = rank_counts[rank]
                    print(f"    Rank {rank}: {count} functions ({count/len(self.nus)*100:.1f}%)", flush=True)
                
                # If not using bzeroflag, we have additional offset parameters
                total_params = len(self.blist)
                if not self.bzeroflag:
                    total_params += self.numtypes
                    print(f"\n  Additional offset parameters (B0): {self.numtypes}", flush=True)
                
                print(f"\n  Total number of parameters to fit: {total_params}", flush=True)
                
                # Memory estimates (rough)
                print("\n[Memory Estimates]", flush=True)
                # Assuming double precision (8 bytes) for design matrix
                bytes_per_double = 8
                
                # For a rough estimate, assume 1000 configs with 100 atoms each
                # This is just an estimate - actual will depend on data
                estimated_configs = 1000
                estimated_atoms_per_config = 100
                
                if self.bikflag:  # Per-atom energy
                    estimated_rows = estimated_configs * estimated_atoms_per_config
                else:  # Per-config energy
                    estimated_rows = estimated_configs
                
                if self.dgradflag:  # Include forces
                    estimated_rows += estimated_configs * estimated_atoms_per_config * 3
                
                design_matrix_size = estimated_rows * total_params * bytes_per_double
                print(f"  Estimated design matrix size (for ~{estimated_configs} configs, ~{estimated_atoms_per_config} atoms/config):", flush=True)
                print(f"    Rows: ~{estimated_rows:,}", flush=True)
                print(f"    Columns: {total_params:,}", flush=True)
                print(f"    Memory: ~{design_matrix_size / (1024**3):.2f} GB", flush=True)
                
                # Distribution across atom types
                if self.numtypes > 1:
                    print("\n[Basis Distribution by Atom Type]", flush=True)
                    # Since basis functions are organized by atom type in blocks of self.ncoeff
                    for atype in range(self.numtypes):
                        start_idx = atype * self.ncoeff
                        end_idx = (atype + 1) * self.ncoeff
                        type_funcs = end_idx - start_idx
                        print(f"  Type {atype} ({self.types[atype]}): {type_funcs} functions", flush=True)
                
                print("\n" + "="*80, flush=True)
                print(flush=True)
            
            print_stats()

        def _generate_b_list(self):
            self.blist = []
            self.nus = []
            self.blank2J = []
            prefac = 1.0
            i = 0

            if self.manuallabs != 'None':
                with open(self.manuallabs,'r') as readjson:
                    labdata = json.load(readjson)
                ranked_chem_nus = [list(ik) for ik in list(labdata.values())]
            elif self.manuallabs == 'None' and self.b_basis == 'minsub':
                from fitsnap3lib.lib.sym_ACE.rpi_lib import descriptor_labels_YSG
                if type(self.lmin) == list:
                    if len(self.lmin) == 1:
                        self.lmin = self.lmin * len(self.ranks)
                    ranked_chem_nus = [descriptor_labels_YSG(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), int(self.mumax),lmin = int(self.lmin[ind]) ) for ind,rnk in enumerate(self.ranks)]
                else:
                    ranked_chem_nus = [descriptor_labels_YSG(int(rnk), int(self.nmax[ind]), int(self.lmax[ind]), int(self.mumax),lmin = int(self.lmin) ) for ind,rnk in enumerate(self.ranks)]
            elif self.manuallabs == 'None' and self.b_basis == 'pa_tabulated':
                ranked_chem_nus = []
                if len(self.lmin) == 1:
                    self.lmin = self.lmin * len(self.ranks)
                for ind,rank in enumerate(self.ranks):
                    rank = int(rank)
                    PA_lammps, not_compat = pa_labels_raw(rank,int(self.nmax[ind]),int(self.lmax[ind]), int(self.mumax),lmin = int(self.lmin[ind]) )
                    ranked_chem_nus.append(PA_lammps)
                    if len(not_compat) > 0:
                        self.pt.single_print('Functions incompatible with lammps for rank %d : '% rank, not_compat)
            highranks = [int(r) for r in self.ranks if int(r) >= 5]
            warnflag = any([ self.lmax_dct[rank] >= 5 and self.lmin[ind] > 1 for ind,rank in enumerate(highranks)])
            if warnflag:
                self.pt.single_print('WARNING: lmax and lmin for your current max rank will generate descriptors that cannot be entered into LAMMPS_PACE - try a lower lmax for ranks >= 4' % warnflag[0])
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
            #config.nus = [item for sublist in list(byattyp.values()) for item in sublist]
            for atype in range(self.numtypes):
                nus = byattyp[str(atype)]
                for nu in nus:
                    i += 1
                    mu0,mu,n,l,L = get_mu_n_l(nu,return_L=True)
                    if L != None:
                        flat_nu = [mu0] + mu + n + l + list(L)
                    else:
                        flat_nu = [mu0] + mu + n + l
                    self.blist.append([i] + flat_nu)
                    self.nus.append(nu)
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
            # Only global rank 0 handles all file I/O since nodes share scratch directory
            @self.pt.rank_zero
            def decorated_write_couple():
                if self.bzeroflag:
                    assert len(self.types) ==  len(self.erefs), "must provide reference energy for each atom type"
                    reference_ens = [float(e0) for e0 in self.erefs]
                elif not self.bzeroflag:
                    reference_ens = [0.0] * len(self.types)
                bondinds=range(len(self.types))
                bonds = [b for b in itertools.product(bondinds,bondinds)]
                bondstrs = ['[%d, %d]' % b for b in bonds]
                #print(f"*** len(self.rcutfac) {len(self.rcutfac)} len(bondstrs) {len(bondstrs)} bondstrs {bondstrs}\n", flush=True);
                assert len(self.rcutfac) == len(bondstrs), "must provide rc (radial cutoff) for each BOND type"
                assert len(self.lmbda) == len(bondstrs), "must provide lambda (radial decay parameter) for each BOND type" 
                assert len(self.rcinner) == len(bondstrs), "must provide rcinner for each BOND type" 
                assert len(self.drcinner) == len(bondstrs), "must provide drcinner for each BOND type" 
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

                ldict = {int(rank):int(lmax) for rank,lmax in zip(self.ranks,self.lmax)}
                L_R = 0 
                M_R = 0
                rankstrlst = ['%s']*len(self.ranks)
                rankstr = ''.join(rankstrlst) % tuple(self.ranks)
                lstrlst = ['%s']*len(self.ranks)
                lstr = ''.join(lstrlst) % tuple(self.lmax)
                
                # Load or create pickle files
                if not self.wigner_flag:
                    try:
                        with open('cg_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                            ccs = pickle.load(handle)
                            #self.pt.single_print(f"Loaded existing CG coupling coefficients from pickle")
                    except FileNotFoundError:
                        #self.pt.single_print(f"Creating CG coupling coefficients...")
                        ccs = get_cg_coupling(ldict,L_R=L_R)
                        store_generalized(ccs, coupling_type='cg',L_R=L_R)
                else:
                    try:
                        with open('wig_LR_%d_r%s_lmax%s.pickle' %(L_R,rankstr,lstr),'rb') as handle:
                            ccs = pickle.load(handle)
                            #self.pt.single_print(f"Loaded existing Wigner coupling coefficients from pickle")
                    except FileNotFoundError:
                        #self.pt.single_print(f"Creating Wigner coupling coefficients...")
                        ccs = get_wig_coupling(ldict,L_R)
                        store_generalized(ccs, coupling_type='wig',L_R=L_R)
                
                # Create AcePot and write the potential file
                apot = AcePot(self.types, reference_ens, [int(k) for k in self.ranks], [int(k) for k in self.nmax],  [int(k) for k in self.lmax], self.nmaxbase, rcvals, lmbdavals, rcinnervals, drcinnervals, [int(k) for k in self.lmin], self.b_basis, **{'ccs':ccs[M_R]})
                apot.write_pot('coupling_coefficients')
            
            decorated_write_couple()
            # Wait for global rank 0 to finish all file I/O
            self.pt.all_barrier()

except ModuleNotFoundError:

    class Ace(Section):
        """
        Dummy class for factory to read if torch is not available for import.
        """
        def __init__(self, name, config, pt, infile, args):
            super().__init__(name, config, pt, infile, args)
            raise ModuleNotFoundError("Missing sympy or pyyaml modules.")
