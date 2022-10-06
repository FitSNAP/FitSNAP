from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
import numpy as np


#config = Config()
#pt = ParallelTools()


class LammpsCustom(LammpsBase):

    def __init__(self, name):
        super().__init__(name)
        self.pt = ParallelTools()
        self.config = Config()
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

    def get_width(self):
        """
        Get width of A matrix, which is different for each Calculator
        """

        if (self.config.sections["SOLVER"].solver == "NETWORK"):
            a_width = 2 # 2 columns for i and j in the sparse neighlist
        else:
            raise NotImplementedError("Only NETWORK solver is implemented with Custom calculator.")

        return a_width
    
    def _prepare_lammps(self):
        self._set_structure()
        # this is super clean when there is only one value per key, needs reworking
        #        self._set_variables(**_lammps_variables(config.sections["BISPECTRUM"].__dict__))

        # needs reworking when lammps will accept variable 2J
        #self._lmp.command(f"variable twojmax equal {max(self.config.sections['BISPECTRUM'].twojmax)}")
        #self._lmp.command(f"variable rcutfac equal {self.config.sections['BISPECTRUM'].rcutfac}")
        #self._lmp.command(f"variable rfac0 equal {self.config.sections['BISPECTRUM'].rfac0}")
        #        self._lmp.command(f"variable rmin0 equal {config.sections['BISPECTRUM'].rmin0}")

        """
        for i, j in enumerate(self.config.sections["BISPECTRUM"].wj):
            self._lmp.command(f"variable wj{i + 1} equal {j}")

        for i, j in enumerate(self.config.sections["BISPECTRUM"].radelem):
            self._lmp.command(f"variable radelem{i + 1} equal {j}")
        """
        for line in self.config.sections["REFERENCE"].lmp_pairdecl:
            #print(line.lower())
            self._lmp.command(line.lower())
        

        self._set_computes()
        self._set_neighbor_list()

    def _set_box(self):
        command = "atom_modify map array sort 1 1000.0\n"
        self._lmp.command(command)
        self._set_box_helper(numtypes=self.config.sections['CUSTOM'].numtypes)

    def _create_atoms(self):
        self._create_atoms_helper(type_mapping=self.config.sections["CUSTOM"].type_mapping)

    def _set_computes(self):
        numtypes = self.config.sections['CUSTOM'].numtypes
        #command = "atom_modify map array sort 1 1000.0\n"
        #self._lmp.command(command)
        """
        radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
        wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

        kw_options = {
            k: self.config.sections["BISPECTRUM"].__dict__[v]
            for k, v in
            {
                "rmin0": "rmin0",
                "bzeroflag": "bzeroflag",
                "quadraticflag": "quadraticflag",
                "switchflag": "switchflag",
                "chem": "chemflag",
                "bnormflag": "bnormflag",
                "wselfallflag": "wselfallflag",
                "bikflag": "bikflag",
                "switchinnerflag": "switchinnerflag",
                "switchflag": "switchflag",
                "sinner": "sinner",
                "dinner": "dinner",
                "dgradflag": "dgradflag",
            }.items()
            if v in self.config.sections["BISPECTRUM"].__dict__
        }
        """

        # remove input dictionary keywords if they are not used, to avoid version problems
        """
        if kw_options["chem"] == 0:
            kw_options.pop("chem")
        if kw_options["bikflag"] == 0:
            kw_options.pop("bikflag")
        if kw_options["switchinnerflag"] == 0:
            kw_options.pop("switchinnerflag")
        if kw_options["dgradflag"] == 0:
            kw_options.pop("dgradflag")
        kw_options["rmin0"] = self.config.sections["BISPECTRUM"].rmin0
        kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
        kwargs = " ".join(kw_substrings)

        # everything is handled by LAMMPS compute snap

        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)
        """

    def _collect_lammps_nonlinear(self):

        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['CUSTOM'].numtypes
        #n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        ptr_pos = self._lmp.extract_atom('x')
        #nlocal = self._lmp.extract_global("nlocal")
        #print(f"nlocal: {nlocal}")
        #for i in range(nlocal+5):
        #    print("(x,y,z) = (", ptr_pos[i][0], ptr_pos[i][1], ptr_pos[i][2], ")")

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # extract other quantities (no reference forces for this calculator yet!)

        ref_forces = np.zeros(3*num_atoms)
        ref_energy = 0

        # set indices for populating shared arrays

        index = self.shared_index # Index telling where to start in the shared arrays on this proc.
                                  # Currently this is an index for the 'a' array (natoms*nconfigs rows).
                                  # This is also an index for the 't' array of types (natoms*nconfigs rows).
                                  # Also made indices for:
                                  # - the 'b' array (nconfigs rows)
                                  # - the 'neighlist' array (natoms)*nneigh*nconfigs rows.
        dindex = self.distributed_index
        index_b = self.shared_index_b
        index_c = self.shared_index_c
        index_neighlist = self.shared_index_neighlist

        # look up the neighbor list
        nlidx = self._lmp.find_pair_neighlist('zero')
        nl = self._lmp.numpy.get_neighlist(nlidx)
        #print(f"nl: {nl}")
        tags = self._lmp.extract_atom('id')
        # print neighbor list contents
        number_of_neighs = 0 # number of neighs for this config
        num_neighs_per_atom = []
        neighlist = []
        xneighs = []
        for i in range(0,nl.size):
            idx, nlist  = nl.get(i)
            #print("\natom {} with ID {} has {} neighbors:".format(idx,tags[idx],nlist.size))
            num_neighs_i = 0
            if nlist.size > 0:
                for n in np.nditer(nlist):
                    num_neighs_i += 1
                    #print("  atom {} with ID {}".format(n,tags[n]))
                    #print(f" ptr_pos[{n}][0]: {ptr_pos[n][0]}")
                    neighlist.append([tags[idx], tags[n]]) #, ptr_pos[n][0], ptr_pos[n][1], ptr_pos[n][2]])
                    xneighs.append([ptr_pos[n][0], ptr_pos[n][1], ptr_pos[n][2]])

            num_neighs_per_atom.append(num_neighs_i)
            number_of_neighs += num_neighs_i

        num_neighs_per_atom = np.array(num_neighs_per_atom)
        assert(np.sum(num_neighs_per_atom) == number_of_neighs)
        neighlist = np.array(neighlist, dtype=int) - 1 # subtract 1 to get indices starting from 0
        xneighs = np.array(xneighs)

        # populate the per-atom array 'a'

        self.pt.shared_arrays['a'].array[index:index+num_atoms,0] = lmp_types
        self.pt.shared_arrays['a'].array[index:index+num_atoms,1] = num_neighs_per_atom
        self.pt.shared_arrays['a'].array[index:index+num_atoms,2:] = self._data["Positions"]
        index += num_atoms

        # populate the per-config arrays 'b' and 'w'

        self.pt.shared_arrays['b'].array[index_b] = (energy - ref_energy)/num_atoms
        self.pt.shared_arrays['w'].array[index_b,0] = self._data["eweight"]
        self.pt.shared_arrays['w'].array[index_b,1] = self._data["fweight"]
        index_b += 1

        # populate the per-atom 3-vector arrays 'c' and 'x'

        self.pt.shared_arrays['c'].array[index_c:(index_c + (3*num_atoms))] = self._data["Forces"].ravel() - ref_forces
        self.pt.shared_arrays['x'].array[index_c:(index_c + (3*num_atoms))] = self._data["Positions"].ravel()
        index_c += 3*num_atoms

        # populate the neighlist array

        nrows_neighlist = number_of_neighs
        self.pt.shared_arrays['neighlist'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = neighlist
        self.pt.shared_arrays['xneigh'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = xneighs
        #print(self.pt.shared_arrays['neighlist'].array[index_neighlist:(index_neighlist+nrows_neighlist)])
        index_neighlist += nrows_neighlist

        # populate the fitsnap dicts
        # these are distributed lists and therefore have different size per proc, but will get 
        # gathered later onto the root proc in calculator.collect_distributed_lists
        # we use fitsnap dicts for NumAtoms and NumNeighs here because they are organized differently 
        # than the corresponding shared arrays. 

        dindex = dindex+1
        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])]
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])]
        self.pt.fitsnap_dict['NumAtoms'][self.distributed_index:dindex] = ['{}'.format(self._data['NumAtoms'])]
        self.pt.fitsnap_dict['NumNeighs'][self.distributed_index:dindex] = ['{}'.format(nrows_neighlist)]
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])]

        # update indices since we are stacking data in the shared arrays

        self.shared_index = index
        self.distributed_index = dindex
        self.shared_index_b = index_b
        self.shared_index_c = index_c
        self.shared_index_neighlist = index_neighlist

    def _collect_lammps_preprocess(self):
        """
        Pre-process LAMMPS data by collecting data needed to allocate shared arrays.
        """
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['CUSTOM'].numtypes
        #n_coeff = self.config.sections['CUSTOM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # look up the neighbor list
        nlidx = self._lmp.find_pair_neighlist('zero')
        nl = self._lmp.numpy.get_neighlist(nlidx)
        #print(nl)
        tags = self._lmp.extract_atom('id')
        #print("half neighbor list with {} entries".format(nl.size))
        # print neighbor list contents
        number_of_neighs = 0
        num_neighs_per_atom = []
        for i in range(0,nl.size):
            idx, nlist  = nl.get(i)
            #print("\natom {} with ID {} has {} neighbors:".format(idx,tags[idx],nlist.size))
            num_neighs_i = 0
            if nlist.size > 0:
                for n in np.nditer(nlist):
                    num_neighs_i += 1
                    #print("  atom {} with ID {}".format(n,tags[n]))
            num_neighs_per_atom.append(num_neighs_i)
            number_of_neighs += num_neighs_i

        # check that number of atoms here is equal to number of atoms in the sliced array

        natoms_sliced = self.pt.shared_arrays['number_of_atoms'].sliced_array[self._i]
        assert(natoms_sliced==num_atoms)
        self.pt.shared_arrays['number_of_neighs_scrape'].sliced_array[self._i] = number_of_neighs

        #print(self.pt.shared_arrays['number_of_neighs_scrape'].array)

        #assert(False)
        """
        # extract SNAP data, including reference potential data

        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)

        ncols_bispectrum = n_coeff + 3
        ncols_reference = 0
        nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-1 #6
        dgrad = lmp_snap[num_atoms:(num_atoms+nrows_dgrad), 3:(n_coeff+3)]

        # strip zero dgrad components (almost equivalent to pruning neighborlist)
         
        nonzero_rows = lmp_snap[num_atoms:(num_atoms+nrows_dgrad),3:(n_coeff+3)] != 0.0
        nonzero_rows = np.any(nonzero_rows, axis=1)
        dgrad = dgrad[nonzero_rows, :]
        nrows_dgrad = np.shape(dgrad)[0]
        
        #self.dgradrows[self._i] = nrows_dgrad # no need to store this in a single proc, use a shared array instead

        # check that number of atoms here is equal to number of atoms in the sliced array

        natoms_sliced = self.pt.shared_arrays['number_of_atoms'].sliced_array[self._i]
        assert(natoms_sliced==num_atoms)
        self.pt.shared_arrays['number_of_dgrad_rows'].sliced_array[self._i] = nrows_dgrad
        """