from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
from fitsnap3lib.lib.neural_networks.descriptors.bessel import Bessel
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

        self.bessel = Bessel(self.config.sections['CUSTOM'].num_descriptors, self.config.sections['CUSTOM'].cutoff) # for calculating Bessel descriptors

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
        for line in self.config.sections["REFERENCE"].lmp_pairdecl:
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
        pass

    def calculate_descriptors(self,x,neighlist,xneigh):
        """
        Calculate descriptors for network standardization (need mean and std)
        This is used in _collect_lammps_nonlinear to calculate pairwise descriptors for a single 
        config.
        """

        basis = self.bessel.numpy_radial_bessel_basis(x, neighlist, xneigh)

        return basis


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
        self.pt.shared_arrays['a'].array[index:index+num_atoms,2:] = lmp_pos #self._data["Positions"]
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
        # calculate descriptors for standardization
        descriptors = self.calculate_descriptors(self._data["Positions"], neighlist, xneighs)
        self.pt.shared_arrays['descriptors'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = descriptors
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
        tags = self._lmp.extract_atom('id')
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
