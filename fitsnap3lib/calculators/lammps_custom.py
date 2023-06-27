from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
from fitsnap3lib.io.input import Config
from fitsnap3lib.lib.neural_networks.descriptors.bessel import Bessel
from fitsnap3lib.lib.neural_networks.descriptors.g3b import Gaussian3Body
import numpy as np


class LammpsCustom(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

        # declare objects for calculating descriptors used to standardize the network

        self.bessel = Bessel(self.config.sections['CUSTOM'].num_radial, self.config.sections['CUSTOM'].cutoff)
        self.g3b = Gaussian3Body(self.config.sections['CUSTOM'].num_3body, self.config.sections['CUSTOM'].cutoff)

    def get_width(self):
        """
        Get width of A matrix, which is different for each Calculator.
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

        Args:
            x (ndarray): Array of positions with size (num_neigh,3), these are atoms i with repeated 
                indices to match the dimensions in `neighlist`. 
            neighlist (ndarray): Neighbor list of integers with size (num_neigh, 2); first column is 
                atom i and second column is atom j. 
            xneigh (ndarray): Array of positions for each atom, indicies correspond with `neighlist`, 
                with size (num_neigh, 3).

        Returns:
            Array of descriptors for each pair with size (num_neigh, num_descriptors). The 2-body 
            and 3-body descriptors are concatenated along the columns.
        """

        diff = x[neighlist[:,0]] - xneigh
        rij = np.linalg.norm(diff, axis=1)[:,None] # need for cutoff and various other functions
        diff_norm = diff/rij # need for g3b

        # calculate cutoff functions used in radial descriptors

        cutoff_functions = self.bessel.cutoff_function(rij, numpy_bool=True)

        # calculate radial basis descriptors for each pair

        basis = self.bessel.radial_bessel_basis(rij, cutoff_functions, numpy_bool=True).numpy()

        # calculate 3 body descriptors for each pair and convert to numpy

        descriptors_3body = self.g3b.calculate(rij, diff_norm, neighlist[:,0], numpy_bool=True).numpy()

        assert(np.shape(basis)[0] == np.shape(descriptors_3body)[0])
        descriptors = np.concatenate([basis,descriptors_3body], axis=1)
        assert(np.shape(descriptors)[1] == self.config.sections['CUSTOM'].num_descriptors)

        return descriptors


    def _collect_lammps_nonlinear(self):

        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['CUSTOM'].numtypes
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        ptr_pos = self._lmp.extract_atom('x')

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")
        assert (np.all(np.round(self._data["Positions"],decimals=6)==np.round(lmp_pos,decimals=6)))

        # extract other quantities (no reference forces for this calculator yet!)

        ref_forces = np.zeros(3*num_atoms)
        ref_energy = 0

        # set indices for populating shared arrays

        """
        Index telling where to start in the shared arrays on this proc.
        Currently this is an index for the 'a' array (natoms*nconfigs rows).
        This is also an index for the 't' array of types (natoms*nconfigs rows).
        Also made indices for:
        - the 'b' array (nconfigs rows)
        - the 'neighlist' array (natoms)*nneigh*nconfigs rows.
        """
        index = self.shared_index
        dindex = self.distributed_index
        index_b = self.shared_index_b
        index_c = self.shared_index_c
        index_neighlist = self.shared_index_neighlist

        # look up the neighbor list

        nlidx = self._lmp.find_pair_neighlist('zero')
        nl = self._lmp.numpy.get_neighlist(nlidx)
        tags = self._lmp.extract_atom('id')
        number_of_neighs = 0 # number of neighs for this config
        num_neighs_per_atom = []
        neighlist = []
        xneighs = []
        transform_x = []
        for i in range(0,nl.size):
            idx, nlist  = nl.get(i)
            num_neighs_i = 0
            if nlist.size > 0:
                for n in np.nditer(nlist):
                    num_neighs_i += 1
                    neighlist.append([tags[idx], tags[n]])
                    xneighs.append([ptr_pos[n][0], ptr_pos[n][1], ptr_pos[n][2]])
                    transform_x.append([ptr_pos[n][0]-lmp_pos[tags[n]-1,0], \
                                        ptr_pos[n][1]-lmp_pos[tags[n]-1,1], \
                                        ptr_pos[n][2]-lmp_pos[tags[n]-1,2]])

            num_neighs_per_atom.append(num_neighs_i)
            number_of_neighs += num_neighs_i

        num_neighs_per_atom = np.array(num_neighs_per_atom)
        assert(np.sum(num_neighs_per_atom) == number_of_neighs)
        neighlist = np.array(neighlist, dtype=int) - 1 # subtract 1 to get indices starting from 0
        xneighs = np.array(xneighs)
        transform_x = np.array(transform_x)

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
        self.pt.shared_arrays['transform_x'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = transform_x

        # assert neighlist-transformed positions are correct

        assert(np.all(np.round(xneighs-lmp_pos[neighlist[:,1]],6) == np.round(transform_x,6)) )

        # calculate descriptors for standardization

        descriptors = self.calculate_descriptors(self._data["Positions"], neighlist, xneighs)
        self.pt.shared_arrays['descriptors'].array[index_neighlist:(index_neighlist+nrows_neighlist)] = descriptors
        index_neighlist += nrows_neighlist

        """
        Populate the fitsnap dicts.
        These are distributed lists and therefore have different size per proc, but will get 
        gathered later onto the root proc in calculator.collect_distributed_lists.
        We use fitsnap dicts for NumAtoms and NumNeighs here because they are organized differently 
        than the corresponding shared arrays. 
        """

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
        number_of_neighs = 0
        num_neighs_per_atom = []
        for i in range(0,nl.size):
            idx, nlist  = nl.get(i)
            num_neighs_i = 0
            if nlist.size > 0:
                for n in np.nditer(nlist):
                    num_neighs_i += 1
            num_neighs_per_atom.append(num_neighs_i)
            number_of_neighs += num_neighs_i

        # check that number of atoms here is equal to number of atoms in the sliced array

        natoms_sliced = self.pt.shared_arrays['number_of_atoms'].sliced_array[self._i]
        assert(natoms_sliced==num_atoms)
        self.pt.shared_arrays['number_of_neighs_scrape'].sliced_array[self._i] = number_of_neighs
