from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np, _extract_commands
from fitsnap3lib.parallel_tools import ParallelTools
import ctypes
from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
import numpy as np


config = Config()
pt = ParallelTools()


class LammpsPace(LammpsBase):

    def get_width(self):
        if (self.config.sections["SOLVER"].solver == "PYTORCH"):
            a_width = self.config.sections["ACE"].ncoeff
        else:
            num_types = config.sections["ACE"].numtypes
            a_width = config.sections["ACE"].ncoeff  * num_types
            if not config.sections["ACE"].bzeroflag:
                a_width += num_types
        return a_width

    def _prepare_lammps(self):
        self._set_structure()
        # this is super clean when there is only one value per key, needs reworking
#        self._set_variables(**_lammps_variables(config.sections["ACE"].__dict__))

        #Needs reworking when lammps will accept variable 2J
        #self._lmp.command(f"variable twojmax equal {max(config.sections['ACE'].twojmax)}")
        self._lmp.command(f"variable rcutfac equal {max(config.sections['ACE'].rcutfac)}")
        #self._lmp.command(f"variable rfac0 equal {config.sections['ACE'].rfac0}")
#        self._lmp.command(f"variable rmin0 equal {config.sections['ACE'].rmin0}")

        #for i,j in enumerate(config.sections["ACE"].wj):
        #    self._lmp.command(f"variable wj{i+1} equal {j}")

        #for i,j in enumerate(config.sections["ACE"].radelem):
        #    self._lmp.command(f"variable radelem{i+1} equal {j}")

        for line in config.sections["REFERENCE"].lmp_pairdecl:
            self._lmp.command(line.lower())

        self._set_computes()

        self._set_neighbor_list()

    def _set_box(self):
        self._set_box_helper(numtypes=config.sections['ACE'].numtypes)

    def _create_atoms(self):
        self._create_atoms_helper(type_mapping=config.sections["ACE"].type_mapping)
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),
         (ay, by, cy),
         (az, bz, cz)) = self._data["Lattice"]

        assert all(abs(c) < 1e-10 for c in (ay, az, bz)), \
            "Cell not normalized for lammps!"
        region_command = \
            f"region pybox prism 0 {ax:20.20g} 0 {by:20.20g} 0 {cz:20.20g} {bx:20.20g} {cx:20.20g} {cy:20.20g}"
        self._lmp.command(region_command)
        self._lmp.command(f"create_box {config.sections['ACE'].numtypes} pybox")
    
    """
    def _create_atoms(self):
        number_of_atoms = len(self._data["AtomTypes"])
        positions = self._data["Positions"].flatten()
        elem_all = [config.sections["ACE"].type_mapping[a_t] for a_t in self._data["AtomTypes"]]
        self._lmp.command(f"create_atoms 1 random {number_of_atoms} 12345 1")
        self._lmp.scatter_atoms("x", 1, 3, (len(positions) * ctypes.c_double)(*positions))
        self._lmp.scatter_atoms("type", 0, 1, (len(elem_all) * ctypes.c_int)(*elem_all))
        n_atoms = int(self._lmp.get_natoms())
        assert number_of_atoms == n_atoms, f"Atom counts don't match when creating atoms: {number_of_atoms}, {n_atoms}"
    """
    def _create_atoms(self):
        for i, (a_t, (a_x, a_y, a_z)) in enumerate(zip(self._data["AtomTypes"], self._data["Positions"])):
            a_t = config.sections["ACE"].type_mapping[a_t]
            self._lmp.command(f"create_atoms {a_t} single {a_x:20.20g} {a_y:20.20g} {a_z:20.20g} remap yes")
        n_atoms = int(self._lmp.get_natoms())
        assert i + 1 == n_atoms, f"Atom counts don't match when creating atoms: {i + 1}, {n_atoms}"

    def _create_spins(self):
        for i, (s_mag, s_x, s_y, s_z) in enumerate(self._data["Spins"]):
            self._lmp.command(f"set atom {i + 1} spin {s_mag:20.20g} {s_x:20.20g} {s_y:20.20g} {s_z:20.20g} ")
        n_atoms = int(self._lmp.get_natoms())
        assert i + 1 == n_atoms, f"Atom counts don't match when assigning spins: {i + 1}, {n_atoms}"

    def _create_charge(self):
        for i, q in enumerate(self._data["Charges"]):
            self._lmp.command(f"set atom {i + 1} charge {q[0]:20.20g} ")
        n_atoms = int(self._lmp.get_natoms())
        assert i + 1 == n_atoms, f"Atom counts don't match when assigning charge: {i + 1}, {n_atoms}"

    def _set_variables(self, **lmp_variable_args):
        for k, v in lmp_variable_args.items():
            self._lmp.command(f"variable {k} equal {v}")
#>>>>>>> refactored ACE code

    def _set_computes(self):
        numtypes = config.sections['ACE'].numtypes
        #radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
        #wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

        # everything is handled by LAMMPS compute pace (same format as compute snap) with same dummy variables currently

        if not config.sections['ACE'].bikflag:
            base_pace = "compute snap all pace coupling_coefficients.yace 0 0"
        elif (config.sections['ACE'].bikflag and not config.sections['ACE'].dgradflag):
            base_pace = "compute snap all pace coupling_coefficients.yace 1 0"
        elif (config.sections['ACE'].bikflag and config.sections['ACE'].dgradflag):
            base_pace = "compute snap all pace coupling_coefficients.yace 1 1"
        self._lmp.command(base_pace)

    def _collect_lammps_nonlinear(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['ACE'].numtypes
        n_coeff = self.config.sections['ACE'].ncoeff
        energy = self._data["Energy"]
        filename = self._data["File"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        #print(lmp_pos[0,0])

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # extract SNAP data, including reference potential data

        bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)
        ncols_bispectrum = n_coeff

        # number of columns in the snap array, add 3 to include indices and Cartesian components.

        ncols_snap = n_coeff + 3
        ncols_reference = 0
        nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-1
        nrows_snap = nrows_energy + nrows_dgrad + 1
        assert nrows_snap == np.shape(lmp_snap)[0]
        index = self.shared_index # Index telling where to start in the shared arrays on this proc.
                                  # Currently this is an index for the 'a' array (natoms*nconfigs rows).
                                  # This is also an index for the 't' array of types (natoms*nconfigs rows).
                                  # Also made indices for:
                                  # - the 'b' array (3*natoms+1)*nconfigs rows.
                                  # - the 'dgrad' array (natoms+1)*nneigh*3*nconfigs rows.
                                  # - the 'dgrad_indices' array which has same number of rows as 'dgrad'
        dindex = self.distributed_index
        index_b = self.shared_index_b
        index_c = self.shared_index_c
        index_dgrad = self.shared_index_dgrad

        # extract the useful parts of the snap array

        bispectrum_components = lmp_snap[0:bik_rows, 3:n_coeff+3]
        ref_forces = lmp_snap[0:bik_rows, 0:3].flatten()
        dgrad = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 3:n_coeff+3]
        dgrad_indices = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 0:3].astype(np.int32)
        ref_energy = lmp_snap[-1, 0]

        # strip zero dgrad components (equivalent to pruning neighborlist)
         
        nonzero_rows = lmp_snap[bik_rows:(bik_rows+nrows_dgrad),3:(n_coeff+3)] != 0.0
        nonzero_rows = np.any(nonzero_rows, axis=1)
        dgrad = dgrad[nonzero_rows, :]
        nrows_dgrad = np.shape(dgrad)[0]
        nrows_snap = np.shape(dgrad)[0] + nrows_energy + 1
        dgrad_indices = dgrad_indices[nonzero_rows, :]

        # populate the bispectrum array 'a'

        self.pt.shared_arrays['a'].array[index:index+bik_rows] = bispectrum_components
        self.pt.shared_arrays['t'].array[index:index+bik_rows] = lmp_types
        index += num_atoms

        # populate the truth array 'b' and weight array 'w'

        self.pt.shared_arrays['b'].array[index_b] = (energy - ref_energy)/num_atoms
        self.pt.shared_arrays['w'].array[index_b,0] = self._data["eweight"]
        self.pt.shared_arrays['w'].array[index_b,1] = self._data["fweight"]
        index_b += 1

        if (self.config.sections['CALCULATOR'].force):
            # populate the force truth array 'c'

            self.pt.shared_arrays['c'].array[index_c:(index_c + (3*num_atoms))] = self._data["Forces"].ravel() - ref_forces
            index_c += 3*num_atoms

            # populate the dgrad arrays 'dgrad' and 'dbdrindx'

            self.pt.shared_arrays['dgrad'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = dgrad
            self.pt.shared_arrays['dbdrindx'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = dgrad_indices
            index_dgrad += nrows_dgrad

        # populate the fitsnap dicts
        # these are distributed lists and therefore have different size per proc, but will get 
        # gathered later onto the root proc in calculator.collect_distributed_lists
        # we use fitsnap dicts for NumAtoms and NumDgradRows here because they are organized differently 
        # than the corresponding shared arrays. 

        dindex = dindex+1
        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])]
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])]
        self.pt.fitsnap_dict['NumAtoms'][self.distributed_index:dindex] = ['{}'.format(self._data['NumAtoms'])]
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])]
        
        if (self.config.sections['CALCULATOR'].force):
            self.pt.fitsnap_dict['NumDgradRows'][self.distributed_index:dindex] = ['{}'.format(nrows_dgrad)]

        # reset indices since we are stacking data in the shared arrays

        self.shared_index = index
        self.distributed_index = dindex
        self.shared_index_b = index_b
        self.shared_index_c = index_c
        self.shared_index_dgrad = index_dgrad

    def _collect_lammps(self):

        num_atoms = self._data["NumAtoms"]
        num_types = config.sections['ACE'].numtypes
        n_coeff = config.sections['ACE'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # Extract positions
        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        # Extract types
        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data

        nrows_energy = 1
        bik_rows = 1
        ndim_force = 3
        nrows_force = ndim_force * num_atoms
        ndim_virial = 6
        nrows_virial = ndim_virial
        nrows_snap = nrows_energy + nrows_force + nrows_virial
        ncols_bispectrum = n_coeff * num_types
        ncols_reference = 1
        ncols_snap = ncols_bispectrum + ncols_reference
        index = self.shared_index
        dindex = self.distributed_index
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))
        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            pt.single_print('WARNING! applying np.nan_to_num()')
            lmp_snap = np.nan_to_num(lmp_snap)
        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))

        irow = 0
        bik_rows = 1
        if config.sections['ACE'].bikflag:
            bik_rows = num_atoms
        icolref = ncols_bispectrum
        if config.sections["CALCULATOR"].energy:
            b_sum_temp = lmp_snap[irow, :ncols_bispectrum] / num_atoms

            # Check for no neighbors using B[0,0,0] components
            # these strictly increase with total neighbor count
            # minimum value depends on SNAP variant

            EPS = 1.0e-10
            b000sum0 = 0.0
            nstride = n_coeff
            b000sum = sum(b_sum_temp[::nstride])
            if not config.sections['ACE'].bikflag:
                if not config.sections["ACE"].bzeroflag:
                    b000sum0 = 1.0
                    if (abs(b000sum - b000sum0) < EPS): 
                        pt.single_print("WARNING: Configuration has no PACE neighbors")

                b_sum_temp.shape = (num_types, n_coeff)
                onehot_atoms = np.zeros((num_types, 1))
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[config.sections["ACE"].type_mapping[atom]-1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                #b_sum_temp.shape = (num_types * (n_coeff + num_types))
                b_sum_temp.shape = (num_types * n_coeff + num_types)
            pt.shared_arrays['a'].array[index] = b_sum_temp * config.sections["ACE"].blank2J
            ref_energy = lmp_snap[irow, icolref]
            pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            pt.shared_arrays['w'].array[index] = self._data["eweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
            pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if config.sections["CALCULATOR"].force:
            db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
            db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
            if not config.sections["ACE"].bzeroflag:
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(db_atom_temp)[0], num_types, 1))
                db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=2)
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types * n_coeff + num_types)
            pt.shared_arrays['a'].array[index:index+num_atoms * ndim_force] = \
                np.matmul(db_atom_temp, np.diag(config.sections["ACE"].blank2J))
            ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
            pt.shared_arrays['b'].array[index:index+num_atoms * ndim_force] = \
                self._data["Forces"].ravel() - ref_forces
            pt.shared_arrays['w'].array[index:index+num_atoms * ndim_force] = \
                self._data["fweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
            pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

        if config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not config.sections["ACE"].bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], num_types, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types * n_coeff + num_types)
            pt.shared_arrays['a'].array[index:index+ndim_virial] = \
                np.matmul(vb_sum_temp, np.diag(config.sections["ACE"].blank2J))
            ref_stress = lmp_snap[irow:irow + nrows_virial, icolref]
            pt.shared_arrays['b'].array[index:index+ndim_virial] = \
                self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress
            pt.shared_arrays['w'].array[index:index+ndim_virial] = \
                self._data["vweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + ndim_virial] = ['Stress'] * ndim_virial
            pt.fitsnap_dict['Atom_I'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            index += ndim_virial
            dindex += ndim_virial

        length = dindex - self.distributed_index

        pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
        pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
        pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
        self.shared_index = index
        self.distributed_index = dindex

    def _collect_lammps_preprocess(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['ACE'].numtypes
        n_coeff = self.config.sections['ACE'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # extract positions

        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)

        # extract types

        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # extract SNAP data, including reference potential data

        bik_rows = 1
        if self.config.sections['ACE'].bikflag:
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

        # check that number of atoms here is equal to number of atoms in the sliced array

        natoms_sliced = self.pt.shared_arrays['number_of_atoms'].sliced_array[self._i]
        assert(natoms_sliced==num_atoms)
        self.pt.shared_arrays['number_of_dgrad_rows'].sliced_array[self._i] = nrows_dgrad
