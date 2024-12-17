from fitsnap3lib.calculators.lammps_base import LammpsBase, _extract_compute_np
import numpy as np


class LammpsReaxff(LammpsBase):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.pt.check_lammps()

    # a array is for per-atom quantities in all configs (eg charge, ...)
    # b array is for per-config quantities like energy
    # c matrix is for per-atom 3-vectors like position and velocity.

    def get_width(self):
        a_width = self.config.sections["REAXFF"].ncoeff #+ 3
        return a_width
    
    def _prepare_lammps(self):
        self._set_structure()

        # needs reworking when lammps will accept variable 2J
        #self._lmp.command(f"variable twojmax equal {max(self.config.sections['BISPECTRUM'].twojmax)}")

        self._set_computes()
        self._set_neighbor_list()

    def _set_box(self):
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),(ay, by, cy),(az, bz, cz)) = self._data["Lattice"]
        self._lmp.command(f'region box block -{ax:20.20g} {ax:20.20g} -{by:20.20g} {by:20.20g} -{cz:20.20g} {cz:20.20g}')
        numtypes=self.config.sections['REAXFF'].numtypes)
        self._lmp.command(f"create_box {numtypes} box")

    def _create_atoms(self):
        self._create_atoms_helper(type_mapping=self.config.sections["REAXFF"].type_mapping)

    def _set_computes(self):
        numtypes = self.config.sections['REAXFF'].numtypes

        # everything is handled by LAMMPS compute snap

        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)

    def _collect_lammps_nonlinear(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['REAXFF'].numtypes
        energy = self._data["Energy"]

        lmp_atom_ids  = self._extract_atom_ids(num_atoms)
        lmp_pos  = self._extract_atom_positions(num_atoms)
        lmp_types  = self._extract_atom_types(num_atoms)
        
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms\nGroup and configuration: {} {}".format(self._data["Group"], self._data["File"])
        
        lmp_volume = self._lmp.get_thermo("vol")

        # extract SNAP data, including reference potential data

        bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)
        ncols_bispectrum = n_coeff


        # extract the useful parts of the snap array

        bispectrum_components = lmp_snap[0:bik_rows, 3:n_coeff+3]
        #print(bispectrum_components)
        ref_forces = lmp_snap[0:bik_rows, 0:3].flatten()
        dgrad = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 3:n_coeff+3]
        dgrad_indices = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 0:3].astype(np.int32)
        ref_energy = lmp_snap[-1, 0]

        if (self.config.sections['CALCULATOR'].force):
            # populate the force truth array 'c'

            self.pt.shared_arrays['c'].array[index_c:(index_c + (3*num_atoms))] = self._data["Forces"].ravel() - ref_forces
            index_c += 3*num_atoms

    def _collect_lammps_single(self):

        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['BISPECTRUM'].numtypes
        n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids  = self._extract_atom_ids(num_atoms)
        lmp_pos  = self._extract_atom_positions(num_atoms)
        lmp_types  = self._extract_atom_types(num_atoms)

        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms\nGroup and configuration: {} {}".format(self._data["Group"], self._data["File"])
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))

        # Get individual A matrices for this configuration.

        if self.config.sections['CALCULATOR'].energy:
            nrows += num_atoms
        if self.config.sections['CALCULATOR'].force:
            nrows += 3*num_atoms
        if self.config.sections['CALCULATOR'].stress:
            pass # FIXME

        a = np.zeros((na, nd))
        b = np.zeros(na)
        w = np.zeros(na)

        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))
        irow = 0
        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        icolref = ncols_bispectrum
        if self.config.sections["CALCULATOR"].energy:
            b_sum_temp = lmp_snap[irow:irow+bik_rows, :ncols_bispectrum]
            if not self.config.sections["BISPECTRUM"].bikflag:
                # Divide by natoms if not extracting per-atom descriptors.
                b_sum_temp /= num_atoms

            # Check for no neighbors using B[0,0,0] components
            # these strictly increase with total neighbor count
            # minimum value depends on SNAP variant

            EPS = 1.0e-10
            b000sum0 = 0.0
            nstride = n_coeff
            if not self.config.sections['BISPECTRUM'].bikflag:
                if not self.config.sections["BISPECTRUM"].bzeroflag:
                    b000sum0 = 1.0
                if self.config.sections["BISPECTRUM"].chemflag:
                    nstride //= num_types*num_types*num_types
                    if self.config.sections["BISPECTRUM"].wselfallflag:
                        b000sum0 *= num_types*num_types*num_types
                b000sum = sum(b_sum_temp[0, ::nstride])
                if abs(b000sum - b000sum0) < EPS:
                    print("! WARNING: Configuration has no SNAP neighbors \nGroup and configuration: {} {}".format(self._data["Group"],self._data["File"]))

            if not self.config.sections["BISPECTRUM"].bzeroflag:
                if self.config.sections['BISPECTRUM'].bikflag:
                    raise NotImplementedError("per atom energy is not implemented without bzeroflag")
                b_sum_temp.shape = (num_types, n_coeff)
                onehot_atoms = np.zeros((num_types, 1))
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[self.config.sections["BISPECTRUM"].type_mapping[atom]-1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                b_sum_temp.shape = (num_types * n_coeff + num_types)

            # Get matrix of descriptors (A).
            a[irow:irow+bik_rows] = b_sum_temp * self.config.sections["BISPECTRUM"].blank2J[np.newaxis, :]

            # Get vector of truths (b).
            ref_energy = lmp_snap[irow, icolref]
            b[irow:irow+bik_rows] = 0.0
            b[irow] = (energy - ref_energy) / num_atoms

            # Get weights (w).
            w[irow] = self._data["eweight"] if "eweight" in self._data else 1.0

            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if self.config.sections["CALCULATOR"].force:
            db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
            db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(db_atom_temp)[0], num_types, 1))
                db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=2)
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types * n_coeff + num_types)
            # Get matrix of descriptor derivatives (A).
            a[irow:irow+nrows_force] = np.matmul(db_atom_temp, np.diag(self.config.sections["BISPECTRUM"].blank2J)) 
            
            # Get vector of true forces (b).
            ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
            b[irow:irow+nrows_force] = self._data["Forces"].ravel() - ref_forces
            
            # Get vector of force weights (w).
            w[irow:irow+nrows_force] = self._data["fweight"] if "fweight" in self._data else 1.0

            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

        if self.config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], num_types, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types * n_coeff + num_types)
            
            # Get matrix of descriptor virials (A).
            a[irow:irow+ndim_virial] = np.matmul(vb_sum_temp, np.diag(self.config.sections["BISPECTRUM"].blank2J))

            # Get vector of true stresses (b).
            ref_stress = lmp_snap[irow:irow + nrows_virial, icolref]
            b[irow:irow+ndim_virial] = self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress

            # Get stress weights (w).
            w[irow:irow+ndim_virial] = self._data["vweight"] if "vweight" in self._data else 1.0

            index += ndim_virial
            dindex += ndim_virial

        self.shared_index = index
        self.distributed_index = dindex

        return a,b,w

    def _collect_lammps(self):

        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['BISPECTRUM'].numtypes
        n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids  = self._extract_atom_ids(num_atoms)
        lmp_pos  = self._extract_atom_positions(num_atoms)
        lmp_types  = self._extract_atom_types(num_atoms)

        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms\nGroup and configuration: {} {}".format(self._data["Group"], self._data["File"])

        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data

        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        nrows_force = ndim_force * num_atoms
        ndim_virial = 6
        nrows_virial = ndim_virial
        nrows_snap = nrows_energy + nrows_force + nrows_virial
        ncols_bispectrum = n_coeff * num_types
        ncols_reference = 1
        ncols_snap = ncols_bispectrum + ncols_reference
        # index = pt.fitsnap_dict['a_indices'][self._i]
        index = self.shared_index
        dindex = self.distributed_index

        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))

        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))
        irow = 0
        bik_rows = 1
        if self.config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        icolref = ncols_bispectrum
        if self.config.sections["CALCULATOR"].energy:
            b_sum_temp = lmp_snap[irow:irow+bik_rows, :ncols_bispectrum] / num_atoms

            # Check for no neighbors using B[0,0,0] components
            # these strictly increase with total neighbor count
            # minimum value depends on SNAP variant

            EPS = 1.0e-10
            b000sum0 = 0.0
            nstride = n_coeff
            if not self.config.sections['BISPECTRUM'].bikflag:
                if not self.config.sections["BISPECTRUM"].bzeroflag:
                    b000sum0 = 1.0
                if self.config.sections["BISPECTRUM"].chemflag:
                    nstride //= num_types*num_types*num_types
                    if self.config.sections["BISPECTRUM"].wselfallflag:
                        b000sum0 *= num_types*num_types*num_types
                b000sum = sum(b_sum_temp[0, ::nstride])
                if abs(b000sum - b000sum0) < EPS:
                    print("! WARNING: Configuration has no SNAP neighbors \nGroup and configuration: {} {}".format(self._data["Group"],self._data["File"]))

            if not self.config.sections["BISPECTRUM"].bzeroflag:
                if self.config.sections['BISPECTRUM'].bikflag:
                    raise NotImplementedError("per atom energy is not implemented without bzeroflag")
                b_sum_temp.shape = (num_types, n_coeff)
                onehot_atoms = np.zeros((num_types, 1))
                for atom in self._data["AtomTypes"]:
                    onehot_atoms[self.config.sections["BISPECTRUM"].type_mapping[atom]-1] += 1
                onehot_atoms /= len(self._data["AtomTypes"])
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                b_sum_temp.shape = (num_types * n_coeff + num_types)

            # Get matrix of descriptors (A).
            self.pt.shared_arrays['a'].array[index:index+bik_rows] = \
                b_sum_temp * self.config.sections["BISPECTRUM"].blank2J[np.newaxis, :]
            
            ref_energy = lmp_snap[irow, icolref]

            # Get vector of truths (b).
            self.pt.shared_arrays['b'].array[index:index+bik_rows] = 0.0
            self.pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms

            # Get weights (w).
            self.pt.shared_arrays['w'].array[index] = self._data["eweight"]
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]

            # create an atom types list for the energy rows, if bikflag=1
            if self.config.sections['BISPECTRUM'].bikflag:
                types_energy = [int(i) for i in lmp_types]
                self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + bik_rows] = types_energy
            else:
                self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + bik_rows] = [0]

            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if self.config.sections["CALCULATOR"].force:
            db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
            db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(db_atom_temp)[0], num_types, 1))
                db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=2)
                db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types * n_coeff + num_types)
            # Get matrix of descriptor derivatives (A).
            self.pt.shared_arrays['a'].array[index:index+nrows_force] = \
                np.matmul(db_atom_temp, np.diag(self.config.sections["BISPECTRUM"].blank2J))
            
            ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
            # Get vector of true forces (b).
            self.pt.shared_arrays['b'].array[index:index+nrows_force] = \
                self._data["Forces"].ravel() - ref_forces
            
            # Get vector of force weights (w).
            self.pt.shared_arrays['w'].array[index:index+nrows_force] = \
                self._data["fweight"]
            # Populate dictionaries.
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
            # create a types list for the force rows
            types_force = [] 
            for typ in lmp_types:
                for a in range(0,3):
                    types_force.append(typ)
            self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + nrows_force] = types_force
            index += nrows_force
            dindex += nrows_force
        irow += nrows_force

        if self.config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not self.config.sections["BISPECTRUM"].bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], num_types, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types * n_coeff + num_types)
            
            # Get matrix of descriptor virials (A).
            self.pt.shared_arrays['a'].array[index:index+ndim_virial] = \
                np.matmul(vb_sum_temp, np.diag(self.config.sections["BISPECTRUM"].blank2J))
            ref_stress = lmp_snap[irow:irow + nrows_virial, icolref]
            # Get vector of true stresses (b).
            self.pt.shared_arrays['b'].array[index:index+ndim_virial] = \
                self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress
            # Get stress weights (w).
            self.pt.shared_arrays['w'].array[index:index+ndim_virial] = \
                self._data["vweight"]
            # Populate dictionaries.
            self.pt.fitsnap_dict['Row_Type'][dindex:dindex + ndim_virial] = ['Stress'] * ndim_virial
            self.pt.fitsnap_dict['Atom_I'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            self.pt.fitsnap_dict['Atom_Type'][dindex:dindex + ndim_virial] = [int(0)] * ndim_virial
            index += ndim_virial
            dindex += ndim_virial

        length = dindex - self.distributed_index
        self.pt.fitsnap_dict['Groups'][self.distributed_index:dindex] = ['{}'.format(self._data['Group'])] * length
        self.pt.fitsnap_dict['Configs'][self.distributed_index:dindex] = ['{}'.format(self._data['File'])] * length
        self.pt.fitsnap_dict['Testing'][self.distributed_index:dindex] = [bool(self._data['test_bool'])] * length
        self.shared_index = index
        self.distributed_index = dindex

    def _collect_lammps_preprocess(self):
        num_atoms = self._data["NumAtoms"]
        num_types = self.config.sections['BISPECTRUM'].numtypes
        n_coeff = self.config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids  = self._extract_atom_ids(num_atoms)
        lmp_pos  = self._extract_atom_positions(num_atoms)
        lmp_types  = self._extract_atom_types(num_atoms)

        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms\nGroup and configuration: {} {}".format(self._data["Group"], self._data["File"])
        lmp_volume = self._lmp.get_thermo("vol")

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
