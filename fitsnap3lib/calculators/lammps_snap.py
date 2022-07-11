import ctypes
from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
import numpy as np


config = Config()
pt = ParallelTools()


class LammpsSnap(Calculator):

    def __init__(self, name):
        super().__init__(name)
        self._data = {}
        self._i = 0
        self._lmp = None
        self._row_index = 0
        self.dgradrows = None
        pt.check_lammps()

        # check lammps version for compute snap settings
        """
        if (config.sections["BISPECTRUM"].__dict__["switchinnerflag"]):
            if (pt.lammps_version < 20220623)
        print("-----------------------------------")
        print(f"LAMMPS version: {pt.lammps_version}")
        print(f"LAMMPS version: {type(pt.lammps_version)}")
        print("-----------------------------------")
        print(config.sections["BISPECTRUM"].__dict__["switchinnerflag"])
        print(config.sections["BISPECTRUM"].__dict__["dgradflag"])
        """

    def get_width(self):
        if (config.sections["SOLVER"].solver == "PYTORCH"):
            a_width = config.sections["BISPECTRUM"].ncoeff #+ 3
        else:
            num_types = config.sections["BISPECTRUM"].numtypes
            a_width = config.sections["BISPECTRUM"].ncoeff * num_types
            if not config.sections["BISPECTRUM"].bzeroflag:
                a_width += num_types
        return a_width

    def create_a(self):
        super().create_a()

    def preprocess_allocate(self, nconfigs):
        #print("----- preprocess_allcoate in lammps_snap.py")
        self.dgradrows = np.zeros(nconfigs).astype(int)
        pt.create_shared_array('number_of_dgradrows', nconfigs, tm=config.sections["SOLVER"].true_multinode)

    def preprocess_configs(self, data, i):
        #print("----- preprocess_configs in lammps_snap.py")
        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps_preprocess()
            self._lmp = pt.close_lammps()
        except Exception as e:
            if config.args.printlammps:
                self._data = data
                self._i = i
                self._initialize_lammps(1)
                self._prepare_lammps()
                self._run_lammps()
                self._collect_lammps_preprocess()
                self._lmp = pt.close_lammps()
            raise e

    def process_configs(self, data, i):
        #print("----- process_configs in lammps_snap.py")
        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps()
            self._lmp = pt.close_lammps()
        except Exception as e:
            if config.args.printlammps:
                self._data = data
                self._i = i
                self._initialize_lammps(1)
                self._prepare_lammps()
                self._run_lammps()
                self._collect_lammps()
                self._lmp = pt.close_lammps()
            raise e

    def process_configs_nonlinear(self, data, i):
        #print("----- process_configs in lammps_snap.py")
        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps_nonlinear()
            self._lmp = pt.close_lammps()
        except Exception as e:
            if config.args.printlammps:
                self._data = data
                self._i = i
                self._initialize_lammps(1)
                self._prepare_lammps()
                self._run_lammps()
                self._collect_lammps_nonlinear()
                self._lmp = pt.close_lammps()
            raise e

    def _initialize_lammps(self, printlammps=0):
        self._lmp = pt.initialize_lammps(config.args.lammpslog, printlammps)

    def _prepare_lammps(self):
        self._lmp.command("clear")
        self._lmp.command("units " + config.sections["REFERENCE"].units)
        self._lmp.command("atom_style " + config.sections["REFERENCE"].atom_style)

        lmp_setup = _extract_commands("""
                atom_modify map array sort 0 2.0
                box tilt large""")
        for line in lmp_setup:
            self._lmp.command(line)

        self._set_box()

        self._create_atoms()

        if config.sections["REFERENCE"].atom_style == "spin":
            self._create_spins()
        if config.sections["REFERENCE"].atom_style == "charge":
            self._create_charge()

        # this is super clean when there is only one value per key, needs reworking
#        self._set_variables(**_lammps_variables(config.sections["BISPECTRUM"].__dict__))

        # needs reworking when lammps will accept variable 2J

        self._lmp.command(f"variable twojmax equal {max(config.sections['BISPECTRUM'].twojmax)}")
        self._lmp.command(f"variable rcutfac equal {config.sections['BISPECTRUM'].rcutfac}")
        self._lmp.command(f"variable rfac0 equal {config.sections['BISPECTRUM'].rfac0}")
#        self._lmp.command(f"variable rmin0 equal {config.sections['BISPECTRUM'].rmin0}")

        for i, j in enumerate(config.sections["BISPECTRUM"].wj):
            self._lmp.command(f"variable wj{i+1} equal {j}")

        for i,j in enumerate(config.sections["BISPECTRUM"].radelem):
            self._lmp.command(f"variable radelem{i+1} equal {j}")

        for line in config.sections["REFERENCE"].lmp_pairdecl:
            self._lmp.command(line.lower())

        self._set_computes()

        self._lmp.command("mass * 1.0e-20")
        self._lmp.command("neighbor 1.0e-20 nsq")
        self._lmp.command("neigh_modify one 10000")

    def _set_box(self):
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),
         (ay, by, cy),
         (az, bz, cz)) = self._data["Lattice"]

        assert all(abs(c) < 1e-10 for c in (ay, az, bz)), \
            "Cell not normalized for lammps!"
        region_command = \
            f"region pybox prism 0 {ax:20.20g} 0 {by:20.20g} 0 {cz:20.20g} {bx:20.20g} {cx:20.20g} {cy:20.20g}"
        self._lmp.command(region_command)
        self._lmp.command(f"create_box {config.sections['BISPECTRUM'].numtypes} pybox")

    def _create_atoms(self):
        number_of_atoms = len(self._data["AtomTypes"])
        positions = self._data["Positions"].flatten()
        elem_all = [config.sections["BISPECTRUM"].type_mapping[a_t] for a_t in self._data["AtomTypes"]]
        self._lmp.create_atoms(
            n=number_of_atoms,
            id=None,
            type=(len(elem_all) * ctypes.c_int)(*elem_all),
            x=(len(positions) * ctypes.c_double)(*positions),
            v=None,
            image=None,
            shrinkexceed=False
        )
        n_atoms = int(self._lmp.get_natoms())
        assert number_of_atoms == n_atoms, f"Atom counts don't match when creating atoms: {number_of_atoms}, {n_atoms}"

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

    def _set_computes(self):
        numtypes = config.sections['BISPECTRUM'].numtypes
        radelem = " ".join([f"${{radelem{i}}}" for i in range(1, numtypes + 1)])
        wj = " ".join([f"${{wj{i}}}" for i in range(1, numtypes + 1)])

        kw_options = {
            k: config.sections["BISPECTRUM"].__dict__[v]
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
                "sinner": "sinner",
                "dinner": "dinner",
                "dgradflag": "dgradflag",
            }.items()
            if v in config.sections["BISPECTRUM"].__dict__
        }

        # remove input dictionary keywords if they are not used, to avoid version problems

        if kw_options["chem"] == 0:
            kw_options.pop("chem")
        if kw_options["bikflag"] == 0:
            kw_options.pop("bikflag")
        if kw_options["switchinnerflag"] == 0:
            kw_options.pop("switchinnerflag")
        if kw_options["dgradflag"] == 0:
            kw_options.pop("dgradflag")
        kw_options["rmin0"] = config.sections["BISPECTRUM"].rmin0
        kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
        kwargs = " ".join(kw_substrings)

        # everything is handled by LAMMPS compute snap

        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)
        #print("----- Setting computes.")

    def _run_lammps(self):
        #print("----- running LAMMPS.")
        self._lmp.command("run 0")

    def _collect_lammps_nonlinear(self):

        if (pt._sub_rank==0):
            print("Collect lammps nonlinear")
        num_atoms = self._data["NumAtoms"]
        num_types = config.sections['BISPECTRUM'].numtypes
        n_coeff = config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # Extract positions
        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        # Extract types
        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data
        bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)
        ncols_bispectrum = n_coeff
        ncols_snap = n_coeff + 3 # Number of columns in the snap array, add 3 to include indices and Cartesian components.
        ncols_reference = 0
        nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-1 #-6
        nrows_snap = nrows_energy + nrows_dgrad + 1 #nrows_virial
        assert nrows_snap == np.shape(lmp_snap)[0]
        index = self.shared_index # Index telling where to start in the shared arrays on this proc.
                                  # Currently this is an index for the 'a' array (natoms*nconfigs rows).
                                  # Need to also make an index for:
                                  # - the 'b' array (3*natoms+1)*nconfigs rows.
                                  # - the 'dgrad' array (natoms+1)*nneigh*3*nconfigs rows.
                                  # - the 'dgrad_indices' array which has same number of rows as 'dgrad'
        dindex = self.distributed_index
        index_b = self.shared_index_b
        index_c = self.shared_index_c
        index_dgrad = self.shared_index_dgrad
        index_unique_j = self.shared_index_unique_j

        # Extract the useful parts of the snap array.
        #bispectrum_components = lmp_snap[0:bik_rows, 0:n_coeff]
        bispectrum_components = lmp_snap[0:bik_rows, 3:n_coeff+3]
        #print(bispectrum_components)
        #ref_forces = lmp_snap[0:bik_rows, -3:].flatten()
        ref_forces = lmp_snap[0:bik_rows, 0:3].flatten()
        #print(ref_forces)
        #dgrad = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 0:n_coeff]
        dgrad = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 3:n_coeff+3]
        #print(np.shape(dgrad))
        #dgrad_indices = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), -3:].astype(np.int32)
        dgrad_indices = lmp_snap[bik_rows:(bik_rows+nrows_dgrad), 0:3].astype(np.int32)
        #print(dgrad_indices)
        #ref_energy = lmp_snap[-6,0]
        ref_energy = lmp_snap[-1, 0]
        #print(ref_energy)

        # Populate the bispectrum array 'a'
        pt.shared_arrays['a'].array[index:index+bik_rows] = bispectrum_components
        index += num_atoms

        # Populate the truth array 'b'
        pt.shared_arrays['b'].array[index_b] = energy - ref_energy
        #print(f"{index_b} {3*num_atoms}")
        #pt.shared_arrays['b'].array[index_b+1:(index_b+1) + 3*num_atoms] = self._data["Forces"].ravel() - ref_forces
        index_b += 1 #3*num_atoms + 1

        # Populate the truth array 'c'
        pt.shared_arrays['c'].array[index_c:(index_c + (3*num_atoms))] = self._data["Forces"].ravel() - ref_forces
        index_c += 3*num_atoms

        # Populate the dgrad arrays 'dgrad' and 'dbdrindx'
        pt.shared_arrays['dgrad'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = dgrad
        pt.shared_arrays['dbdrindx'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = dgrad_indices
        #index_dgrad += nrows_dgrad

        # Populate the unique_j_indices array.
        unique_j_indices = []
        jold = dgrad_indices[0,1]
        #print(jold)
        #counter = index_unique_j
        for jindx in range(0,nrows_dgrad):
            jtmp = dgrad_indices[jindx,1]
            if (jold==jtmp):
                value = index_unique_j
                unique_j_indices.append(value)
                #print(f"{value} {jtmp}")
                #pt.shared_arrays['unique_j_indices'].array[jindx] = jindx
            else:
                jold = jtmp
                index_unique_j = index_unique_j + 1
                value = index_unique_j
                unique_j_indices.append(value)

        unique_j_indices = np.array(unique_j_indices)
        #print(unique_j_indices)
        #print(f"{np.size(unique_j_indices)} {nrows_dgrad}")
        assert(np.size(unique_j_indices) == nrows_dgrad)
        pt.shared_arrays['unique_j_indices'].array[index_dgrad:(index_dgrad+nrows_dgrad)] = unique_j_indices


        index_dgrad += nrows_dgrad
        index_unique_j = index_unique_j + 1

        # Reset indices since we are stacking data in the shared arrays.
        self.shared_index = index
        self.distributed_index = dindex
        self.shared_index_b = index_b
        self.shared_index_c = index_c
        self.shared_index_dgrad = index_dgrad
        self.shared_index_unique_j = index_unique_j


    def _collect_lammps(self):

        num_atoms = self._data["NumAtoms"]
        num_types = config.sections['BISPECTRUM'].numtypes
        n_coeff = config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # Extract positions
        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        # Extract types
        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data

        bik_rows = 1
        if config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)
        if not config.sections['BISPECTRUM'].dgradflag:
            ncols_bispectrum = n_coeff * num_types
            ncols_reference = 1
            nrows_force = ndim_force * num_atoms
            nrows_snap = nrows_energy + nrows_force + nrows_virial
        else:
            ncols_bispectrum = n_coeff + 3
            ncols_reference = 0
            nrows_force = ndim_force * num_atoms
            nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-6
            nrows_snap = nrows_energy + nrows_dgrad + nrows_virial
        #print(f"----- nrows_snap: {nrows_snap}")
        #print(f"----- nrows_dgrad: {nrows_dgrad}")
        ncols_snap = ncols_bispectrum + ncols_reference
        #print(f"ncols_snap: {ncols_snap}")
        # index = pt.fitsnap_dict['a_indices'][self._i]
        index = self.shared_index
        dindex = self.distributed_index

        #lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))
        #print("----- calculators/lammps_snap.py/lmp_snap:")
        #print(lmp_snap)
        #print(lmp_snap[bik_rows:bik_rows+(3*bik_rows), :])

        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))
        irow = 0
        bik_rows = 1
        if config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        icolref = ncols_bispectrum
        if config.sections["CALCULATOR"].energy:
            if not config.sections['BISPECTRUM'].dgradflag:
                b_sum_temp = lmp_snap[irow:irow+bik_rows, :ncols_bispectrum] / num_atoms
            else:
                b_sum_temp = lmp_snap[irow:irow+bik_rows, :(ncols_bispectrum)]

            # Check for no neighbors using B[0,0,0] components
            # these strictly increase with total neighbor count
            # minimum value depends on SNAP variant

            if not config.sections['BISPECTRUM'].dgradflag:
                EPS = 1.0e-10
                b000sum0 = 0.0
                nstride = n_coeff
                if not config.sections['BISPECTRUM'].bikflag:
                    if not config.sections["BISPECTRUM"].bzeroflag:
                        b000sum0 = 1.0
                    if config.sections["BISPECTRUM"].chemflag:
                        nstride //= num_types*num_types*num_types
                        if config.sections["BISPECTRUM"].wselfallflag:
                            b000sum0 *= num_types*num_types*num_types
                    b000sum = sum(b_sum_temp[0, ::nstride])
                    if abs(b000sum - b000sum0) < EPS:
                        print("WARNING: Configuration has no SNAP neighbors")

                if not config.sections["BISPECTRUM"].bzeroflag:
                    if config.sections['BISPECTRUM'].bikflag:
                        raise NotImplementedError("per atom energy is not implemented without bzeroflag")
                    b_sum_temp.shape = (num_types, n_coeff)
                    onehot_atoms = np.zeros((num_types, 1))
                    for atom in self._data["AtomTypes"]:
                        onehot_atoms[config.sections["BISPECTRUM"].type_mapping[atom]-1] += 1
                    onehot_atoms /= len(self._data["AtomTypes"])
                    b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                    b_sum_temp.shape = (num_types * n_coeff + num_types)

                #print(b_sum_temp)
                pt.shared_arrays['a'].array[index:index+bik_rows] = \
                    b_sum_temp * config.sections["BISPECTRUM"].blank2J[np.newaxis, :]
                #print(np.shape(pt.shared_arrays['a'].array[index:index+bik_rows]))
                #print(pt.shared_arrays['a'].array[index:index+bik_rows])
                ref_energy = lmp_snap[irow, icolref]
            else:
                pt.shared_arrays['a'].array[index:index+bik_rows] = \
                    b_sum_temp[:, 0:n_coeff+3]
                print(np.shape(pt.shared_arrays['a'].array[index:index+bik_rows]))
                ref_energy = 0 # Need to add this later
            #pt.shared_arrays['b'].array[index:index+bik_rows] = 0.0
            pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            #print(f"----- b energy: {pt.shared_arrays['b'].array[index]}")
            #pt.shared_arrays['w'].array[index] = self._data["eweight"]
            pt.fitsnap_dict['Row_Type'][dindex:dindex + bik_rows] = ['Energy'] * nrows_energy
            pt.fitsnap_dict['Atom_I'][dindex:dindex + bik_rows] = [int(i) for i in range(nrows_energy)]
            index += nrows_energy
            dindex += nrows_energy
        irow += nrows_energy

        if config.sections["CALCULATOR"].force:
            if not config.sections['BISPECTRUM'].dgradflag:
                db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
                db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
                if not config.sections["BISPECTRUM"].bzeroflag:
                    db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types, n_coeff)
                    onehot_atoms = np.zeros((np.shape(db_atom_temp)[0], num_types, 1))
                    db_atom_temp = np.concatenate([onehot_atoms, db_atom_temp], axis=2)
                    db_atom_temp.shape = (np.shape(db_atom_temp)[0], num_types * n_coeff + num_types)
                pt.shared_arrays['a'].array[index:index+nrows_force] = \
                    np.matmul(db_atom_temp, np.diag(config.sections["BISPECTRUM"].blank2J))
                #print(np.diag(config.sections["BISPECTRUM"].blank2J))
                #print(np.shape(pt.shared_arrays['a'].array[index:index+nrows_force]))
                #print(pt.shared_arrays['a'].array[index:index+nrows_force])
                ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
                pt.shared_arrays['b'].array[index:index+nrows_force] = \
                    self._data["Forces"].ravel() - ref_forces
                pt.shared_arrays['w'].array[index:index+nrows_force] = \
                    self._data["fweight"]
                pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
                pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
                index += nrows_force
                dindex += nrows_force
            else:
                print("----- Preparing descriptor derivatives.")
                db_atom_temp = lmp_snap[irow:irow + nrows_dgrad, :ncols_bispectrum]
                print(f"index: {index}")
                print(f"nrows_dgrad: {nrows_dgrad}")
                print(f"index+nrows_dgrad: {index+nrows_dgrad}")
                print(np.shape(db_atom_temp))
                print(np.shape(pt.shared_arrays['a'].array[index:index+nrows_dgrad]))
                pt.shared_arrays['a'].array[index:(index+nrows_dgrad)] = \
                    db_atom_temp
                #print(np.diag(config.sections["BISPECTRUM"].blank2J))
                #print(np.shape(pt.shared_arrays['a'].array[index:index+nrows_force]))
                #print(pt.shared_arrays['a'].array[index:index+nrows_force])
                #ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
                ref_forces = np.zeros(3*nrows_energy)
                #pt.shared_arrays['b'].array[index:index+nrows_force] = \
                #    self._data["Forces"].ravel() - ref_forces
                #pt.shared_arrays['w'].array[index:index+nrows_force] = \
                #    self._data["fweight"]
                pt.fitsnap_dict['Row_Type'][dindex:dindex + nrows_force] = ['Force'] * nrows_force
                pt.fitsnap_dict['Atom_I'][dindex:dindex + nrows_force] = [int(np.floor(i/3)) for i in range(nrows_force)]
                index += nrows_dgrad
                dindex += nrows_force
        irow += nrows_force

        if config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not config.sections["BISPECTRUM"].bzeroflag:
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types, n_coeff)
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0], num_types, 1))
                vb_sum_temp = np.concatenate([onehot_atoms, vb_sum_temp], axis=2)
                vb_sum_temp.shape = (np.shape(vb_sum_temp)[0], num_types * n_coeff + num_types)
            pt.shared_arrays['a'].array[index:index+ndim_virial] = \
                np.matmul(vb_sum_temp, np.diag(config.sections["BISPECTRUM"].blank2J))
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
        num_types = config.sections['BISPECTRUM'].numtypes
        n_coeff = config.sections['BISPECTRUM'].ncoeff
        energy = self._data["Energy"]

        lmp_atom_ids = self._lmp.numpy.extract_atom_iarray("id", num_atoms).ravel()
        assert np.all(lmp_atom_ids == 1 + np.arange(num_atoms)), "LAMMPS seems to have lost atoms"

        # Extract positions
        lmp_pos = self._lmp.numpy.extract_atom_darray(name="x", nelem=num_atoms, dim=3)
        # Extract types
        lmp_types = self._lmp.numpy.extract_atom_iarray(name="type", nelem=num_atoms).ravel()
        lmp_volume = self._lmp.get_thermo("vol")

        # Extract SNAP data, including reference potential data

        bik_rows = 1
        if config.sections['BISPECTRUM'].bikflag:
            bik_rows = num_atoms
        nrows_energy = bik_rows
        ndim_force = 3
        ndim_virial = 6
        nrows_virial = ndim_virial
        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, None)
        if not config.sections['BISPECTRUM'].dgradflag:
            ncols_bispectrum = n_coeff * num_types
            ncols_reference = 1
            nrows_force = ndim_force * num_atoms
        else:
            ncols_bispectrum = n_coeff + 3
            ncols_reference = 0
            nrows_dgrad = np.shape(lmp_snap)[0]-nrows_energy-1 #6
            dgrad = lmp_snap[num_atoms:(num_atoms+nrows_dgrad), 3:(n_coeff+3)]
            # take out nonzero dgrad fx_components
            #nonzero_rows = lmp_snap[natoms:(natoms+nrows_force),3:(nd+3)] != 0.0
            #nonzero_rows = np.any(nonzero_rows, axis=1)
            #dgrad = dDdR[nonzero_rows, :]
        #print(f"nrows_force: {nrows_force}")
        self.dgradrows[self._i] = nrows_dgrad

# this is super clean when there is only one value per key, needs reworking
def _lammps_variables(bispec_options):
    d = {k: bispec_options[k] for k in
         ["rcutfac",
          "rfac0",
          "rmin0",
          "twojmax"]}
    d.update(
        {
            (k + str(i + 1)): bispec_options[k][i]
            # "zblz", "wj", "radelem"
            for k in ["wj", "radelem"]
            for i, v in enumerate(bispec_options[k])
        })
    return d


def _extract_compute_np(lmp, name, compute_style, result_type, array_shape=None):
    """
    Convert a lammps compute to a numpy array.
    Assumes the compute stores floating point numbers.
    Note that the result is a view into the original memory.
    If the result type is 0 (scalar) then conversion to numpy is
    skipped and a python float is returned.

    From LAMMPS/src/library.cpp:
    style = 0 for global data, 1 for per-atom data, 2 for local data
    type = 0 for scalar, 1 for vector, 2 for array

    """

    if array_shape is None:
        array_np = lmp.numpy.extract_compute(name,compute_style, result_type)
    else:
        ptr = lmp.extract_compute(name, compute_style, result_type)
        if result_type == 0:
            # No casting needed, lammps.py already works
            return ptr
        if result_type == 2:
            ptr = ptr.contents
        total_size = np.prod(array_shape)
        buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * total_size))
        array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
        array_np.shape = array_shape
    return array_np


def _extract_commands(string):
    return [x for x in string.splitlines() if x.strip() != '']
