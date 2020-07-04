import ctypes
from fitsnap3.calculators.calculator import Calculator
from fitsnap3.parallel_tools import pt
from fitsnap3.io.input import config
import numpy as np

# TODO: Ask about onehot_fraction for energy


class LammpsSnap(Calculator):

    def __init__(self, name):
        super().__init__(name)
        self._data = {}
        self._i = 0
        self._lmp = None
        pt.check_lammps()

    def create_a(self):
        super().create_a()

    def process_configs(self, data, i):
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

        self._set_variables(**_lammps_variables(config.sections["BISPECTRUM"].__dict__))

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
        for i, (a_t, (a_x, a_y, a_z)) in enumerate(zip(self._data["AtomTypes"], self._data["Positions"])):
            a_t = config.sections["BISPECTRUM"].type_mapping[a_t]
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
            self._lmp.command(f"set atom {i + 1} charge {q:20.20g} ")
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
            k: config.sections["CALCULATOR"].__dict__[v]
            for k, v in
            {
                "rmin0": "rmin0",
                "bzeroflag": "bzeroflag",
                "quadraticflag": "quadraticflag",
                "switchflag": "switchflag",
                "chem": "chemflag",
                "bnormflag": "bnormflag",
                "wselfallflag": "wselfallflag",
            }.items()
            if v in config.sections["CALCULATOR"].__dict__
        }
        if kw_options["chem"] == 0:
            kw_options.pop("chem")
        kw_options["rmin0"] = config.sections["BISPECTRUM"].rmin0
        kw_substrings = [f"{k} {v}" for k, v in kw_options.items()]
        kwargs = " ".join(kw_substrings)

        # everything is handled by LAMMPS compute snap

        base_snap = "compute snap all snap ${rcutfac} ${rfac0} ${twojmax}"
        command = f"{base_snap} {radelem} {wj} {kwargs}"
        self._lmp.command(command)

    def _run_lammps(self):
        self._lmp.command("run 0")

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

        nrows_energy = 1
        ndim_force = 3
        nrows_force = ndim_force * num_atoms
        ndim_virial = 6
        nrows_virial = ndim_virial
        nrows_snap = nrows_energy + nrows_force + nrows_virial
        ncols_bispectrum = num_types * n_coeff
        ncols_reference = 1
        ncols_snap = ncols_bispectrum + ncols_reference
        index = pt.shared_arrays['a'].indices[self._i]

        lmp_snap = _extract_compute_np(self._lmp, "snap", 0, 2, (nrows_snap, ncols_snap))
        if (np.isinf(lmp_snap)).any() or (np.isnan(lmp_snap)).any():
            raise ValueError('Nan in computed data of file {} in group {}'.format(self._data["File"],
                                                                                  self._data["Group"]))

        irow = 0
        icolref = ncols_bispectrum
        if config.sections["CALCULATOR"].energy:
            b_sum_temp = lmp_snap[irow, :ncols_bispectrum] / num_atoms
            if not config.sections["CALCULATOR"].bzeroflag:
                b_sum_temp.shape = (num_types, n_coeff)
                onehot_atoms = np.ones((num_types, 1))
                b_sum_temp = np.concatenate((onehot_atoms, b_sum_temp), axis=1)
                b_sum_temp.shape = (num_types * n_coeff + num_types)
            # fix onehot_atoms to be true fraction not just 1
            # onehot_atoms = np.stack([np.eye(num_types)[x - 1, :].sum(axis=0) for x in lmp_types])
            # onehot_fraction = onehot_atoms / onehot_atoms.sum(axis=1)[:, np.newaxis
            pt.shared_arrays['a'].array[index] = b_sum_temp
            ref_energy = lmp_snap[irow, icolref]
            pt.shared_arrays['b'].array[index] = (energy - ref_energy) / num_atoms
            pt.shared_arrays['w'].array[index] = self._data["eweight"]
            irow += nrows_energy
            index += 1

        if config.sections["CALCULATOR"].force:
            db_atom_temp = lmp_snap[irow:irow + nrows_force, :ncols_bispectrum]
            db_atom_temp.shape = (num_atoms * ndim_force, n_coeff * num_types)
            if not config.sections["CALCULATOR"].bzeroflag:
                onehot_atoms = np.zeros((np.shape(db_atom_temp)[0],))
                db_atom_temp = np.concatenate([onehot_atoms[:, np.newaxis], db_atom_temp], axis=1)
            pt.shared_arrays['a'].array[index:index+num_atoms * ndim_force] = db_atom_temp
            ref_forces = lmp_snap[irow:irow + nrows_force, icolref]
            pt.shared_arrays['b'].array[index:index+num_atoms * ndim_force] = \
                self._data["Forces"].ravel() - ref_forces
            pt.shared_arrays['w'].array[index:index+num_atoms * ndim_force] = \
                self._data["fweight"]
            irow += nrows_force
            index += num_atoms * ndim_force

        if config.sections["CALCULATOR"].stress:
            vb_sum_temp = 1.6021765e6*lmp_snap[irow:irow + nrows_virial, :ncols_bispectrum] / lmp_volume
            vb_sum_temp.shape = (ndim_virial, n_coeff * num_types)
            if not config.sections["CALCULATOR"].bzeroflag:
                onehot_atoms = np.zeros((np.shape(vb_sum_temp)[0],))
                vb_sum_temp = np.concatenate([onehot_atoms[:, np.newaxis], vb_sum_temp], axis=1)
            pt.shared_arrays['a'].array[index:index+ndim_virial] = vb_sum_temp
            ref_stress = lmp_snap[irow:irow + nrows_virial, icolref]
            pt.shared_arrays['b'].array[index:index+ndim_virial] = \
                self._data["Stress"][[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].ravel() - ref_stress
            pt.shared_arrays['w'].array[index:index+ndim_virial] = \
                self._data["vweight"]
            index += ndim_virial


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


def _extract_compute_np(lmp, name, compute_style, result_type, array_shape):
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
