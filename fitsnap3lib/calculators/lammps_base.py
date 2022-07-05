import ctypes
from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.parallel_tools import ParallelTools, DistributedList
from fitsnap3lib.io.input import Config
import numpy as np


config = Config()
pt = ParallelTools()


class LammpsBase(Calculator):

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

    def _prepare_lammps(self):
        raise NotImplementedError

    def _collect_lammps(self):
        raise NotImplementedError

    def _set_box(self):
        raise NotImplementedError

    def _create_atoms(self):
        raise NotImplementedError

    def _set_computes(self):
        raise NotImplementedError

    def _initialize_lammps(self, printlammps=0):
        self._lmp = pt.initialize_lammps(config.args.lammpslog, printlammps)

    def _set_structure(self):
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

    def _set_neighbor_list(self):
        self._lmp.command("mass * 1.0e-20")
        self._lmp.command("neighbor 1.0e-20 nsq")
        self._lmp.command("neigh_modify one 10000")

    def _set_box_helper(self, numtypes):
        self._lmp.command("boundary p p p")
        ((ax, bx, cx),
         (ay, by, cy),
         (az, bz, cz)) = self._data["Lattice"]

        assert all(abs(c) < 1e-10 for c in (ay, az, bz)), \
            "Cell not normalized for lammps!"
        region_command = \
            f"region pybox prism 0 {ax:20.20g} 0 {by:20.20g} 0 {cz:20.20g} {bx:20.20g} {cx:20.20g} {cy:20.20g}"
        self._lmp.command(region_command)
        self._lmp.command(f"create_box {numtypes} pybox")

    def _create_atoms_helper(self, type_mapping):
        number_of_atoms = len(self._data["AtomTypes"])
        positions = self._data["Positions"].flatten()
        elem_all = [type_mapping[a_t] for a_t in self._data["AtomTypes"]]
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
        spin = self._data["Spins"].flatten()
        self._lmp.scatter_atoms("sp", 1, 4, (len(spin) * ctypes.c_double)(*spin))
        n_atoms = int(self._lmp.get_natoms())
        n_spins = len(self._data["Spins"])
        assert n_spins == n_atoms, f"Atom counts don't match when assigning spins: {n_spins}, {n_atoms}"

    def _create_charge(self):
        charges = self._data["Charges"].flatten()
        self._lmp.scatter_atoms("q", 1, 1, (len(charges) * ctypes.c_int)(*charges))
        n_atoms = int(self._lmp.get_natoms())
        n_charges = len(self._data["Charges"])
        assert n_charges == n_atoms, f"Atom counts don't match when assigning charge: {n_charges}, {n_atoms}"

    def _set_variables(self, **lmp_variable_args):
        for k, v in lmp_variable_args.items():
            self._lmp.command(f"variable {k} equal {v}")

    def _run_lammps(self):
        self._lmp.command("run 0")


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
