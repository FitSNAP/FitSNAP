import ctypes
from fitsnap3lib.calculators.calculator import Calculator
import numpy as np


class LammpsBase(Calculator):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0
        self._lmp = None
        self.pt.check_lammps()

    def create_a(self):
        super().create_a()

    def preprocess_allocate(self, nconfigs: int):
        """
        Allocate arrays to be used by this proc. These arrays have size nconfigs.

        Args:
            nconfigs : number of configs on this proc
        """
        self.dgradrows = np.zeros(nconfigs).astype(int)
        # number of dgrad rows per config, organized like the other shared arrays in calculator.py
        self.pt.create_shared_array('number_of_dgradrows', nconfigs, tm=self.config.sections["SOLVER"].true_multinode)
        # number of neighbors per config, organized like the other shared arrays in calculator.py
        self.pt.create_shared_array('number_of_neighs', nconfigs, tm=self.config.sections["SOLVER"].true_multinode)
        self.nconfigs = nconfigs

    def preprocess_configs(self, data, i):
        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps_preprocess()
            self._lmp = self.pt.close_lammps()
        except Exception as e:
            #if self.config.args.printlammps:
            self._data = data
            self._i = i
            self._initialize_lammps(1)
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps_preprocess()
            self._lmp = self.pt.close_lammps()
            raise e

    def process_configs(self, data, i):
        """
        Calculate descriptors for a given configuration.
        Action of this function is altered by certain attributes.

        Args:
            transpose_trick : Don't touch shared arrays in `_collect_lammps()` if true. Instead 
                              store smaller matrices `self.aw`, `self.bw`.
        """
        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps()
            self._lmp = self.pt.close_lammps()
        except Exception as e:
            #if self.config.args.printlammps:
            self._data = data
            self._i = i
            self._initialize_lammps(1)
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps()
            self._lmp = self.pt.close_lammps()
            raise e
        
    def process_configs_nonlinear(self, data, i):
        try:
            self._data = data
            self._i = i
            self._initialize_lammps()
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps_nonlinear()
            self._lmp = self.pt.close_lammps()
        except Exception as e:
            #if self.config.args.printlammps:
            self._data = data
            self._i = i
            self._initialize_lammps(1)
            self._prepare_lammps()
            self._run_lammps()
            self._collect_lammps_nonlinear()
            self._lmp = self.pt.close_lammps()
            raise e

        
    def process_single(self, data, i=0):
        """
        Calculate descriptors on a single configuration without touching the shraed arrays.

        Args:
            data: dictionary of structural and fitting info for a configuration in fitsnap
                  data dictionary format.
            i: integer index which is optional, mainly for debugging purposes.
        
        Returns: 
            - A matrix of descriptors depending on settings declared in `CALCULATOR`. If 
              `bikflag` is 0 (default) then A has 1 and 0s in the first column since it is ready to 
              fit with linear solvers; the descriptors are also divided by no. atoms in this case. 
              If `bikflag` is 1, then A is simply an unaltered per-atom descriptor matrix.
            - b vector of truths
            - w vector of weights
        """
        self._data = data
        self._i = i
        self._initialize_lammps()
        self._prepare_lammps()
        self._run_lammps()
        a,b,w = self._collect_lammps_single()
        self._lmp = self.pt.close_lammps()
        return a,b,w

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
        self._lmp = self.pt.initialize_lammps(self.config.args.lammpslog, printlammps)

    def _set_structure(self):
        self._lmp.command("clear")
        self._lmp.command("units " + self.config.sections["REFERENCE"].units)
        self._lmp.command("atom_style " + self.config.sections["REFERENCE"].atom_style)

        lmp_setup = _extract_commands("""
                        atom_modify map array sort 0 2.0
                        box tilt large""")
        for line in lmp_setup:
            self._lmp.command(line)

        self._set_box()

        self._create_atoms()

        if self.config.sections["REFERENCE"].atom_style == "spin":
            self._create_spins()
        if self.config.sections["REFERENCE"].atom_style == "charge":
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

            # no casting needed, lammps.py already works

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
