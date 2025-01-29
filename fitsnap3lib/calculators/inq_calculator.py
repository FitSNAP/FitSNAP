from fitsnap3lib.calculators.calculator import Calculator
import numpy as np

import sys
sys.path.append('/Users/mitch/.local/lib/python3.13/site-packages')
import pinq


class INQCalculator(Calculator):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        self._data = {}
        self._i = 0


    def bond_scan(self):

        # FIXME
        pass


    def process_configs(self, data, i):
        """
        Calculate QM training data for a given configuration.
        """
        try:
            self._data = data
            self._i = i
            self._initialize_inq()
            self._prepare_inq()
            self._run_inq()
            self._collect_inq()
        except Exception as e:
            # FIXME
            raise e


    def _initialize_inq(self):

        pinq.clear()
        pinq.cell.cubic(5, "Angstrom", periodicity=3)
        pinq.electrons.cutoff(40.0, "Hartree")
        #pinq.electrons.spin_polarized()
        #pinq.ground_state.max_steps(500)
        #pinq.ground_state.mixing(.2)
        #pinq.ground_state.tolerance(1e-6)
        #pinq.kpoints.grid(2, 2, 2)
        #pinq.theory.hartree_fock()
        #pinq.theory.pbe0()
        #pinq.theory.b3lyp()


    def _prepare_inq(self):

        self._create_atoms()


    def _create_atoms(self):

        pinq.ions.clear()

        for atom, position in zip(self._data["AtomTypes"], self._data["Positions"]):
            pinq.ions.insert(atom, position, "Angstrom")

        #number_of_atoms = len(self._data["AtomTypes"])
        #n_atoms = int(self._lmp.get_natoms())
        #assert number_of_atoms == n_atoms, "Atom counts don't match when creating atoms: {}, {}\nGroup and configuration: {} {}".format(number_of_atoms, n_atoms, self._data["Group"], self._data["File"])


    def _run_inq(self):

        try:

            pinq.run.ground_state()

        except:

            # FIXME: handle exception


    def _collect_inq(self):

        if self.energy:
            self._data['Energy'] = pinq.results.ground_state.energy.total()

        if self.force:
            self._data['Forces'] = pinq.results.ground_state.forces()

        # Note that the dipole is only calculated for the non-periodic directions.
        # For the periodic directions is set to zero since the dipole is not properly defined.
        #if self.dipole:


    def _extract_atom_positions(self, num_atoms):

        # FIXME: might be needed one day for geometry optimization
        pass

