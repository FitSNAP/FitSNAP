#import ctypes
from fitsnap3lib.calculators.calculator import Calculator
import numpy as np

#import pinq


class InqCalculator(Calculator):

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

        # FIXME
        pass


    def _prepare_inq(self):

        # FIXME
        pass


    def _create_atoms(self):

        # FIXME
        number_of_atoms = len(self._data["AtomTypes"])
        positions = self._data["Positions"].flatten()
        #elem_all = [type_mapping[a_t] for a_t in self._data["AtomTypes"]]
        #n_atoms = int(self._lmp.get_natoms())
        #assert number_of_atoms == n_atoms, "Atom counts don't match when creating atoms: {}, {}\nGroup and configuration: {} {}".format(number_of_atoms, n_atoms, self._data["Group"], self._data["File"])


    def _collect_inq(self):

        # FIXME
        pass


    def _run_inq(self):

        # FIXME
        pass


    def _extract_atom_positions(self, num_atoms):

        # FIXME: might be needed one day for geometry optimization
        pass

