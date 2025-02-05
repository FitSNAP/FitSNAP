from fitsnap3lib.calculators.calculator import Calculator
from fitsnap3lib.units.units import convert
import pinq, json
import numpy as np
from pprint import pprint

# has to be outside class, cant pass a class
# around with MPICommExecutor
def process_single_task(self, data, i=0):

    inq_calculator.process_single(self, data, i)


class INQ(Calculator):

    def __init__(self, name, pt, config):
        super().__init__(name, pt, config)
        #self.theory = self.config.sections["CALCULATOR"].theory
        self._data = {}
        self._i = 0

        if self.config.sections["REFERENCE"].units == "real":
            self.ha_to_energy_units = convert('energy', 'ha', 'kcalmol')
            self.distance_units = "Angstrom"
        elif self.config.sections["REFERENCE"].units == "metal":
            self.ha_to_energy_units = convert('energy', 'ha', 'ev')
            self.distance_units = "Angstrom"
        else:
            raise NotImplementedError("INQ calculator only supports 'real' or 'metal' units.")


    def process_single(self, data, i=0):
      """
      Calculate QM training data for a given configuration.
      """

      try:
          inq_calculator._data = data
          inq_calculator._i = i
          inq_calculator._initialize_inq()
          inq_calculator._prepare_inq()
          inq_calculator._run_inq()
          inq_calculator._collect_inq()
      except Exception as e:
          # FIXME: handle exception
          raise e

    def process_configs(self, data):

        global inq_calculator
        inq_calculator = self

        if self.pt.stubs == 1:
            #pprint(data)
            for d in data: self.process_single(d)
        else:
            from mpi4py import MPI
            from mpi4py.futures import MPICommExecutor, wait

            with MPICommExecutor(MPI.COMM_WORLD, root=0) as self.executor:
                if self.executor is not None:
                    self.executor.map(process_single_task, data, unordered=True)

        ground_energy = np.min(np.asarray([d["Energy"] for d in data]))
        for d in data: d["Energy"] -= ground_energy


    def _initialize_inq(self):

        pinq.clear()

        if (cell := self.config.sections["INQ"].cell).startswith('cubic'):
            _, side, units, periodicity = cell.split()
            pinq.cell.cubic(5, units, periodicity=periodicity)
            # FIXME: check self.distance_units == units
        else:
            # FIXME: cell orthorhombic and lattice
            pass

        pinq.electrons.cutoff(40.0, "Hartree")
        #pinq.electrons.spin_polarized()
        pinq.ground_state.max_steps(500)
        #pinq.ground_state.mixing(.2)
        pinq.ground_state.tolerance(1e-6)
        #pinq.kpoints.grid(2, 2, 2)

        # gives same results as PBE for N2 example
        #pinq.theory.hartree_fock()
        pinq.theory.pbe0()

        # fancier functionals dont work because INQ dispersion not implemented
        #pinq.theory.b3lyp()


    def _prepare_inq(self):

        self._create_atoms()


    def _create_atoms(self):

        pinq.ions.clear()

        for atom, position in zip(self._data["AtomTypes"], self._data["Positions"]):
            pinq.ions.insert(atom, position, self.distance_units)

        # FIXME: error checking correct number of atoms created
        #number_of_atoms = len(self._data["AtomTypes"])
        #n_atoms = int(self._lmp.get_natoms())
        #assert number_of_atoms == n_atoms, "Atom counts don't match when creating atoms: {}, {}\nGroup and configuration: {} {}".format(number_of_atoms, n_atoms, self._data["Group"], self._data["File"])


    def _run_inq(self):

        try:
            pinq.run.ground_state()
        except:
            # FIXME: handle exception
            pass
            

    def _collect_inq(self):

        if self.energy:
            self._data['Energy'] = self.ha_to_energy_units * pinq.results.ground_state.energy.total()

        if self.force:
            self._data['Forces'] = [list(f) for f in pinq.results.ground_state.forces()]

        if self.dipole:
            self._data['Dipole'] = pinq.results.ground_state.dipole()

        # Note that the dipole is only calculated for the non-periodic directions.
        # For the periodic directions is set to zero since the dipole is not properly defined.
        #if self.dipole:


    def _extract_atom_positions(self, num_atoms):

        # FIXME: might be needed later for geometry optimization
        pass

