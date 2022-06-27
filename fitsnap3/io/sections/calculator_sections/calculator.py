from ..sections import Section
from ....parallel_tools import pt


class Calculator(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['calculator', 'energy', 'per_atom_energy', 'force', 'stress']
        self._check_section()

        self.calculator = self.get_value("CALCULATOR", "calculator", "LAMMPSSNAP")
        self.energy = self.get_value("CALCULATOR", "energy", "True", "bool")
        if self.energy:
            self.per_atom_energy = self.get_value("CALCULATOR", "per_atom_energy", "False", "bool")
        self.dee = self.check_path(self.get_value("CALCULATOR", "dee", "detailed_energy_errors.dat"))
        pt.add_2_fitsnap("energy", self.energy)
        pt.add_2_fitsnap("per_atom_energy", self.per_atom_energy)
        self.force = self.get_value("CALCULATOR", "force", "True", "bool")
        self.dfe = self.check_path(self.get_value("CALCULATOR", "dfe", "detailed_force_errors.dat"))
        pt.add_2_fitsnap("force", self.force)
        self.stress = self.get_value("CALCULATOR", "stress", "True", "bool")
        self.dse = self.check_path(self.get_value("CALCULATOR", "dee", "detailed_stress_errors.dat"))
        pt.add_2_fitsnap("stress", self.stress)
        self.delete()
