from fitsnap3.io.sections.sections import Section
from fitsnap3.parallel_tools import ParallelTools


pt = ParallelTools()


class Calculator(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['calculator', 'energy', 'force', 'stress']
        self._check_section()

        self.calculator = self.get_value("CALCULATOR", "calculator", "LAMMPSSNAP")
        self.energy = self.get_value("CALCULATOR", "energy", "True", "bool")
        pt.add_2_fitsnap("energy", self.energy)
        self.force = self.get_value("CALCULATOR", "force", "True", "bool")
        pt.add_2_fitsnap("force", self.force)
        self.stress = self.get_value("CALCULATOR", "stress", "True", "bool")
        pt.add_2_fitsnap("stress", self.stress)
        self.delete()
