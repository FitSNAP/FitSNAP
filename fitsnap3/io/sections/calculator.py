from fitsnap3.io.sections.sections import Section
from fitsnap3.parallel_tools import pt


class Calculator(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.calculator = self.get_value("CALCULATOR", "calculator", "LAMMPSSNAP")
        self.energy = self.get_value("CALCULATOR", "energy", "True", "bool")
        pt.add_calculator_option("energy", self.energy)
        self.force = self.get_value("CALCULATOR", "force", "True", "bool")
        pt.add_calculator_option("force", self.force)
        self.stress = self.get_value("CALCULATOR", "stress", "True", "bool")
        pt.add_calculator_option("stress", self.stress)
        self.delete()
