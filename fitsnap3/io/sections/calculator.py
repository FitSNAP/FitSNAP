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
        self.chemflag = self.get_value("CALCULATOR", "chemflag", "0", "bool")
        self.bnormflag = self.get_value("CALCULATOR", "bnormflag", "0", "bool")
        self.wselfallflag = self.get_value("CALCULATOR", "wselfallflag", "0", "bool")
        self.bzeroflag = self.get_value("CALCULATOR", "bzeroflag", "0", "bool")
        self.quadraticflag = self.get_value("CALCULATOR", "quadraticflag", "0", "bool")
        self.delete()
